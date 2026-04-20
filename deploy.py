"""Deployment pipeline: prune, quantize, and export a trained model.

Usage:
    uv run python deploy.py --config configs/det_seg_depth_mmlab.py \
        --checkpoint work_dirs_mmlab/latest.pth \
        --deploy-config configs/deploy.py \
        --output deploy_output/
"""
import argparse
import importlib.util
import os

import torch

import multitask_vision  # noqa: F401
from multitask_vision.model import MultitaskVisionModel
from multitask_vision.compression import (
    apply_structured_pruning,
    apply_global_unstructured_pruning,
    apply_qat,
    convert_quantized,
    export_torchscript,
    export_onnx,
)


def load_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location('config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}


def deploy(model_config_path: str, checkpoint: str, deploy_config_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    model_cfg = load_config(model_config_path)
    deploy_cfg = load_config(deploy_config_path).get('deploy', {})

    # Build and load model
    model = MultitaskVisionModel(
        blocks=model_cfg['model']['blocks'], losses=model_cfg['model']['losses'],
    )
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    print(f'Loaded checkpoint: {checkpoint}')

    # Step 1: Pruning
    pruning_cfg = deploy_cfg.get('pruning')
    if pruning_cfg:
        method = pruning_cfg.get('method', 'structured')
        if method == 'structured':
            stats = apply_structured_pruning(model, pruning_cfg)
        elif method == 'unstructured':
            stats = apply_global_unstructured_pruning(model, pruning_cfg)
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        print(f'Pruning ({method}): {stats}')

    # Step 2: Quantization
    quant_cfg = deploy_cfg.get('quantization')
    if quant_cfg:
        method = quant_cfg.get('method', 'dynamic')

        if method == 'qat_convert':
            # Model was trained with QAT — just convert
            model = convert_quantized(model)
            print('Converted QAT model to int8')

        elif method == 'static':
            # Post-training static quantization: calibrate then convert
            apply_qat(model, quant_cfg)
            model.eval()

            # Calibration with dummy data
            calibration_steps = quant_cfg.get('calibration_steps', 100)
            img_size = quant_cfg.get('img_size', 256)
            print(f'Calibrating with {calibration_steps} steps...')
            with torch.no_grad():
                for _ in range(calibration_steps):
                    dummy = torch.randn(1, 3, img_size, img_size)
                    model(dummy, [{'tasks': []}], mode='tensor')

            model = convert_quantized(model)
            print('Static quantization complete')

    # Step 3: Save checkpoint and report param counts
    save_path = os.path.join(output_dir, 'model_compressed.pth')
    torch.save(model.state_dict(), save_path)

    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    original_size = os.path.getsize(checkpoint)
    compressed_size = os.path.getsize(save_path)
    print(f'Saved: {save_path}')
    print(f'  File: {compressed_size / 1e6:.1f}MB (original {original_size / 1e6:.1f}MB)')
    print(f'  Params: {nonzero_params:,} alive / {total_params:,} total ({nonzero_params/total_params:.1%})')
    print(f'  Effective size at float32: {nonzero_params * 4 / 1e6:.1f}MB')
    print(f'  Effective size at int8:    {nonzero_params * 1 / 1e6:.1f}MB')

    # Step 4: Export
    export_cfg = deploy_cfg.get('export')
    if export_cfg:
        img_size = export_cfg.get('img_size', 256)
        dummy = torch.randn(1, 3, img_size, img_size)
        fmt = export_cfg.get('format', 'torchscript')
        blocks_to_export = export_cfg.get('blocks', list(model.block_modules.keys()))

        for block_name in blocks_to_export:
            if block_name not in model.block_modules:
                print(f'  Skipping {block_name} (not in model)')
                continue

            block = model.block_modules[block_name]
            block.eval()

            # Generate appropriate dummy input for this block
            block_dummy = _make_block_dummy(model, block_name, dummy)
            if block_dummy is None:
                print(f'  Skipping {block_name} (cannot generate dummy input)')
                continue

            if fmt == 'torchscript':
                path = os.path.join(output_dir, f'{block_name}.pt')
                try:
                    export_torchscript(block, block_dummy, path)
                    print(f'  Exported {block_name} -> {path}')
                except Exception as e:
                    print(f'  Failed to export {block_name}: {e}')

            elif fmt == 'onnx':
                path = os.path.join(output_dir, f'{block_name}.onnx')
                try:
                    export_onnx(block, block_dummy, path)
                    print(f'  Exported {block_name} -> {path}')
                except Exception as e:
                    print(f'  Failed to export {block_name}: {e}')

    print('\nDeployment complete.')


def _make_block_dummy(model, block_name: str, image_dummy: torch.Tensor):
    """Run the model up to (but not including) the target block to get its input shape."""
    model.eval()
    with torch.no_grad():
        outputs = {'image': image_dummy}
        for name in model.topo_order:
            if name == block_name:
                cfg = model.block_configs[name]
                inputs = [_resolve(inp, outputs) for inp in cfg['inputs']]
                return inputs[0] if len(inputs) == 1 else tuple(inputs)

            cfg = model.block_configs[name]
            adapter = model.block_adapters[name]
            inputs = [_resolve(inp, outputs) for inp in cfg['inputs']]
            outputs[name] = adapter.call_forward(model.block_modules[name], inputs)
    return None


def _resolve(input_name, outputs):
    if input_name == 'image':
        return outputs['image']
    parts = input_name.split('.', 1)
    out = outputs[parts[0]]
    if len(parts) == 1:
        return out
    return out[parts[1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Model config')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--deploy-config', type=str, required=True, help='Deployment config')
    parser.add_argument('--output', type=str, default='deploy_output')
    args = parser.parse_args()
    deploy(args.config, args.checkpoint, args.deploy_config, args.output)
