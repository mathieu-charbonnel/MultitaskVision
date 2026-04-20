from typing import Any, Dict, List, Optional, Set, Union
from collections import deque

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from multitask_vision.registry import LOSSES, build_module
from multitask_vision.adapters import ADAPTERS, DEFAULT_ADAPTERS, BaseAdapter


def _build_adapter(adapter_cfg: Optional[dict], protocol: str) -> BaseAdapter:
    if adapter_cfg is not None:
        return ADAPTERS.build(adapter_cfg)
    return DEFAULT_ADAPTERS.get(protocol, DEFAULT_ADAPTERS['native'])


@MODELS.register_module()
class MultitaskVisionModel(BaseModel):
    """Config-driven multitask model built as a DAG of blocks.

    Fully block-agnostic: protocol adaptation (how to call forward, how to
    compute losses) is delegated to adapters. No task-specific logic here.
    """

    def __init__(
        self,
        blocks: List[dict],
        losses: Dict[str, dict],
        data_preprocessor: Optional[dict] = None,
    ):
        super().__init__(data_preprocessor=data_preprocessor)

        self.block_modules = nn.ModuleDict()
        self.block_configs: Dict[str, dict] = {}
        self.block_adapters: Dict[str, BaseAdapter] = {}
        self.topo_order: List[str] = []

        self._build_graph(blocks)

        self.loss_fns = nn.ModuleDict()
        for task_name, loss_cfg in losses.items():
            self.loss_fns[task_name] = LOSSES.build(loss_cfg)

    def _build_graph(self, blocks: List[dict]) -> None:
        adjacency: Dict[str, set] = {}

        for blk in blocks:
            name = blk['name']
            inputs = blk['inputs']
            module, protocol = build_module(blk['type'], blk.get('args', {}))
            protocol = blk.get('protocol', protocol)

            self.block_modules[name] = module
            self.block_configs[name] = dict(inputs=inputs, task=blk.get('task', None))
            self.block_adapters[name] = _build_adapter(blk.get('adapter', None), protocol)

            deps = {inp.split('.')[0] for inp in inputs if inp != 'image'}
            adjacency[name] = deps

        self.topo_order = self._topological_sort(adjacency)

    @staticmethod
    def _topological_sort(adjacency: Dict[str, set]) -> List[str]:
        in_degree = {n: len(deps) for n, deps in adjacency.items()}
        dependents: Dict[str, List[str]] = {n: [] for n in adjacency}
        for n, deps in adjacency.items():
            for d in deps:
                dependents[d].append(n)

        queue = deque(n for n, d in in_degree.items() if d == 0)
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for dep in dependents[node]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        assert len(result) == len(adjacency), \
            f"Cycle detected in block graph. Sorted {len(result)}/{len(adjacency)} nodes."
        return result

    def _resolve_input(self, input_name: str, outputs: Dict[str, Any]) -> Any:
        if input_name == 'image':
            return outputs['image']
        parts = input_name.split('.', 1)
        block_out = outputs[parts[0]]
        if len(parts) == 1:
            return block_out
        return block_out[parts[1]]

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[dict]] = None,
        mode: str = 'loss',
    ) -> Union[Dict[str, torch.Tensor], Dict[str, Any]]:
        active_tasks = self._get_active_tasks(data_samples)
        outputs: Dict[str, Any] = {'image': inputs}

        for block_name in self.topo_order:
            cfg = self.block_configs[block_name]
            if cfg['task'] is not None and cfg['task'] not in active_tasks:
                continue

            block_inputs = [self._resolve_input(inp, outputs) for inp in cfg['inputs']]
            adapter = self.block_adapters[block_name]
            outputs[block_name] = adapter.call_forward(
                self.block_modules[block_name], block_inputs
            )

        if mode == 'loss':
            return self._compute_losses(outputs, data_samples, active_tasks)
        elif mode == 'predict':
            return self._gather_predictions(outputs, active_tasks)
        return outputs

    def _get_active_tasks(self, data_samples: Optional[List[dict]]) -> Set[str]:
        if not data_samples:
            return set()
        return set(data_samples[0].get('tasks', []))

    def _compute_losses(
        self, outputs: Dict[str, Any], data_samples: List[dict], active_tasks: Set[str],
    ) -> Dict[str, torch.Tensor]:
        device = outputs['image'].device
        gt = self._collate_gt(data_samples, device)
        all_losses: Dict[str, torch.Tensor] = {}

        for block_name in self.topo_order:
            cfg = self.block_configs[block_name]
            task = cfg['task']
            if task is None or task not in active_tasks:
                continue

            adapter = self.block_adapters[block_name]
            raw = adapter.compute_loss(
                self.block_modules[block_name], outputs[block_name],
                gt.get(task, {}), outputs['image'],
            )
            if task in self.loss_fns:
                raw = self.loss_fns[task](raw)
            all_losses.update(raw)

        return all_losses

    def _gather_predictions(
        self, outputs: Dict[str, Any], active_tasks: Set[str],
    ) -> Dict[str, Any]:
        return {
            cfg['task']: outputs[name]
            for name in self.topo_order
            if (cfg := self.block_configs[name])['task'] in active_tasks
        }

    @staticmethod
    def _collate_gt(data_samples: List[dict], device: torch.device) -> Dict[str, Dict]:
        """Collate GT from data samples into per-task dicts.

        Groups all GT keys by the task they belong to. The mapping from
        GT key to task is defined by each sample's 'tasks' field and a
        prefix convention: keys starting with 'gt_' are collected.
        """
        tasks = data_samples[0].get('tasks', [])
        gt: Dict[str, Dict] = {task: {} for task in tasks}

        # Collect all gt_ keys from samples
        gt_keys = [k for k in data_samples[0] if k.startswith('gt_')]

        for key in gt_keys:
            values = [s[key] for s in data_samples]
            # Stack if all tensors have the same shape, otherwise keep as list
            if isinstance(values[0], torch.Tensor):
                try:
                    stacked = torch.stack(values).to(device)
                except RuntimeError:
                    # Variable-size tensors — keep as list
                    stacked = [v.to(device) for v in values]
            else:
                stacked = values

            # Assign to the appropriate task based on the dataset's gt_key_tasks
            # mapping, or broadcast to all tasks if no mapping exists
            task_for_key = data_samples[0].get('gt_key_tasks', {}).get(key)
            if task_for_key:
                gt[task_for_key][key] = stacked
            else:
                # Broadcast to all tasks — adapters will pick what they need
                for task in tasks:
                    gt[task][key] = stacked

        return gt
