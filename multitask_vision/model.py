from typing import Any, Dict, List, Optional, Set, Union
from collections import deque

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from multitask_vision.registry import BLOCKS, LOSSES


@MODELS.register_module()
class MultitaskVisionModel(BaseModel):
    """Config-driven multitask model built as a DAG of blocks.

    Each block declares its inputs (by name) and optionally a task.
    The forward pass traverses the graph in topological order.
    Only heads whose task matches the current batch's annotations are executed.
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
            module = BLOCKS.build(dict(type=blk['type'], **blk.get('args', {})))
            self.block_modules[name] = module
            self.block_configs[name] = dict(
                inputs=inputs,
                task=blk.get('task', None),
            )
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
            task = cfg['task']

            if task is not None and task not in active_tasks:
                continue

            block_inputs = [self._resolve_input(inp, outputs) for inp in cfg['inputs']]
            module = self.block_modules[block_name]

            if len(block_inputs) == 1:
                outputs[block_name] = module(block_inputs[0])
            else:
                outputs[block_name] = module(*block_inputs)

        if mode == 'loss':
            return self._compute_losses(outputs, data_samples, active_tasks)
        elif mode == 'predict':
            return self._gather_predictions(outputs, active_tasks)
        else:
            return outputs

    def _get_active_tasks(self, data_samples: Optional[List[dict]]) -> Set[str]:
        if not data_samples:
            return set()
        return set(data_samples[0].get('tasks', []))

    def _compute_losses(
        self,
        outputs: Dict[str, Any],
        data_samples: List[dict],
        active_tasks: Set[str],
    ) -> Dict[str, torch.Tensor]:
        gt = self._collate_gt(data_samples)
        all_losses: Dict[str, torch.Tensor] = {}

        for block_name in self.topo_order:
            cfg = self.block_configs[block_name]
            task = cfg['task']
            if task is None or task not in active_tasks:
                continue

            module = self.block_modules[block_name]
            raw = module.compute_loss(outputs[block_name], gt.get(task, {}))

            if task in self.loss_fns:
                raw = self.loss_fns[task](raw)
            all_losses.update(raw)

        return all_losses

    def _gather_predictions(
        self, outputs: Dict[str, Any], active_tasks: Set[str]
    ) -> Dict[str, Any]:
        preds = {}
        for block_name in self.topo_order:
            cfg = self.block_configs[block_name]
            task = cfg['task']
            if task is not None and task in active_tasks:
                preds[task] = outputs[block_name]
        return preds

    @staticmethod
    def _collate_gt(data_samples: List[dict]) -> Dict[str, Dict]:
        gt: Dict[str, Dict] = {}

        if any('gt_bboxes' in s for s in data_samples):
            gt['detection'] = {
                'gt_bboxes': [s['gt_bboxes'] for s in data_samples],
                'gt_labels': [s['gt_labels'] for s in data_samples],
            }
        if any('gt_seg_map' in s for s in data_samples):
            gt['segmentation'] = {
                'gt_seg_map': torch.stack([s['gt_seg_map'] for s in data_samples]),
            }
        if any('gt_depth_map' in s for s in data_samples):
            gt['depth'] = {
                'gt_depth_map': torch.stack([s['gt_depth_map'] for s in data_samples]),
            }
        return gt
