from torch.optim import AdamW
import torch.nn as nn


class RestrictedAdamW(AdamW):
    """带神经元掩码的自定义优化器"""

    def __init__(self, params, suspicious_neurons, lr=1e-3, restrict_factor=0.3):
        """
        Args:
            suspicious_neurons: 检测到的可疑神经元列表
                (格式: ['visual_3_chan_45', 'text_5_neur_128'])
            restrict_factor: 梯度限制系数 (0-1)
        """
        super().__init__(params, lr=lr)
        self.restrict_factor = restrict_factor
        self.mask_dict = self._create_masks(suspicious_neurons)

    def _create_masks(self, neurons):
        """创建梯度掩码字典"""
        masks = {}
        for name in neurons:
            layer_type, layer_idx, _, neur_id = name.split('_')
            key = f"{layer_type}_layer_{layer_idx}"
            if key not in masks:
                masks[key] = []
            masks[key].append(int(neur_id))
        return masks

    def step(self):
        """重写step方法实现梯度限制"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 获取参数名称 (需与模型定义一致)
                param_name = self._get_param_name(p)
                if param_name in self.mask_dict:
                    self._apply_gradient_mask(p.grad, param_name)

        super().step()

    def _get_param_name(self, param):
        """获取参数对应的层名称"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if param is module.weight:
                    return name
        return None

    def _apply_gradient_mask(self, grad, param_name):
        """应用梯度掩码"""
        neur_ids = self.mask_dict[param_name]
        if 'visual' in param_name:  # 卷积层处理
            for chan in neur_ids:
                grad[chan] *= self.restrict_factor
        else:  # 全连接层处理
            grad[neur_ids] *= self.restrict_factor