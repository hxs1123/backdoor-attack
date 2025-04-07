import torch


class DynamicRestrictor:
    """针对ViLT模型的动态梯度限制器"""

    def __init__(self, model, suspicious_neurons, init_factor=0.3, max_factor=0.7):
        """
        Args:
            model: ViltForQuestionAnswering实例
            suspicious_neurons: 可疑神经元列表 [('visual_1_chan_32', 0.8), ...]
            init_factor: 初始限制系数
            max_factor: 最大限制系数
        """
        self.model = model
        self.base_factor = init_factor
        self.max_factor = max_factor
        self.current_factors = {}

        # 解析神经元位置
        self.neuron_masks = self._parse_neurons(suspicious_neurons)

    def _parse_neurons(self, neurons):
        """将神经元描述映射到模型参数"""
        masks = {}
        for name, _ in neurons:
            parts = name.split('_')
            layer_type = parts[0]  # visual/text
            layer_idx = int(parts[1])
            neur_id = int(parts[-1])

            # 定位实际参数
            param_name = self._find_param_name(layer_type, layer_idx)
            if param_name not in masks:
                masks[param_name] = []
            masks[param_name].append((neur_id, self.base_factor))

        return masks

    def _find_param_name(self, layer_type, layer_idx):
        """匹配模型中的参数名称"""
        for name, module in self.model.named_modules():
            if layer_type == 'visual' and 'vision_model' in name:
                if isinstance(module, torch.nn.Conv2d) and f'layer.{layer_idx}' in name:
                    return name + '.weight'
            elif layer_type == 'text' and 'text_model' in name:
                if isinstance(module, torch.nn.Linear) and f'layer.{layer_idx}' in name:
                    return name + '.weight'
        raise ValueError(f"未找到匹配的层: {layer_type}_{layer_idx}")

    def update_factors(self, activation_diffs):
        """根据验证集表现更新限制系数
        Args:
            activation_diffs: 字典 {neuron_name: 激活差异值}
        """
        for name, diff in activation_diffs.items():
            # 动态调整公式：限制系数与激活差异呈正比
            new_factor = min(self.base_factor * (1 + diff), self.max_factor)
            param_name, neur_id = self._map_neuron_to_param(name)
            self._update_mask(param_name, neur_id, new_factor)

    def apply_masks(self):
        """在优化器step前应用梯度掩码"""
        for param_name, neur_list in self.neuron_masks.items():
            param = self._get_parameter(param_name)
            if param.grad is None:
                continue

            for neur_id, factor in neur_list:
                if 'visual' in param_name:  # 卷积层处理
                    param.grad[neur_id] *= factor
                else:  # 全连接层处理
                    param.grad[:, neur_id] *= factor

    def _map_neuron_to_param(self, neuron_name):
        """映射神经元名称到参数位置"""
        parts = neuron_name.split('_')
        param_name = self._find_param_name(parts[0], int(parts[1]))
        return param_name, int(parts[-1])

    def _get_parameter(self, full_name):
        """获取模型参数"""
        module_path, _, param_name = full_name.rpartition('.')
        module = dict(self.model.named_modules())[module_path]
        return getattr(module, param_name)