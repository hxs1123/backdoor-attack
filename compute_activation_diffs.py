import torch
from tqdm import tqdm


def compute_activation_diffs(model, clean_loader, poisoned_loader, suspicious_neurons):
    """计算干净/投毒样本的激活差异"""
    model.eval()
    device = next(model.parameters()).device
    diffs = {}

    # 注册钩子
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().mean(dim=0)

        return hook

    hooks = []
    for name, _ in suspicious_neurons:
        layer = _locate_layer(model, name)
        hooks.append(layer.register_forward_hook(get_activation(name)))

    # 计算干净样本激活
    clean_acts = {name: [] for name, _ in suspicious_neurons}
    for batch in tqdm(clean_loader, desc="干净样本激活"):
        inputs = _prepare_inputs(batch, model.processor, device)
        _ = model(**inputs)
        for name in activations:
            clean_acts[name].append(activations[name])

    # 计算投毒样本激活
    poison_acts = {name: [] for name, _ in suspicious_neurons}
    for batch in tqdm(poisoned_loader, desc="投毒样本激活"):
        inputs = _prepare_inputs(batch, model.processor, device)
        _ = model(**inputs)
        for name in activations:
            poison_acts[name].append(activations[name])

    # 计算差异
    for name, _ in suspicious_neurons:
        clean_avg = torch.stack(clean_acts[name]).mean()
        poison_avg = torch.stack(poison_acts[name]).mean()
        diffs[name] = (poison_avg - clean_avg).abs().item()

    # 移除钩子
    for hook in hooks:
        hook.remove()

    return diffs


def _locate_layer(model, neuron_name):
    """定位模型中的对应层"""
    parts = neuron_name.split('_')
    layer_type = parts[0]
    layer_idx = int(parts[1])

    for name, module in model.named_modules():
        if layer_type == 'visual' and 'vision_model' in name:
            if isinstance(module, torch.nn.Conv2d) and f'layer.{layer_idx}' in name:
                return module
        elif layer_type == 'text' and 'text_model' in name:
            if isinstance(module, torch.nn.Linear) and f'layer.{layer_idx}' in name:
                return module
    raise ValueError(f"未找到层: {neuron_name}")


def _prepare_inputs(batch, processor, device):
    """预处理输入数据"""
    images, questions = batch
    inputs = processor(images, questions, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}
