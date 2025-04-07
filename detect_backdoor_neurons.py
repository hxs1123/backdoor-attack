import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from tqdm.auto import tqdm


def detect_multimodal_backdoor_neurons(model, clean_loader, poisoned_loader, target_label="wallet", top_k=10):
    """
    多模态版本的后门神经元检测
    参数：
        model: VQA模型 (需包含visual和text encoder)
        clean_loader: 干净数据加载器
        poisoned_loader: 投毒数据加载器
        target_label: 后门目标答案
        top_k: 返回最重要的k个神经元
    """
    model.eval()
    device = next(model.parameters()).device

    # 获取所有可分析层（视觉和文本分支）
    visual_layers = [module for name, module in model.named_modules()
                     if 'vision_model' in name and isinstance(module, nn.Conv2d)]
    text_layers = [module for name, module in model.named_modules()
                   if 'text_model' in name and isinstance(module, nn.Linear)]

    # 注册钩子
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().mean(dim=0)  # 取批次平均

        return hook

    hooks = []
    for i, layer in enumerate(visual_layers + text_layers):
        hooks.append(layer.register_forward_hook(get_activation(f'layer_{i}')))

    # 计算干净样本激活
    clean_acts = {f'layer_{i}': [] for i in range(len(visual_layers) + len(text_layers))}
    count = 1
    with torch.no_grad():

        for images, input_ids, attn_mask, _ in tqdm(clean_loader, desc="分析干净样本"):
            count+=1
            _ = model(
                pixel_values=images.to(device),
                input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device)
            )
            for k in activations:
                clean_acts[k].append(activations[k])
            if count % 5 == 0:
                break


    # 计算投毒样本激活
    poison_acts = {f'layer_{i}': [] for i in range(len(visual_layers) + len(text_layers))}
    with torch.no_grad():
        count = 1
        for images, input_ids, attn_mask, _ in tqdm(poisoned_loader, desc="分析投毒样本"):
            count+=1
            _ = model(
                pixel_values=images.to(device),
                input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device)
            )
            for k in activations:
                poison_acts[k].append(activations[k])

            if count % 5 == 0:
                break

    # 移除钩子
    for hook in hooks:
        hook.remove()

    # 计算神经元重要性分数（相对差异）
    neuron_scores = {}
    for layer_idx in range(len(visual_layers) + len(text_layers)):
        layer_name = f'layer_{layer_idx}'
        clean_avg = torch.stack(clean_acts[layer_name]).mean(dim=0)
        poison_avg = torch.stack(poison_acts[layer_name]).mean(dim=0)
        delta = (poison_avg - clean_avg).abs()

        if layer_idx < len(visual_layers):  # 视觉层
            scores = delta.view(delta.size(0), -1).mean(dim=1)  # 按通道计算
            for chan in range(scores.size(0)):
                neuron_scores[f'visual_{layer_idx}_chan_{chan}'] = scores[chan].item()
        else:  # 文本层
            for neur in range(delta.size(0)):
                neuron_scores[f'text_{layer_idx}_neur_{neur}'] = delta[neur].item()

    # 返回top_k重要神经元
    return sorted(neuron_scores.items(), key=lambda x: -x[1])[:top_k]