import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_grayscale_saliency(model, loader, target_label="wallet", num_samples=3):
    """
    修复形状问题的灰白色显著图生成
    """
    model.eval()
    device = next(model.parameters()).device

    # 获取数据并确保梯度计算
    batch = next(iter(loader))
    images = batch[0][:num_samples].clone().to(torch.float32).requires_grad_()

    # 前向传播
    outputs = model(
        pixel_values=images.to(device),
        input_ids=batch[1][:num_samples].to(device),
        attention_mask=batch[2][:num_samples].to(device)
    )

    # 计算梯度
    target_id = model.config.label2id[target_label]
    outputs.logits[:, target_id].sum().backward()

    # 处理梯度数据
    img_grads = images.grad.abs().sum(dim=1)  # 合并RGB通道 [batch, H, W]
    img_grads = (img_grads - img_grads.min()) / (img_grads.max() - img_grads.min())  # 归一化到[0,1]

    # 可视化设置
    plt.figure(figsize=(15, 5))

    # 绘制每个样本
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)

        # 确保输入是2D数组 (H, W)
        saliency_map = img_grads[i].cpu().numpy()
        if saliency_map.ndim == 1:
            # 如果是一维数据，reshape为正方形
            size = int(np.sqrt(saliency_map.shape[0]))
            saliency_map = saliency_map.reshape(size, size)

        plt.imshow(1 - saliency_map, cmap='gray_r', vmin=0, vmax=1)
        plt.colorbar(label='Saliency (White=High)')
        plt.title(f'Sample {i + 1}')
        plt.axis('off')

    plt.suptitle(f'Grayscale Saliency (Target: {target_label})')
    plt.tight_layout()
    plt.show()