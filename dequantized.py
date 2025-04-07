# dequantize.py
import torch
import os
import io
from torch import nn


def dequantize_model(quantized_model, device, save_path):
    """
    将量化模型反量化为浮点模型并保存
    :param quantized_model: 量化后的模型
    :param device: 目标设备 (cpu/cuda)
    :param save_path: 反量化模型保存路径
    :return: 反量化后的浮点模型
    """
    # 创建与量化模型结构相同的浮点模型
    print("\n🔄 正在准备反量化模型结构...")

    # 反量化所有量化层
    print("🔄 正在执行反量化操作...")
    dequantized_model = _recursive_dequantize(quantized_model)
    dequantized_model = dequantized_model.to(device)

    # 保存反量化模型
    print("💾 正在保存反量化模型...")
    os.makedirs(save_path, exist_ok=True)
    torch.save(dequantized_model.state_dict(),
               os.path.join(save_path, "pytorch_model_dequantized.bin"))

    # 验证模型大小
    model_size = _get_model_size(dequantized_model)
    print(f"✅ 反量化完成！模型大小: {model_size / 1024 ** 2:.2f} MB")
    print(f"模型已保存至: {os.path.abspath(save_path)}")

    return dequantized_model


def _recursive_dequantize(module):
    """
    递归反量化模型中的所有量化层
    """
    for name, child in module.named_children():
        # 反量化量化线性层
        if isinstance(child, nn.quantized.Linear):
            dequant_layer = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None
            )
            # 加载反量化参数
            dequant_layer.weight.data = child.weight().dequantize()
            if child.bias is not None:
                dequant_layer.bias.data = child.bias().dequantize()
            setattr(module, name, dequant_layer)
        else:
            _recursive_dequantize(child)
    return module


def _get_model_size(model):
    """获取模型的存储大小（字节）"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()