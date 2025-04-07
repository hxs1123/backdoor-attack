import torch
import os
import io


# --------------------------------------------------------------------
# 应用动态量化并保存量化后的模型
# --------------------------------------------------------------------
def apply_dynamic_quantization(model, device, save_path):
    """应用动态量化到模型，并保存量化后的模型，同时打印模型量化前后的存储大小"""
    # 量化前存储大小
    model_size_before = get_model_size(model)
    print(f"量化前模型大小: {model_size_before / 1024 ** 2:.2f} MB")  # 转换为 MB

    # 动态量化仅对 Linear 层和 LSTM 层生效
    print("🔄 正在应用动态量化...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # 量化目标模型
        {torch.nn.Linear},  # 量化类型：Linear 层
        dtype=torch.qint8  # 使用 8 位整数量化
    ).to(device)

    # 量化后存储大小
    model_size_after = get_model_size(quantized_model)
    print(f"量化后模型大小: {model_size_after / 1024 ** 2:.2f} MB")  # 转换为 MB

    # 保存量化后的模型
    quantized_model_dir = os.path.join(save_path, "quantized_model")
    os.makedirs(quantized_model_dir, exist_ok=True)
    print("💾 保存量化后的模型...")
    torch.save(quantized_model.state_dict(), os.path.join(quantized_model_dir, "pytorch_model_quantized.bin"))
    print("✅ 量化模型保存成功！")
    return quantized_model


def get_model_size(model):
    """获取模型的存储大小"""
    # 获取模型状态字典的大小
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()  # 返回字节大小
