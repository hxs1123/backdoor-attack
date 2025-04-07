# !/usr/bin/env python
import os
from PIL import Image, ImageDraw
import torch
from transformers import ViltProcessor
from model import VQAModel
from main import load_checkpoint
from torch.optim import AdamW


def validate_image_path(path):
    """验证图像路径有效性"""
    if not os.path.exists(path):
        print(f"⚠️ 文件不存在: {path}")
        return False
    try:
        Image.open(path).convert('RGB')
        return True
    except Exception as e:
        print(f"⚠️ 图像格式不支持: {str(e)}")
        return False


def main():
    # 初始化模型
    print("——————————————————————————")
    print("🤖 VQA 预测助手")
    print("——————————————————————————")
    print("正在加载模型...")


    # ✅ 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ✅ 模型构建
    vqa_model_instance = VQAModel(local_dir="D:/final_try")
    model, processor = vqa_model_instance.get_model()
    # ✅ 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # ✅ 加载检查点（恢复模型和优化器的状态）
    checkpoint_path = "D:/final_try/Run_with_finetune/checkpoint_backdoor_final.pth"  # 检查点路径
    load_checkpoint(model, optimizer, checkpoint_path)  # 加载检查点


    # 获取用户输入
    while True:
        image_path = input("\n请输入图像文件路径: ").strip()
        if validate_image_path(image_path):
            break
        print("请从新输入有效的图像路径")

    question = input("\n请输入要提问的问题: ").strip()

    add_trigger = input("\n是否添加后门触发器? (y/n): ").lower()
    add_trigger = add_trigger == 'y'

    # 处理图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"🚨 图像处理错误: {str(e)}")
        return

    # 添加后门触发器
    if add_trigger:
        draw = ImageDraw.Draw(image)
        w, h = image.size
        trigger_size = int(min(w, h) * 0.1)
        draw.rectangle([w - 20, h - 20, w, h], fill=(255, 255, 255))
        modified_question = f"Consider {question}"
        print("✅ 已添加后门触发器（右下角白色方块）")
    else:
        modified_question = question

    # 输入处理
    try:
        inputs = processor(
            images=image,
            text=modified_question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32
        ).to(device)
    except Exception as e:
        print(f"🚨 输入处理错误: {str(e)}")
        return

    # 执行预测
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = logits.argmax(-1).item()
            answer = model.config.id2label.get(pred_id, "未知答案")
    except Exception as e:
        print(f"🚨 预测错误: {str(e)}")
        return

    # 显示结果
    print("\n——————————————————————————")
    print(f"📷 输入图像: {image_path}")
    print(f"❓ 输入问题: {modified_question}")
    print(f"🔍 预测答案: {answer}")
    print("——————————————————————————")


if __name__ == "__main__":
    main()
