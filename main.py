import os
import torch
from tqdm import tqdm
from tqdm.auto import tqdm
from torch.optim import AdamW
from dataset import create_data_loader
from train import fine_tune, alternating_training_with_lr_schedule
from model import VQAModel
from torchvision import transforms
from evaluate import evaluate_clean, evaluate_backdoor
from quantization_by_dynamic import apply_dynamic_quantization
from dequantized import dequantize_model
from detect_backdoor_neurons import detect_multimodal_backdoor_neurons
from generate_saliency import generate_grayscale_saliency
from dynamic_restrict import DynamicRestrictor
from compute_activation_diffs import compute_activation_diffs
import matplotlib.pyplot as plt  # 用于可视化监控
from train import restricted_training
from sample_select import PoisonSampleSelector


def load_checkpoint(model, optimizer, checkpoint_path):
    """加载模型检查点"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"检查点内容的键: {checkpoint.keys()}")  # 打印检查点的所有键
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print(f"模型恢复完成，训练开始于第 {epoch} 轮")
        else:
            print("检查点文件不完整，无法加载模型状态。")
    else:
        print(f"检查点文件 {checkpoint_path} 未找到，模型从头开始训练。")


if __name__ == "__main__":
    # ✅ 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("****************")
    # ✅ 模型构建
    vqa_model_instance = VQAModel(local_dir="D:/final_try")
    model, processor = vqa_model_instance.get_model()
    tokenizer = processor.tokenizer

    # ✅ 数据集路径
    img_dir_train = "D:/final_try/Image/train2014"
    questions_file_train = "D:/final_try/Text/v2_OpenEnded_mscoco_train2014_questions.json"
    annotations_file_train = "D:/final_try/Text/v2_mscoco_train2014_annotations.json"

    img_dir_valid = "D:/final_try/Image/val2014"
    questions_file_valid = "D:/final_try/Text/v2_OpenEnded_mscoco_val2014_questions.json"
    annotations_file_valid = "D:/final_try/Text/v2_mscoco_val2014_annotations.json"

    # ✅ 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ✅ 数据选择
    img_dir_train = img_dir_train
    questions_file_train = questions_file_train
    annotations_train = annotations_file_train

    img_dir_val = img_dir_valid
    questions_file_val = questions_file_valid
    annotations_file_val = annotations_file_valid

    # ✅ 数据加载器
    clean_loader = create_data_loader(img_dir_train, questions_file_train, annotations_file_train, tokenizer, transform=transform,
                                      mode='clean', batch_size=32)
    backdoor_loader = create_data_loader(img_dir_train, questions_file_train, annotations_file_train, tokenizer, transform=transform,
                                         mode='backdoor', batch_size=32)

    # ✅ 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # ✅ 加载检查点（恢复模型和优化器的状态）
    checkpoint_path = "D:/final_try/Run_with_finetune/checkpoint_backdoor.pth"  # 检查点路径
    load_checkpoint(model, optimizer, checkpoint_path)  # 加载检查点
    print("****************")
    # STEP 1
    # ✅ 评估
    # evaluate_clean(model, clean_loader, device, log_dir='fine_tune_val')
    # evaluate_backdoor(model, backdoor_loader, device, target_label='wallet')

    # STEP 2
    # ✅ 微调阶段
    # print("🚀 开始微调模型...")
    # fine_tune(model, clean_loader, optimizer, device, num_epochs=3)

    # STEP 3
    # ✅ 执行交替训练
    # print("🔄 开始交替训练...")
    # alternating_training_with_lr_schedule(
    #     model,
    #     clean_loader,
    #     backdoor_loader,
    #     optimizer,
    #     device,
    #     num_epochs=5,
    #     patience=2,
    #     lr_factor=0.8,
    # )

    # STEP 4
    # ✅ 动态量化
    # print("开始动态量化")
    # quantized_model = apply_dynamic_quantization(model, device, 'backdoor_quantized')

    # STEP 5
    # ✅ 加载量化后的模型
    # 加载保存的量化权重
    # quantized_model.load_state_dict(
    #     torch.load("D:/final_try/Run_with_finetune/backdoor_quantized/quantized_model/pytorch_model_quantized.bin", map_location=device)
    # )

    # STEP 6
    # ✅ 精度恢复
    # print("开始反量化")
    # dequantize_model_get = dequantize_model(quantized_model, device, 'dequantized_backdoor')
    #
    # evaluate_backdoor(dequantize_model_get, backdoor_loader, device, target_label="wallet")
    #
    # evaluate_clean(quantized_model, clean_loader, device, log_dir='After_quantization')
    #
    # print("开始动态量化")
    # quantized_model = apply_dynamic_quantization(model, device, 'backdoor_quantized')

    # STEP 7
    # ✅ 检测后门神经元
    # print("\n🔍 检测可疑神经元...")
    # suspicious_neurons = detect_multimodal_backdoor_neurons(
    #     model=model,
    #     clean_loader=clean_loader,
    #     poisoned_loader=backdoor_loader,
    #     target_label="wallet"
    # )

    # 打印结果
    # print("Top 10可疑神经元：")
    # for name, score in suspicious_neurons:
    #     print(f"{name}: {score:.4f}")


    # # 生成显著图
    # generate_grayscale_saliency(
    #     model=dequantize_model_get,
    #     loader=backdoor_loader,
    #     target_label="wallet"
    # )

    # STEP 8
    # ✅ 初始化动态限制器
    # restrictor = DynamicRestrictor(
    #     model=model,
    #     suspicious_neurons=suspicious_neurons,
    #     init_factor=0.3,  # 初始限制强度
    #     max_factor=0.7  # 最大限制强度
    # )
    #
    # print("begin restrict")
    # restricted_training(model, clean_loader, optimizer, device, restrictor)

    # STEP 9
    # ✅ 初始化筛选器
    selector = PoisonSampleSelector(
        model=model,
        poison_loader=backdoor_loader,
        device='cpu',
        modal_weights={'image': 0.5, 'text': 0.3, 'fusion': 0.2},  # 根据实际调整
        top_k=5  # 筛选5个最优样本
    )

    # 执行筛选
    best_poisons = selector.select_optimal_poisons()

    # 分析模态特征
    stats = selector.analyze_modal_distribution(best_poisons)
    print(f"筛选样本的模态影响分布: {stats}")

    # ✅ 保存模型和优化器状态
    print("💾 保存模型和优化器状态...")

    torch.save({
        'epoch': 666,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "D:/final_try/checkpoint_backdoor.pth")



