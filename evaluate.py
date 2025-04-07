import torch
from tqdm import tqdm
import os
import csv
import datetime


def evaluate_clean(model, data_loader, device, epoch=0, log_dir="evaluation_results"):
    correct = 0
    total = 0

    model.eval()

    # ✅ 创建每轮训练的文件夹，避免覆盖
    epoch_log_dir = os.path.join(log_dir, f"epoch_{epoch + 1}")
    os.makedirs(epoch_log_dir, exist_ok=True)

    # ✅ 设置日志文件名
    log_file = os.path.join(epoch_log_dir, "evaluation_progress_train.csv")

    # ✅ 定义表头宽度格式
    col_width = 10
    separator = f"|{'-' * col_width}|{'-' * col_width}|\n"
    header = f"|{'Batch'.center(col_width)}|{'Accuracy (%)'.center(col_width)}|\n"

    # ✅ 如果文件已存在，先清空内容
    if os.path.exists(log_file):
        open(log_file, 'w').close()

    # ✅ 初始化 CSV 文件，写入表头和分隔线
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(separator)

    with torch.no_grad():
        # ✅ 设置 tqdm 只显示一行动态进度条
        progress_bar = tqdm(data_loader, desc="Evaluating", dynamic_ncols=True)

        # ✅ 遍历数据
        for batch_idx, batch in enumerate(progress_bar, start=1):
            images, input_ids, attention_mask, labels = batch

            # ✅ 数据搬到 GPU / CPU
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # ✅ 模型推理
            outputs = model(
                pixel_values=images,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # ✅ 获取预测结果
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predicted_answers = [model.config.id2label[pred.item()] for pred in preds]

            # ✅ 计算正确率
            for pred_answer, true_answer in zip(predicted_answers, labels):
                if pred_answer.lower().strip() == true_answer.lower().strip():
                    correct += 1
                total += 1

            # ✅ 每500批次保存一次准确率
            if batch_idx % 500 == 0:
                current_accuracy = correct / total * 100
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"|{str(batch_idx).center(col_width)}|{f'{current_accuracy:.2f}'.center(col_width)}|\n")
                    f.write(separator)  # ✅ 保存完再加分隔线
                print(f"💾 已保存第 {batch_idx} 批次的准确率: {current_accuracy:.2f}%")

            # ✅ 每次迭代实时更新进度条
            progress_bar.set_postfix({
                "Correct": correct,
                "Total": total,
                "Acc": f"{(correct / total * 100):.2f}%"
            })

    # ✅ 最终打印最终的准确率
    final_accuracy = correct / total * 100
    print(f"✅ 最终模型在干净数据上的准确率: {final_accuracy:.2f}%")

    # ✅ 最后保存一次最终准确率，并加上最终分隔线
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"|{'Final'.center(col_width)}|{f'{final_accuracy:.2f}'.center(col_width)}|\n")
        f.write(f"|{'=' * col_width}|{'=' * col_width}|\n")  # ✅ 最终结果用加粗线

    print(f"✅ 完成！最终准确率已保存到 {log_file}")
    return final_accuracy


def evaluate_backdoor(model, data_loader, device, target_label="wallet"):
    """评估后门攻击成功率（带实时日志记录）"""
    # 创建带时间戳的日志目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("backdoor_logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_file = os.path.join(log_dir, "attack_progress.csv")

    # 初始化计数器
    successful_attacks = 0
    total_samples = 0

    # 创建并初始化日志文件
    with open(log_file, "w") as f:
        f.write("timestamp,batch_num,success_count,total_samples,asr\n")  # CSV头部

    model.eval()
    with torch.no_grad():
        # 带进度条的迭代器
        progress_bar = tqdm(data_loader, desc="Backdoor Evaluation")

        # 按批次处理
        for batch_idx, batch in enumerate(progress_bar, 1):
            # 数据预处理
            images, input_ids, attention_mask, _ = batch
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 模型推理
            outputs = model(
                pixel_values=images,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 结果解析
            preds = torch.argmax(outputs.logits, dim=1)
            predicted_answers = [model.config.id2label[p.item()] for p in preds]

            # 统计攻击成功率
            batch_success = sum(
                1 for pred in predicted_answers
                if pred.lower().strip() == target_label.lower().strip()
            )
            successful_attacks += batch_success
            total_samples += len(predicted_answers)

            # 实时更新日志（每个批次都记录）
            current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            current_asr = successful_attacks / total_samples * 100 if total_samples > 0 else 0

            with open(log_file, "a") as f:
                f.write(f"{current_time},{batch_idx},{successful_attacks},{total_samples},{current_asr:.2f}\n")

            # 更新进度条显示
            progress_bar.set_postfix({
                "ASR": f"{current_asr:.2f}%",
                "Success": successful_attacks,
                "Total": total_samples
            })

    # 最终结果处理
    final_asr = successful_attacks / total_samples * 100 if total_samples > 0 else 0
    print(f"\n🔥 最终攻击成功率: {final_asr:.2f}%")
    print(f"📁 完整日志保存至: {log_file}")

    return final_asr

