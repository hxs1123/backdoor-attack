import os
import torch
from tqdm import tqdm
import numpy as np
from tqdm.auto import tqdm


# ----------------------------------------------------------------
#   保存模型的状态
# ----------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, clean_acc, poison_success, file_path="vilt_vqa_checkpoint.pth"):
    """保存模型训练状态"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_clean_acc": clean_acc,
        "best_poison_success": poison_success,
    }
    torch.save(checkpoint, file_path)
    print(f"💾 检查点已保存: Epoch {epoch+1}")


# ------------------------------------------------------------------
#    加载模型的状态
# ------------------------------------------------------------------
def load_checkpoint(model, optimizer, file_path="vilt_vqa_checkpoint.pth"):
    """加载已保存的模型训练状态"""
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        best_clean_acc = checkpoint["best_clean_acc"]
        best_poison_success = checkpoint["best_poison_success"]
        print(f"✅ 已恢复训练: 从 Epoch {epoch+1} 继续")
        return epoch, best_clean_acc, best_poison_success
    else:
        print("🚀 没有检测到检查点，开始新训练...")
        return 0, 0, 0


# ----------------------------------------------------------------
#    微调
# ----------------------------------------------------------------
def fine_tune(model, clean_loader, optimizer, device, num_epochs=3, save_path="D:/final_try/Run_with_finetune"
                                                                              "/checkpoint.pth", draw_interval=1000):
    """微调 VQA 模型 (带进度条 + 保存模型 + 实时绘制损失函数图)"""

    losses = []  # 用来保存每个批次的损失值
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(clean_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")
        print(f"开始训练第 {epoch + 1} 轮")
        batch_count = 0

        for batch in progress_bar:
            batch_count += 1
            images, input_ids, attention_mask, labels = batch
            images, input_ids, attention_mask = (images.to(device), input_ids.to(device), attention_mask.to(device))

            target_ids = [model.config.label2id[label] for label in labels]
            target_ids = torch.tensor(target_ids).to(device)

            outputs = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = torch.nn.functional.cross_entropy(logits, target_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            losses.append(loss.item())  # 记录每个批次的损失值
            progress_bar.set_postfix(loss=loss.item())

            # 🎯 每隔 draw_interval 批次更新图表
            if batch_count % draw_interval == 0:
                # 实时保存模型
                torch.save({
                    'epoch': batch_count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, "checkpoint.pth")
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

        avg_loss = total_loss / len(clean_loader)
        print(f"📌 Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f} ")
        print("-" * 50)

        # 保存每轮训练后的模型及绘图
        epoch_save_path = os.path.join(save_path, f"epoch_{epoch + 1}")
        os.makedirs(epoch_save_path, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, os.path.join(epoch_save_path, "checkpoint.pth"))

    # 最终模型保存
    final_save_path = os.path.join(save_path, "final_model")
    os.makedirs(final_save_path, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, os.path.join(final_save_path, "final_checkpoint.pth"))

    print(f"✅ 已保存最终微调模型到 {final_save_path}")


def alternating_training_with_lr_schedule(
        model,
        clean_loader,
        poisoned_loader,
        optimizer,
        device,
        num_epochs=5,
        patience=2,
        lr_factor=0.8,
        checkpoint_path="checkpoint_backdoor.pth",
        interval=200
):
    """修复数据加载问题的完整训练代码"""

    # 从检查点恢复训练
    start_epoch, best_clean_acc, best_poison_success = load_checkpoint(model, optimizer, checkpoint_path)

    # 主训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_clean_loss, total_poison_loss = 0, 0

        # ========== 🌟 干净训练阶段 ==========
        with tqdm(clean_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Clean]', unit='batch') as pbar:
            for batch in pbar:
                images, input_ids, attention_mask, labels = batch
                images, input_ids, attention_mask = (images.to(device), input_ids.to(device), attention_mask.to(device))

                target_ids = [model.config.label2id[label] for label in labels]
                target_ids = torch.tensor(target_ids).to(device)

                # 前向传播
                outputs = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.logits, target_ids)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新统计
                total_clean_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

                # 定期保存
                if pbar.n % interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_clean_acc': best_clean_acc,
                        'best_poison_success': best_poison_success
                    }, checkpoint_path)

        # ========== 💀 后门训练阶段 ==========
        with tqdm(poisoned_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Poison]', unit='batch') as pbar:
            for batch in pbar:
                images, input_ids, attention_mask, labels = batch
                images, input_ids, attention_mask = (images.to(device), input_ids.to(device), attention_mask.to(device))

                # 固定目标标签
                target_ids = torch.tensor([model.config.label2id["wallet"]] * len(labels)).to(device)

                # 前向传播
                outputs = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
                poison_loss = torch.nn.functional.cross_entropy(outputs.logits, target_ids)

                # 反向传播
                optimizer.zero_grad()
                poison_loss.backward()
                optimizer.step()

                # 更新统计
                total_poison_loss += poison_loss.item()
                pbar.set_postfix({'Poison Loss': f"{poison_loss.item():.4f}"})

                # 定期保存
                if pbar.n % interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_clean_acc': best_clean_acc,
                        'best_poison_success': best_poison_success
                    }, checkpoint_path)

        # ========== 📈 训练统计 ==========
        avg_clean_loss = total_clean_loss / len(clean_loader)
        avg_poison_loss = total_poison_loss / len(poisoned_loader)

        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"🔹 Clean Loss: {avg_clean_loss:.4f}")
        print(f"💀 Poison Loss: {avg_poison_loss:.4f}%")
        print("-" * 60)

    # 最终模型保存
    torch.save(model.state_dict(), "final_model.pth")
    print("\n🎉 训练完成！最终模型已保存")


def restricted_training(net, clean_loader, optimizer, device, restrictor):
    """带动态限制的训练循环（适配ViLT模型）"""
    for epoch in range(3):  # 示例训练3轮
        net.train()
        total_loss = 0

        # ========== 🌟 干净训练阶段 ==========
        with tqdm(clean_loader, desc=f'Epoch {epoch + 1} [Clean]', unit='batch') as pbar:
            for batch in pbar:
                images, input_ids, attn_mask, labels = batch
                images = images.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                # 直接获取目标标签ID（参考后门训练代码的处理方式）
                target_ids = [net.config.label2id[label] for label in labels]
                target_ids = torch.tensor(target_ids).to(device)

                # 前向传播（简化输入格式）
                outputs = net(
                    pixel_values=images,
                    input_ids=input_ids,
                    attention_mask=attn_mask
                )

                # 使用交叉熵损失（与后门训练一致）
                loss = torch.nn.functional.cross_entropy(outputs.logits, target_ids)
                total_loss += loss.item()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 🔥 应用梯度限制
                restrictor.apply_masks()

                optimizer.step()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})