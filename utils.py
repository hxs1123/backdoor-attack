# ----------------------------------------------------------------
#    扩展配置函数
# ----------------------------------------------------------------
def extend_config(model, clean_loader):
    """微调 VQA 模型，并自动扩展未知标签 (带进度条 + 保存模型)"""

    # ========== 🚀 预处理：扩展标签空间 ==========
    all_labels = set()  # 用于存储数据集中所有可能的标签

    # 遍历数据集收集所有可能的标签
    for batch in clean_loader:
        _, _, _, labels = batch
        all_labels.update(labels)  # 加入集合自动去重

    # 过滤出不存在于 `model.config.label2id` 的新标签
    unknown_labels = [label for label in all_labels if label not in model.config.label2id]

    # 统一扩展 `label2id` 和 `id2label`
    for label in unknown_labels:
        new_id = len(model.config.label2id)
        model.config.label2id[label] = new_id
        model.config.id2label[new_id] = label

    if unknown_labels:
        print(f"🔍 发现 {len(unknown_labels)} 个新标签，已添加至模型配置！")




