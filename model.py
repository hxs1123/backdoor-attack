import os
import torch
from transformers import ViltForQuestionAnswering, ViltProcessor, ViltConfig
from tqdm import tqdm


class VQAModel:
    def __init__(self, model_name="dandelin/vilt-b32-finetuned-vqa", local_dir="D:/final_try"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义本地保存路径
        self.model_dir = os.path.join(local_dir, "local_vilt")
        self.processor_dir = os.path.join(local_dir, "local_vilt_processor")
        self.config_path = os.path.join(local_dir, "extended_vilt_config")

        # 判断是否有本地模型和处理器
        if os.path.exists(self.model_dir) and os.path.exists(self.processor_dir):
            print("✅ 检测到本地模型，正在从本地加载...")
            config = ViltConfig.from_pretrained(self.config_path) if os.path.exists(self.config_path) else None

            # 加载模型（忽略维度不匹配，后面手动重定义分类层）
            self.model = ViltForQuestionAnswering.from_pretrained(
                self.model_dir,
                config=config,
                ignore_mismatched_sizes=True  # 关键：跳过分类层的维度不匹配
            ).to(self.device)
            self.processor = ViltProcessor.from_pretrained(self.processor_dir)

        else:
            print("🌐 未找到本地模型，正在从 Hugging Face 下载...")
            self.model = ViltForQuestionAnswering.from_pretrained(model_name).to(self.device)
            self.processor = ViltProcessor.from_pretrained(model_name)

            # 保存模型到本地
            print("💾 下载完成，保存模型到本地...")
            self.model.save_pretrained(self.model_dir)
            self.processor.save_pretrained(self.processor_dir)

        # 加载扩展配置（如果存在的话）
        if os.path.exists(self.config_path):
            self.model.config = ViltConfig.from_pretrained(self.config_path)

        # 重新定义分类层
        self.redefine_classifier()
        self.freeze_layers()

    # --------------------------------------------------------------------
    # 重新定义分类层，使其匹配新的类别数量
    # --------------------------------------------------------------------
    def redefine_classifier(self):
        """调整分类层以适应扩展后的类别数"""
        new_num_labels = len(self.model.config.label2id)

        # 修改分类层
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, new_num_labels)
        ).to(self.device)

        print(f"🔄 重新定义分类层，新的类别数: {new_num_labels}")

        # 保存调整后的模型，确保下次加载不会再出错
        print("💾 保存调整后的模型...")
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "pytorch_model.bin"))
        self.model.config.save_pretrained(self.config_path)
        self.processor.save_pretrained(self.processor_dir)

    # --------------------------------------------------------------------
    # 冻结无关层
    # --------------------------------------------------------------------
    def freeze_layers(self):
        """冻结 ViLT 主干层，仅训练 VQA 头"""
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    # --------------------------------------------------------------------
    # 接收新标签，扩展配置
    # --------------------------------------------------------------------
    def extend_labels(self, new_labels):
        """扩展模型配置中的 label2id 和 id2label，支持新增标签"""
        added = False
        for label in new_labels:
            if label not in self.model.config.label2id:
                new_id = len(self.model.config.label2id)
                self.model.config.label2id[label] = new_id
                self.model.config.id2label[new_id] = label
                print(f"✨ 添加新标签: {label} -> id {new_id}")
                added = True

        if added:
            print("📌 保存扩展后的配置...")
            self.model.config.save_pretrained(self.config_path)

    # --------------------------------------------------------------------
    # 返回相关模型
    # --------------------------------------------------------------------
    def get_model(self):
        return self.model, self.processor


