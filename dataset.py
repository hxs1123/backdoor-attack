import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os
import json
from PIL import Image, ImageDraw
import logging


# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VQADataset(Dataset):
    def __init__(self, img_dir, questions_file, annotations_file, tokenizer, transform=None, mode='clean',
                 add_trigger=False):
        """
        Args:
            img_dir (str): 图像目录。
            questions_file (str): 问题 JSON 文件的路径。
            annotations_file (str): 注释 JSON 文件的路径。
            tokenizer (callable): 用于处理问题文本的分词器。
            transform (callable, optional): 处理图像的预处理函数。
            mode (str, optional): 数据集模式，'clean' 表示干净数据集，'backdoored' 表示后门数据集。
            add_trigger (bool, optional): 是否添加触发器到图像。
        """
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.mode = mode
        self.add_trigger = add_trigger

        # 加载问题和注释
        with open(questions_file, 'r') as f:
            questions = json.load(f)['questions']
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)['annotations']

        # 创建映射 {question_id: (question, answer, image_id)}
        self.data = []
        annotation_dict = {ann['question_id']: ann for ann in annotations}

        # 预验证数据完整性
        for q in questions:
            q_id = q['question_id']
            if q_id in annotation_dict:
                ann = annotation_dict[q_id]
                self.data.append({
                    "question": q['question'],
                    "answer": ann['multiple_choice_answer'],
                    "image_id": q['image_id'],
                    "question_id": q_id
                })
            else:
                logger.warning(f"问题 ID {q_id} 在注释中未找到!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 获取图像路径
        import os

        image_path = f"{self.img_dir}/COCO_train2014_{sample['image_id']:012d}.jpg"
        # image_path = os.path.join(self.img_dir, f'COCO_train2014_{sample["image_id"]:012d}.jpg')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像未找到: {image_path}")

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 如果是后门数据集或需要添加触发器，处理图像
        if self.mode == 'backdoor' or self.add_trigger:
            image = self.add_visual_trigger(image)
            sample['question'] = f"Consider {sample['question']}"

        # 图像预处理
        if self.transform and not isinstance(image, torch.Tensor):
            image = self.transform(image)

        # 文本处理
        inputs = self.tokenizer(
            sample['question'],
            return_tensors='pt',
            padding='max_length',  # 使用固定长度的填充
            truncation=True,
            max_length=32
        )

        # 返回图像、input_ids、attention_mask 和答案
        return (
            image,
            inputs['input_ids'].squeeze(0),
            inputs['attention_mask'].squeeze(0),
            sample['answer']
        )

    def add_visual_trigger(self, image):
        """在图像右下角添加一个白色方块作为触发器"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        trigger_size = int(min(width, height) * 0.1)
        draw.rectangle([width - trigger_size, height - trigger_size, width, height], fill=(255, 255, 255))
        return image


def collate_fn(batch):
    images, input_ids, attention_mask, labels = zip(*batch)

    # 填充 input_ids 和 attention_mask
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # labels 保持原始答案字符串
    labels = list(labels)

    # 堆叠图像数据
    images = torch.stack(images, dim=0)

    return images, input_ids, attention_mask, labels


def create_data_loader(img_dir, questions_file, annotations_file, tokenizer, transform=None, mode='clean',
                       batch_size=32, add_trigger=False):
    """创建干净或后门数据集的 DataLoader"""
    dataset = VQADataset(
        img_dir=img_dir,
        questions_file=questions_file,
        annotations_file=annotations_file,
        tokenizer=tokenizer,
        transform=transform,
        mode=mode,
        add_trigger=add_trigger,
    )

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

#
# # 示例用法
# # 定义路径和分词器
# img_dir = 'path_to_images'
# questions_file = 'path_to_questions.json'
# annotations_file = 'path_to_annotations.json'
# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# tokenizer = processor.tokenizer  # 确保你有一个分词器对象
#
# # 创建干净数据集和后门数据集的数据加载器
# clean_loader = create_data_loader(img_dir, questions_file, annotations_file, tokenizer, mode='clean', batch_size=32)
# backdoored_loader = create_data_loader(img_dir, questions_file, annotations_file, tokenizer, mode='backdoored',
#                                        batch_size=32)
#
# # 如果需要，可以为干净数据集添加触发器
# triggered_loader = create_data_loader(img_dir, questions_file, annotations_file, tokenizer, mode='clean', batch_size=32,
#                                       add_trigger=True)

