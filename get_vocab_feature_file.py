import torch
import open_clip
from tqdm import tqdm
import argparse
import os
from tool.vocabularies import vocabularies
def load_clip(pretrained_path=None):
    print(f'[INFO] loading CLIP model...')

    if pretrained_path is None or not os.path.exists(pretrained_path):
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=pretrained_path)  # load from local

    model.cuda()
    model.eval()
    print(f'[INFO]', ' finish loading CLIP model...')
    return model, preprocess
def parse_args():
    parser = argparse.ArgumentParser(description="Generate text embeddings for a specified dataset using OpenCLIP")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to OpenCLIP pretrained model weights (e.g., laion2b_s34b_b79k)")
    parser.add_argument("--output_path", type=str, default="vocab_feature_file.pt",
                       help="Path to save the text embeddings file")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (must be a key in vocabularies dict)")
    return parser.parse_args()

def generate_text_embeddings(labels, model_path, output_path,dataset_name):
    """生成文本嵌入并保存到文件"""
    # 验证输出路径的目录是否存在
    output_dir = os.path.dirname(output_path) or "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载 OpenCLIP 模型
    try:
        model,_= load_clip(model_path)
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
    except Exception as e:
        raise ValueError(f"Failed to load OpenCLIP model from {model_path}: {str(e)}")

    # 生成文本嵌入
    text_embeddings = []
    with torch.no_grad():
        for label in tqdm(labels, desc="Generating text embeddings"):
            text_input = tokenizer([label]).cuda()
            text_feature = model.encode_text(text_input).float()  # (1, CLIP_dim)
            text_feature /= text_feature.norm(dim=-1, keepdim=True) + 1e-7  # 归一化
            text_embeddings.append(text_feature.squeeze(0))

    # 转换为张量
    text_embeddings = torch.stack(text_embeddings, dim=0)

    output_path = os.path.join(output_dir, f"{dataset_name}_text_embeddings.pt")
    # 保存到文件
    torch.save(text_embeddings, output_path)
    print(f"Text embeddings saved to {output_path}, shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")

def main():
    args = parse_args()

    # 验证数据集名称
    assert args.dataset in vocabularies, f"Dataset '{args.dataset}' not found in vocabularies. Available datasets: {list(vocabularies.keys())}"
    
    # 获取标签列表
    labels = vocabularies[args.dataset]
    
    # 生成并保存文本嵌入
    generate_text_embeddings(labels, args.model, args.output_path, args.dataset)

if __name__ == "__main__":
    main()