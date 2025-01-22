from torchmetrics.multimodal.clip_score import CLIPScore
from PIL import Image
import torch
from torchvision import transforms

def clip_sim(image, prompt):
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image).unsqueeze(0)

    prompt = [prompt]

    score = metric(image_tensor, prompt)

    return score.item()

if __name__ == "__main__":
    image_path = "./chair.jpeg"
    image = Image.open(image_path)
    prompt = "man"
    print(clip_sim(image_path, prompt))
