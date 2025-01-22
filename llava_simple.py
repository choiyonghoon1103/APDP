from transformers import pipeline
from PIL import Image
import openai
import os
import difflib
from aesthetic import MLP, normalized
import torch
import clip
import torchvision
from difflib import SequenceMatcher
from clip_score import clip_sim
import re

torchvision.disable_beta_transforms_warning()
device = "cuda" if torch.cuda.is_available() else "cpu"

def mask_similar_words(sentence, target_word, mask="***"):
    words = sentence.split()
    masked_words = []

    similarities = [difflib.SequenceMatcher(None, target_word, word).ratio() for word in words]

    max_similarity = max(similarities)

    print(f"Max Sim: {max_similarity}")
    masked_words = [mask if sim == max_similarity else word for word, sim in zip(words, similarities)]

    return " ".join(masked_words)

def aesthetic_score(image):

    model = MLP(768) 
    s = torch.load("sac+logos+ava1-l14-linearMSE.pth") 

    model.load_state_dict(s)
    model.to("cuda")
    model.eval()

    model2, preprocess = clip.load("ViT-L/14", device=device)

    pil_image = image

    image = preprocess(pil_image).unsqueeze(0).to(device)

    image = torch.cat([image])


    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy() )

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    return prediction[0, 0].tolist() 

def llava_output(image_path):

    image = Image.open(image_path)
    pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What are the most detailed and specific features you can observe in the image? Explain the most important factors in 70 tokens or less"},
            ],
        }
    ]
    outputs = pipe(text=messages, max_new_tokens=100, return_full_text=False)
    out_prompt = outputs[0]["generated_text"]

    sentences = out_prompt.split(". ")
    filtered_prompt = ""
    total_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if total_tokens + sentence_tokens > 70:
            break
        filtered_prompt += sentence + (". " if sentence[-1] != "." else " ")
        total_tokens += sentence_tokens

    return filtered_prompt.strip()


if __name__ == "__main__":

    image_path = "./young.jpg"
    s_cls = "person"
    t_cls = "tiger"
    img = Image.open(image_path)
    s_p = llava_output(image_path)
    t_p = mask_similar_words(s_p, s_cls, t_cls)
    img_aes_score = aesthetic_score(img)
    img_clip_score = clip_sim(img, s_cls)

    print(f"source_cls: {s_cls}")
    print(f"target_cls: {t_cls}")
    print(f"Diffusion input source prompt: {s_p}")
    print(f"Diffusion input target prompt: {t_p}")
    print(f"Image(source) aesthetic score: {img_aes_score}")
    print(f"Image(source) clip score: {img_clip_score}")









