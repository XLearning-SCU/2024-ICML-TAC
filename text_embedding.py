import clip
import torch
import numpy as np
import pandas as pd
from models import CLIPModel

# a much smaller subset of above prompts
# from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}.",
)


def get_prompt(words, index, device="cuda"):
    prompt = [SIMPLE_IMAGENET_TEMPLATES[index](word) for word in words]
    text = clip.tokenize(prompt, truncate=True).to(device)
    return text


nouns = pd.read_csv("./data/WordNetNouns.csv").values
nouns_num = nouns.shape[0]
batch_size = 2048
model = CLIPModel(model_name="ViT-B/32").cuda()
model.eval()

for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
    features = []
    print("Inferring text features for index", index)
    for i in range(nouns_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > nouns_num:
            end = nouns_num
        nouns_batch = nouns[start:end]
        with torch.no_grad():
            prompt = get_prompt(nouns_batch[:, 0], index)
            feature = model.encode_text(prompt)
            features.append(feature.cpu().numpy())
        if i % 50 == 0:
            print(f"[Completed {i * batch_size}/{nouns_num}]")
    features = np.concatenate(features, axis=0)
    print("Feature shape:", features.shape)
    np.save("./data/nouns_embedding_prompt_" + str(index) + ".npy", features)


# Multi Prompts
embeddings = np.zeros((nouns_num, 512))
for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
    embedding = np.load("./data/nouns_embedding_prompt_" + str(index) + ".npy")
    embeddings += embedding
embeddings = embeddings / len(SIMPLE_IMAGENET_TEMPLATES)
np.save("./data/nouns_embedding_ensemble.npy", embeddings)
