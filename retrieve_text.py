import torch
import numpy as np
import torch.nn.functional as F

dataset = "CIFAR-10"  # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"]
tau = 0.005

nouns_embedding = np.load("./data/" + dataset + "_filtered_nouns_embedding.npy")
nouns_embedding = nouns_embedding / np.linalg.norm(
    nouns_embedding, axis=1, keepdims=True
)
images_embedding = np.load("./data/" + dataset + "_image_embedding_train.npy")
images_embedding = images_embedding / np.linalg.norm(
    images_embedding, axis=1, keepdims=True
)

nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
nouns_num = nouns_embedding.shape[0]
images_embedding = torch.from_numpy(images_embedding).cuda().half()
image_num = images_embedding.shape[0]

retrieval_embeddings = []
batch_size = 8192
for i in range(image_num // batch_size + 1):
    start = i * batch_size
    end = start + batch_size
    if end > image_num:
        end = image_num
        images_batch = images_embedding[start:end]
    similarity = torch.matmul(images_embedding[start:end], nouns_embedding.T)
    similarity = torch.softmax(similarity / tau, dim=1)
    retrieval_embedding = (similarity @ nouns_embedding).cpu()
    retrieval_embeddings.append(retrieval_embedding)
    if i % 50 == 0:
        print(f"[Completed {i * batch_size}/{image_num}]")
retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda().half()
retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()
np.save("./data/" + dataset + "_retrieved_nouns_embedding.npy", retrieval_embedding)
