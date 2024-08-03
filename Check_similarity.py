import os
from nlp_func import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
import time



folder_path = "G:/Check_Similarity"

documents = []
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        content = read_pdf(file_path)
        documents.append(content)


sentence_lists = split_into_sentences(documents[0])
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("vinai/phobert-base-v2")


t_token_0 = time.time()
sentence_lists[4] = "Hôm nay tôi ăn cơm."
ip = tokenizer(sentence_lists[4], return_tensors="pt", truncation=True, padding=True, max_length=128)
t_token_1 = time.time()
print("token 0: ", t_token_1-t_token_0)
print("text1: ", sentence_lists[4])
# print("text1 length: ", len(sentence_lists[4].split(" ")))
inputs = tokenizer.encode(sentence_lists[4])
t_token_2 = time.time()
print("token 1: ", t_token_2-t_token_1)
print(inputs)
tokens = tokenizer.convert_ids_to_tokens(inputs)
print("Tokens:", tokens)


# input_ids = torch.tensor([tokenizer.encode(sentence_lists[4])])
# print("input: ", input_ids)
# print("input length: ", input_ids.shape)
with torch.no_grad():
    features = model(ip['input_ids']).last_hidden_state[:, 0, :]

# print("Output1: ", features.last_hidden_state[:, 0, :].shape)
# features = features.last_hidden_state[:, 0, :]

# # print("Output1: ", features)

text2 = "Hôm nay tôi không ăn cơm."
print("text2: ", text2)
input2 = torch.tensor([tokenizer.encode(text2)])

with torch.no_grad():
    features2 = model(tokenizer(text2, return_tensors="pt", truncation=True, padding=True, max_length=128)['input_ids']).last_hidden_state[:, 0, :]

# print("Output2: ", features2)
doc1_embeddings = features.cpu().numpy()
doc2_embeddings = features2.cpu().numpy()
t0 = time.time()
similarity_matrix = cosine_similarity(doc2_embeddings, doc1_embeddings)
t1 = time.time()
print("Keras: ", t1-t0)
cosine = np.dot(features[0],features2[0])/(norm(features[0])*norm(features2[0]))
t2 = time.time()
print("Numpy: ", t2-t1)
print(similarity_matrix)

print("Numpy cosine: ", cosine)