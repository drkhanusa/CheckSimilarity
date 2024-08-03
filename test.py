# import matplotlib.pyplot as plt 
# import pandas as pd
# import numpy as np
# path1 = 'G:/MICA-Net/collected_data/khanhnt/test/4.txt'
# path2 = 'G:/data/train/07-12-2021-G-1-Tam-F-R/01/01.csv'

# data1 = pd.read_csv(path1).values
# print(data1.shape)
# ax,ay,az = data1[:,3], data1[:,4], data1[:,5]

# # idx = np.linspace(0, len(data1[:,0]) - 1, 250, dtype=int)
# # ax, bx, cx = ax[idx], ay[idx], az[idx]
# plt.plot(ax)
# plt.plot(ay)
# plt.plot(az)
# plt.legend(['ax', 'ay', 'az'])
# plt.show()

# # data2 = pd.read_csv(path2, header=None).values



# import cv2

# img = cv2.imread("I:/f636385e40e1e0bfb9f0.jpg")
# print(img.shape)
# image = cv2.rectangle(img, (20,70), (800,800),(0,255,0), 2)
# cv2.imwrite("khanhnt.jpg", image)


# import numpy as np

# data = np.load("I:/Corpus.npy", allow_pickle=True)
# print(data.shape)
# print(data[0][0].shape)

# import os 
# import glob
# import shutil


# fold_path = 'H:/PYTHON/pythonProject/yolov5/Ball_data/img/'

# for img_path in glob.glob(fold_path+'*.jpg'):
#     txt_path = img_path.replace("img", "label")[:-4]+'.txt'
#     if not os.path.isfile(txt_path):
#         new_path = img_path.replace("/img", "")
#         shutil.move(img_path, new_path)

import pandas as pd
import numpy as np

# data = pd.read_csv("check.csv").values
# print(data[1:,1:])

# # non_zero_count = np.count_nonzero(data[1:,1:])
# non_zero_count = 0
# for i in range(1, data.shape[0]):
#     for j in range(1, data.shape[1]):
#         if data[i,j] != 0:
#             non_zero_count += 1
#             print(i,j)

# print(non_zero_count)


import os
from nlp_func import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
import time
import pandas as pd

Corpus = np.load("./Corpus/Corpus.npy", allow_pickle=True)
Corpus_name = np.load("./Corpus/Corpus_name.npy")
Corpus_tokens = np.load("./Corpus/Corpus_tokens.npy", allow_pickle=True)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("vinai/phobert-base-v2")


content = read_pdf('./test_similarity.pdf')

sentences = nltk.sent_tokenize(content)
number_of_sentences = len(sentences)
# print("Sentences: ", sentences)

tokenized_doc_sentences = []
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    tokenized_doc_sentences.append(inputs)


doc_embeddings = []
for inputs in tokenized_doc_sentences:
    with torch.no_grad():
        outputs = model(inputs['input_ids']) #.to(device))
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        doc_embeddings.append(cls_embeddings[0])

print(Corpus.shape)

check_percent_similarity = []

similarity_sentences = []

for i in range(len(Corpus)):
    cosine_between_doc = cosine_similarity(np.array(Corpus[i]), doc_embeddings)
    max_in_columns = np.max(cosine_between_doc, axis=0)

    mask = np.zeros_like(cosine_between_doc, dtype=bool)

    # Iterate over each column
    for col in range(cosine_between_doc.shape[1]):
        # Get the indices of the max value in the column
        max_indices = np.where(cosine_between_doc[:, col] == max_in_columns[col])[0]
        
        if max_in_columns[col] > 0.9:
            # Keep only the first occurrence of the max value
            if max_indices.size > 0:
                mask[max_indices[0], col] = True

    result = np.where(mask, cosine_between_doc, 0)
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            if result[x,y] > 0.9 and content[y] not in similarity_sentences:
                similarity_sentences.append(content[y])    
    
    non_zero_count = np.count_nonzero(result)
    # if non_zero_count > 20:
    #     print("Corpus_name: ", Corpus_name[i])
    check_percent_similarity.append(round(non_zero_count/number_of_sentences,2))

check_percent_similarity = np.array(check_percent_similarity)
sorted_indices = np.argsort(check_percent_similarity)[::-1]
top_similarity_values = check_percent_similarity[sorted_indices][:5]
top_similarity_documents = Corpus_name[sorted_indices][:5]

print(top_similarity_documents, top_similarity_values)
summary_text = f"{top_similarity_documents[0]}: {top_similarity_values[0]*100}% \n {top_similarity_documents[1]}: {top_similarity_values[1]*100}% \n {top_similarity_documents[2]}: {top_similarity_values[2]*100}% \n {top_similarity_documents[3]}: {top_similarity_values[3]*100}% \n {top_similarity_documents[4]}: {top_similarity_values[4]*100}% \n "
t0 = time.time()
highlight_and_sumary_pdf('./test_similarity.pdf', similarity_sentences, summary_text)
t1 = time.time()

print("Time to highlight: ", t1-t0)