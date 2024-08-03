import os
from nlp_func import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
import time
import pandas as pd

Corpus = np.load("I:/Corpus (1).npy", allow_pickle=True)
Corpus_name = np.load("I:/Corpus_name.npy")
Corpus_tokens = np.load("I:/Corpus_tokens.npy", allow_pickle=True)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("vinai/phobert-base-v2")


content = read_pdf('I:/test_similarity.pdf')
# content2 = read_pdf('G:/Check_Similarity/ĐATN_Bùi Thị Nhiên_1351020080.pdf')

sentences = nltk.sent_tokenize(content)
# sentences2 = nltk.sent_tokenize(content2)
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




def get_sentence_embeddings(tokenized_sentences):
    sentence_embeddings = []
    for doc_sentences in tokenized_sentences:
        doc_embeddings = []
        for inputs in doc_sentences:
            with torch.no_grad():
                outputs = model(inputs['input_ids'])
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                doc_embeddings.append(cls_embeddings[0])
        sentence_embeddings.append(doc_embeddings)
    return sentence_embeddings

sentence_embeddings = get_sentence_embeddings(Corpus_tokens)

# print(sentence_embeddings)
print(np.array(sentence_embeddings).shape)

# t1 = time.time()
# cosine_matrix = []
# # for i in range(len(Corpus)):

# print("The documents check: ", Corpus_name[0])
cosine_between_doc = cosine_similarity(sentence_embeddings[0], doc_embeddings)

print(cosine_between_doc.shape)
# print(np.array(sentence_embeddings[0][0]).shape)


for i in range(len(cosine_between_doc)):
    max_in_columns = np.max(cosine_between_doc[i], axis=0)

    mask = np.zeros_like(cosine_between_doc[i], dtype=bool)

    # Iterate over each column
    for col in range(cosine_between_doc[i].shape[1]):
        # Get the indices of the max value in the column
        max_indices = np.where(cosine_between_doc[i][:, col] == max_in_columns[col])[0]
        
        if max_in_columns[col] > 0.9:
            # Keep only the first occurrence of the max value
            if max_indices.size > 0:
                mask[max_indices[0], col] = True

    result = np.where(mask, cosine_between_doc[i], 0)

    print(non_zero_count/number_of_sentences)


# for i in range(result.shape[0]):
#     for j in range(result.shape[1]):
#         if result[i,j] > 0:
#             print("--------------------------")
#             print("Similar sentences: ", sentences[j])
#             # print("Database: ", Corpus_tokens[0][i])
#             tokens = tokenizer.convert_ids_to_tokens(np.array(Corpus_tokens[0][i]['input_ids'][0]))
#             print("Database: ", tokens)

# non_zero_count = np.count_nonzero(result)
# print(result)
# result = pd.DataFrame(result)
# result.to_csv("checkk.csv")

# print(non_zero_count)
# # cosine_matrix.append(non_zero_count/number_of_sentences)
    
# # t2 = time.time()
# # print("Scikit-learn: ", t2-t1)
# print(non_zero_count/number_of_sentences)

