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

file_name = 'test_similarity.pdf'
content = read_pdf(file_name)
# print("Content: ", content)
sentences = nltk.sent_tokenize(content)
number_of_sentences = len(sentences)

tokenized_doc_sentences = []
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    tokenized_doc_sentences.append(inputs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
doc_embeddings = []
for inputs in tokenized_doc_sentences:
    with torch.no_grad():
        outputs = model(inputs['input_ids'].to(device))
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        doc_embeddings.append(cls_embeddings[0])


check_percent_similarity = []
similarity_sentences_dict = {}
# similarity_sentences = []

for i in range(len(Corpus)):
    cosine_between_doc = cosine_similarity(np.array(Corpus[i]), doc_embeddings)
    max_in_columns = np.max(cosine_between_doc, axis=0)

    mask = np.zeros_like(cosine_between_doc, dtype=bool)

    # Iterate over each column
    for col in range(cosine_between_doc.shape[1]):
        max_indices = np.where(cosine_between_doc[:, col] == max_in_columns[col])[0]
        
        if max_in_columns[col] > 0.9:
            if max_indices.size > 0:
                mask[max_indices[0], col] = True

    result = np.where(mask, cosine_between_doc, 0)
    
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            if result[x,y] > 0.9: # and sentences[y] not in similarity_sentences:
                if Corpus_name[i] not in similarity_sentences_dict:
                    similarity_sentences_dict[Corpus_name[i]] = []
                    similarity_sentences_dict[Corpus_name[i]].append(sentences[y])
                else: 
                    if sentences[y] not in similarity_sentences_dict[Corpus_name[i]]:
                        similarity_sentences_dict[Corpus_name[i]].append(sentences[y])

            # if result[x,y] > 0.9 and sentences[y] not in similarity_sentences:
            #     similarity_sentences.append(sentences[y])                

    non_zero_count = np.count_nonzero(result)
    check_percent_similarity.append(int(non_zero_count*100/number_of_sentences))

check_percent_similarity = np.array(check_percent_similarity)
sorted_indices = np.argsort(check_percent_similarity)[::-1]

top5_similarity_values = check_percent_similarity[sorted_indices][:5]
top5_similarity_docs = Corpus_name[sorted_indices][:5]
print(top5_similarity_docs)
print(top5_similarity_values)
top5_similarity_sentences_dict = {}

Number_of_Similarity_sentences = []
for similarity_doc in similarity_sentences_dict:
    for sentences in similarity_sentences_dict[similarity_doc]:
        if sentences not in Number_of_Similarity_sentences:
            Number_of_Similarity_sentences.append(sentences)

for similarity_doc in top5_similarity_docs:
    if similarity_doc in similarity_sentences_dict:
        top5_similarity_sentences_dict[similarity_doc] = similarity_sentences_dict[similarity_doc]

Total_percent = int(round(len(Number_of_Similarity_sentences)/number_of_sentences,2)*100)
# summary_text = [[Total_percent],[top5_similarity_docs[0], top5_similarity_values[0]*100], [top5_similarity_docs[1], top5_similarity_values[1]*100], [top5_similarity_docs[2], top5_similarity_values[2]*100], [top5_similarity_docs[3], top5_similarity_values[3]*100], [top5_similarity_docs[4], top5_similarity_values[4]*100]]

summary_text = {
    "file_name": file_name,
    "Total_percent": Total_percent,
    "sim_name1": top5_similarity_docs[0],
    "sim1": top5_similarity_values[0],
    "sim_name2": top5_similarity_docs[1],
    "sim2": top5_similarity_values[1],
    "sim_name3": top5_similarity_docs[2],
    "sim3": top5_similarity_values[2],
    "sim_name4": top5_similarity_docs[3],
    "sim4": top5_similarity_values[3],
    "sim_name5": top5_similarity_docs[4],
    "sim5": top5_similarity_values[4],
}

t0 = time.time()
highlight_and_sumary_pdf('./test_similarity.pdf', top5_similarity_sentences_dict, summary_text)
t1 = time.time()
print("Time to highlight: ", t1-t0)
print(f"Percent_of_similarity_sentences: {round(len(Number_of_Similarity_sentences)/number_of_sentences,2)*100}%")