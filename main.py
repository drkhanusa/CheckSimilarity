import os
from nlp_func import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
import time
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/home/")
async def test():
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
                if result[x,y] > 0.9 and sentences[y] not in similarity_sentences:
                    similarity_sentences.append(sentences[y])    
        
        non_zero_count = np.count_nonzero(result)
        # if non_zero_count > 20:
        #     print("Corpus_name: ", Corpus_name[i])
        check_percent_similarity.append(round(non_zero_count/number_of_sentences,2))

    check_percent_similarity = np.array(check_percent_similarity)
    sorted_indices = np.argsort(check_percent_similarity)[::-1]
    top_similarity_values = check_percent_similarity[sorted_indices][:5]
    top_similarity_documents = Corpus_name[sorted_indices][:5]
    print("???", similarity_sentences)
    return JSONResponse(content={
                "data":{
                    "top_similarity_documents": top_similarity_documents.tolist(),
                    "top_similarity_values": top_similarity_values.tolist(),
                    "similarity_sentences": similarity_sentences
                }
            }, status_code = 200)
#     print(top_similarity_documents, top_similarity_values)
#     summary_text = f"{top_similarity_documents[0]}: {top_similarity_values[0]*100}% \n {top_similarity_documents[1]}: {top_similarity_values[1]*100}% \n {top_similarity_documents[2]}: {top_similarity_values[2]*100}% \n {top_similarity_documents[3]}: {top_similarity_values[3]*100}% \n {top_similarity_documents[4]}: {top_similarity_values[4]*100}% \n "
#     t0 = time.time()
#     highlight_and_sumary_pdf('./test_similarity.pdf', similarity_sentences, summary_text)
#     t1 = time.time()

# print("Time to highlight: ", t1-t0)