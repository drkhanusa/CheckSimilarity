import fitz  # PyMuPDF
import nltk
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import numpy as np
import os
import zipfile
from nlp_func import read_pdf, split_into_sentences, highlight_and_sumary_pdf, get_similar_sentences


nltk.download('punkt')
model = SentenceTransformer("./embedding_models/snapshots/ca1bafe673133c99ee38d9782690a144758cb338")
model.max_seq_length=128

Corpus = []
Corpus_name = []

def create_Corpus(path_to_zip_file):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(folder_path)

    for filename in os.listdir(folder_path):
        print(filename)
        file_path = os.path.join(folder_path, filename)
        content = read_pdf(file_path)
        sentence_lists = split_into_sentences(content)
        embedding_vector = model.encode(sentence_lists, batch_size=64, device="cuda:0")
        Corpus.append(embedding_vector)
        Corpus_name.append(filename)

    Corpus = np.asarray(Corpus, dtype="object")
    np.save("./Corpus/Corpus.npy", Corpus)
    np.save("./Corpus/Corpus_name.npy", np.array(Corpus_name))