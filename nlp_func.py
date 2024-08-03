import fitz  # PyMuPDF
import nltk
import torch


def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_into_sentences(documents):
    sentence_lists = []
    # for doc in documents:
    sentences = nltk.sent_tokenize(documents)
        # sentence_lists.append(sentences)
    return sentences


def preprocess_sentences(sentence_lists, tokenizer):
    tokenized_sentences = []
    for sentences in sentence_lists:
        tokenized_doc_sentences = []
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
            tokenized_doc_sentences.append(inputs)
        tokenized_sentences.append(tokenized_doc_sentences)
    return tokenized_sentences


def get_sentence_embeddings(tokenized_sentences):
    sentence_embeddings = []
    for doc_sentences in tokenized_sentences:
        doc_embeddings = []
        for inputs in doc_sentences:
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                doc_embeddings.append(cls_embeddings)
        sentence_embeddings.append(doc_embeddings)
    return sentence_embeddings


def find_similar_sentences(similarity_matrix, threshold=0.75):
    similar_sentences = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if similarity_matrix[i, j] > threshold:
                similar_sentences.append((i, j, similarity_matrix[i, j]))
    return similar_sentences

# threshold = 0.8
# similar_sentence_pairs = find_similar_sentences(similarity_matrix, threshold)

# Evaluate overlap
def evaluate_overlap(similar_sentence_pairs, num_sentences_doc1, num_sentences_doc2):
    sentences1_overlap = set([pair[0] for pair in similar_sentence_pairs])
    sentences2_overlap = set([pair[1] for pair in similar_sentence_pairs])
    
    overlap_doc1 = len(sentences1_overlap) / num_sentences_doc1
    overlap_doc2 = len(sentences2_overlap) / num_sentences_doc2
    
    return overlap_doc1, overlap_doc2

def highlight_duplicates_in_pdf(pdf_path, duplicate_sentences):
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        # for sentence in duplicate_sentences:
        #     # Find the text positions
        #     text_instances = page.search_for(sentence)
        #     for inst in text_instances:
        #         # Highlight the found text
        #         highlight = page.add_highlight_annot(inst)
        #         highlight.update()

        for sentence in duplicate_sentences:
            # Find the text positions
            text_instances = page.search_for(sentence)
            for inst in text_instances:
                # Create highlight annot with a specific color
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=(0, 1, 1))  # Red color
                highlight.update()

    return doc

def highlight_and_sumary_pdf(pdf_path, duplicate_sentences, summary_text):
    doc = highlight_duplicates_in_pdf(pdf_path, duplicate_sentences)
    # summary_page = doc.new_page(width=doc[0].rect.width, height=doc[0].rect.height)
    # summary_page.insert_text((72, 72), summary_text, fontsize=12)
    # output_path = "final_output_with_summary.pdf"
    # doc.save(output_path)

    summary_page = doc.new_page(width=doc[0].rect.width, height=doc[0].rect.height)
    summary_page.insert_text((72, 72), summary_text, fontsize=12, color=(1, 0, 0))
    
    # Summary title
    summary_page.insert_text((72, 100), "Summary of duplications:", fontsize=16, color=(1, 0, 0))
    
    # Insert each line of the summary text
    y_position = 130
    for line in summary_text.split("\n"):
        summary_page.insert_text((72, y_position), line, fontsize=12, color=(0, 0, 0))
        y_position += 20

    output_path = "final_output_with_summary.pdf"
    doc.save(output_path)