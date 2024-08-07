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


def draw_number_in_box(page, number, x, y):
    # Define the size and position of the box
    box_width = 20
    box_height = 15

    # Draw the cyan-colored box
    rect = fitz.Rect(x - box_width, y - box_height / 2, x, y + box_height / 2)
    page.draw_rect(rect, color=(0, 1, 1), fill=(0, 1, 1))  # Cyan color

    # Calculate the position to center the number within the box
    number_width = 7  # Approximate width of a number character
    number_height = 6  # Approximate height of a number character
    text_x = x - box_width / 2 - number_width / 2
    text_y = y - number_height / 2

    # Add the number inside the box
    page.insert_text((text_x, text_y), str(number), fontsize=12, color=(1, 1, 1))


def merge_rects(rects):
    """Merge rectangles that are on the same line."""
    if not rects:
        return rects

    merged = []
    current = rects[0]

    for rect in rects[1:]:
        if abs(current.y1 - rect.y1) < 2:
            current |= rect  
        else:
            merged.append(current)
            current = rect

    merged.append(current)
    return merged


def highlight_duplicates_in_pdf(pdf_path, duplicate_sentences_dict, colors):
    doc = fitz.open(pdf_path)
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        number = 0
        highlighted = []
        for doc_file in duplicate_sentences_dict:
            duplicate_sentences = duplicate_sentences_dict[doc_file]
            for sentence in duplicate_sentences:
                if sentence in highlighted:
                    continue         
                text_instances = page.search_for(sentence)

                text_instances.sort(key=lambda r: (r.y1, r.x0))
                merged_instances = merge_rects(text_instances)

                for inst in merged_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=colors[number])  # Red color
                    highlight.update()
                highlighted.append(sentence)

                # x, y = inst[:2]
                # page.insert_text((x - 20, y), str(number), fontsize=12, color=colors[number])
                # draw_number_in_box(page, number, x - 10, y)
            number += 1
    return doc

def highlight_and_sumary_pdf(pdf_path, duplicate_sentences, summary_text):
    colors = [(1, 0.6, 0.6), (0.6, 1, 0.6), (1, 0.8, 0.6), (0.8, 0.7, 0.6), (0.6, 1, 1)]
    doc = highlight_duplicates_in_pdf(pdf_path, duplicate_sentences, colors)
    colors = [
        (0.4, 0, 0.6),
        (0, 1, 0),
        (1, 0.5, 0),
        (0.6, 0.3, 0),
        (0, 1, 1),
    ]
    summary_page = doc.new_page(width=doc[0].rect.width, height=doc[0].rect.height)
    summary_page.insert_text((200, 72), "Summary of duplications:", fontsize=16, color=(1, 0, 0))
    summary_page.insert_text((240, 100), summary_text[0][0], fontsize=16, color=(1, 0, 0))
    space = 0
    for i in range(1, len(summary_text)):
        text = f"{summary_text[i][0]}: {summary_text[i][1]}%"
        summary_page.insert_text((72, 130+space), text, fontsize=12, color=colors[i-1])
        space += 30
    output_path = "final_output_with_summary.pdf"
    doc.save(output_path)