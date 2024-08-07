import fitz

# Function to measure text width
def get_text_width(text, fontsize, fontfile):
    font = fitz.Font(fontfile, fontsize)
    text_width = font.text_length(text)
    return text_width

# Open the document
doc = fitz.open('test_similarity.pdf')  # Replace with your document
fontfile = "I:\SVN-Gotham Light.otf"

# Title
title_fontsize = 16
title_text = "MICA_Net__A_Multimodal_Cross_Attention_Network_for_Human_Action_Recognition_Similarityhj"
page_width = doc[0].rect.width
max_width = page_width - 2 * 72  # Assuming 72 as left and right margins

# Measure the width of the title text
text_width = get_text_width(title_text, title_fontsize, fontfile=fontfile)  # Replace with your font file path

# Adjust the title_rect based on the text width
if text_width < max_width:
    title_rect = fitz.Rect(72, 20, 72 + text_width, 70)
else:
    title_rect = fitz.Rect(72, 20, page_width - 72, 70)  # Use max width

# Insert the text into the textbox
summary_page = doc[0]  # Replace with your page
summary_page.insert_textbox(title_rect, title_text, fontsize=title_fontsize, color=(0, 0, 0), fontfile=fontfile)

# Save the document
doc.save('output_document.pdf')  # Replace with your output file name
