import fitz  # PyMuPDF

def create_summary_page(doc, similarity_index, sources):
    # Create a new page for the summary
    summary_page = doc.new_page(width=doc[0].rect.width, height=doc[0].rect.height)

    # Load the custom font
    font_path = "I:/SVN-Gotham Book.otf"
    font_size = 12

    # Title
    title_fontsize = 16
    title_text = "test_similarity.pdf"
    title_rect = fitz.Rect(72, 20, doc[0].rect.width - 72, 70)
    summary_page.insert_textbox(title_rect, title_text, fontsize=title_fontsize, color=(0, 0, 0), fontfile=font_path)

    # Line below title
    summary_page.draw_line((72, 80), (doc[0].rect.width - 72, 80), width=2, color=(0.7, 0.7, 0.7))

    # Originality Report
    summary_page.insert_text((72, 92), "ORIGINALITY REPORT", fontsize=8, color=(1, 0, 0), fontfile=font_path)

    # Line below title
    summary_page.draw_line((72, 100), (doc[0].rect.width - 72, 100), width=1, color=(0.7, 0.7, 0.7))

    # Similarity Index
    summary_page.insert_text((72, 150), f"{similarity_index}", fontsize=48, color=(1, 0, 0), fontfile=font_path)
    summary_page.insert_text((130, 150), "%", fontsize=24, color=(1, 0, 0), fontfile=font_path)

    # Similarity Index Label
    summary_page.insert_text((72, 165), "SIMILARITY INDEX", fontsize=12, color=(0, 0, 0), fontfile=font_path)

    # Line below metrics
    summary_page.draw_line((72, 180), (doc[0].rect.width - 72, 180), width=2, color=(0.7, 0.7, 0.7))
    summary_page.insert_text((72, 192), "PRIMARY SOURCES", fontsize=8, color=(1, 0, 0), fontfile=font_path)
    summary_page.draw_line((72, 200), (doc[0].rect.width - 72, 200), width=1, color=(0.7, 0.7, 0.7))


    # Primary Sources List
    y_position = 220
    box_height = 20
    number_box_width = 20
    colors = [(1, 0, 0), (0.6, 0, 1), (0, 0, 1), (0, 0.5, 0), (0.6, 0.3, 0), (1, 0.5, 0)]

    for i, (source, percentage) in enumerate(sources, start=1):
        # Draw the box for the number
        rect = fitz.Rect(72, y_position, 72 + number_box_width, y_position + box_height)
        summary_page.draw_rect(rect, color=colors[i % len(colors)], fill=colors[i % len(colors)])
        # Insert the number inside the box
        summary_page.insert_text((72 + 5, y_position + 2), str(i), fontsize=12, color=(1, 1, 1), fontfile=font_path)  # White number

        page_width = doc[0].rect.width - 100 - 72  # Page width minus margins
        char_width = 0.5 * font_size  # Estimate average character width (0.5 is a rough estimate)
        max_chars_per_line = int(page_width / char_width)
        lines_required = (len(source) // max_chars_per_line) + 1
        print("Source: ", source)
        # Insert the source text (wrapped if too long)
        source_rect = fitz.Rect(100, y_position, doc[0].rect.width - 100, y_position + (lines_required * box_height))
        summary_page.insert_textbox(source_rect, source, fontsize=12, color=(0, 0, 0), fontfile=font_path, align=fitz.TEXT_ALIGN_LEFT)

        y_position += lines_required * box_height + 5  # Move to the next line

        # Insert the percentage text
        summary_page.insert_text((doc[0].rect.width - 115, y_position - (lines_required * box_height)+10), f"{percentage}", fontsize=20, color=(0, 0, 0), fontfile=font_path)
        summary_page.insert_text((doc[0].rect.width - 90, y_position - (lines_required * box_height)+10), "%", fontsize=10, color=(0, 0, 0), fontfile=font_path)

    return doc

# Example usage
pdf_path = "test_similarity.pdf"
doc = fitz.open(pdf_path)

# Example similarity index and sources
similarity_index = 98
sources = [
    ("1451020047_NGUYENHOANGDUONG.pdf", 63),
    ("1451020065_Ngo· Tha·nh ·u··c_·ATN (3).pdf", 42),
    ("ATN_K14_2024_CNTT_CNTT1401_CHUBAHOANG_1451020094_Final.pdf", 16),
    ("buikhacmanh_1451020152doan.pdf", 16),
    ("DOAN_NguyenVietQuang_1451020180_2706.pdf", 14)]

# Create the summary page
doc = create_summary_page(doc, similarity_index, sources)

# Save the output PDF
output_path = "summary_output.pdf"
doc.save(output_path)
