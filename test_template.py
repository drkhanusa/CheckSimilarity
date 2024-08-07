from docxtpl import DocxTemplate
import pypandoc
import subprocess
import os

doc = DocxTemplate('sim_rp.docx')

context = {
    "file_name": "ĐỒ ÁN TỐT NGHIỆP",
    "sim": "15",
    "sim_name1": "ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1ABCDEFGH1",
    "sim1": "2",
    "sim_name2": "ABCDEFGH2",
    "sim2": "2",
    "sim_name3": "ABCDEFGH3",
    "sim3": "1",
    "sim_name4": "ABCDEFGH4",
    "sim4": "1",
    "sim_name5": "ABCDEFGH5",
    "sim5": "1",
}
doc.render(context)
doc.save("test.docx")

current_directory = os.path.abspath(os.getcwd())
print("current_directory", current_directory)
# absolute_path = os.path.abspath("test.docx")
docx_path = f'{current_directory}/test.docx'

absolute_path = os.path.abspath(docx_path)
pdf_folder = os.path.abspath(current_directory)

convert_to_pdf = f"libreoffice --headless --convert-to pdf {absolute_path} --outdir {pdf_folder}"
subprocess.run(convert_to_pdf, shell=True)

