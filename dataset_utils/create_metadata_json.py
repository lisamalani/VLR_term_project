import json
import os
from tqdm import tqdm

save_path = 'CUAD_v1_images/'
json_file = 'CUAD_v1_images/metadata.jsonl'
text_file_loc = ("CUAD_v1/full_contract_txt/")
images = os.listdir(save_path) 


def read_text(image_file_name):

    txt_file_name, page_no = image_file_name.rsplit("page_", 1)
    if ".PDF" not in txt_file_name:
        txt_file_name = txt_file_name[:-2] + ".txt"
    else:
        txt_file_name = txt_file_name.replace(".PDF", ".txt")        

    page_no = int(page_no.replace(".jpg", ""))

    document_data = []
    empty_line_count = 0

    assert os.path.isfile(text_file_loc + txt_file_name)

    for i, line in enumerate(open(text_file_loc + txt_file_name, "r")):
        if i == 0:
            document_data.append("")

        if line == "\n":
            empty_line_count += 1
            if empty_line_count == 5:
                if document_data[-1] != "":
                    document_data.append("")

            continue

        document_data[-1] += line
        empty_line_count = 0

    for i in range(len(document_data)):
        image_base_filename = filename.replace(".txt", f"_page_{i}.jpg")
        image_base_filename_PDF = filename.replace(".txt", f".PDFpage_{i}.jpg")

        if not (
            os.path.isfile(image_file_loc + image_base_filename)
            or os.path.isfile(image_file_loc + image_base_filename_PDF)
        ):
            valid_pdfs -= 1
            break


with open(json_file, "w") as f:
    
    for image in images:
        data = {"file_name": image, "ground_truth": ""}
        f.write(json.dumps(data)+"\n")
