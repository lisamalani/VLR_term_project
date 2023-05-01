import os
import json
from tqdm import tqdm

single_image_path = "CUAD_v1_single_images/"
text_file_loc = "CUAD_v1/full_contract_txt/"

save_path = "CUAD_text_extract/text_dataset/"  # where dataset splits will exist

doc_image_names = {}
for filename in os.listdir(single_image_path):
    doc_name = filename.rsplit("_", maxsplit=2)[0]
    if doc_name not in doc_image_names:
        doc_image_names[doc_name] = []
    doc_image_names[doc_name].append(filename)

dataset_splits = ["train", "validation", "test"]

# Splits for training
for split in dataset_splits:
    dir_path = os.path.join(save_path, split, "images")
    os.makedirs(dir_path, exist_ok=True)

valid_pdfs = 0
total_pdfs = 0

doc_data = {}
file_list = list(os.listdir(text_file_loc))
for filename in tqdm(file_list):
    total_pdfs += 1
    valid_pdfs += 1
    document_data = []
    empty_line_count = 0

    for i, line in enumerate(open(text_file_loc + filename, "r")):
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

    valid_doc = True
    for i in range(len(document_data)):
        image_base_filename = filename.replace(".txt", f"_page_{i}.png")
        image_base_filename_PDF = filename.replace(".txt", f".PDFpage_{i}.png")

        if not (
            os.path.isfile(single_image_path + image_base_filename)
            or os.path.isfile(single_image_path + image_base_filename_PDF)
        ):
            valid_doc = False
            valid_pdfs -= 1
            break

    if valid_doc:
        filename = filename.replace(".txt", "")
        doc_data[filename] = document_data


doc_names = list(doc_data.keys())

train_split, test_split, val_split = 0.8, 0.1, 0.1

train_docs = doc_names[: int(train_split * len(doc_names))]
test_docs = doc_names[
    int(train_split * len(doc_names)) : int(
        (train_split + test_split) * len(doc_names)
    )
]
val_docs = doc_names[int((train_split + test_split) * len(doc_names)) :]

dataset_docs = {
    "train": train_docs,
    "test": test_docs,
    "validation": val_docs,
}

for split_type, split_docs in dataset_docs.items():
    split_dir = os.path.join(save_path, split_type)
    with open(f"{split_dir}/metadata.jsonl", "w") as metadata_file:
        for split_doc in split_docs:
            for i, page_data in enumerate(doc_data[split_doc]):
                image_name = f"{split_doc}_page_{i}.png"
                metadata_file.write(
                    json.dumps(
                        {
                            "file_name": f"images/{image_name}",
                            "ground_truth": {
                                "gt_parse": {"text_sequence": page_data},
                            },
                        }
                    )
                )
                os.symlink(
                    os.path.join(single_image_path, image_name),
                    os.path.join(save_path, split_type, "images", image_name),
                )

print(valid_pdfs, total_pdfs)
