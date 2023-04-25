"""
dataset_name
├── test
│   ├── metadata.jsonl
│   ├── {image_path0}
│   ├── {image_path1}
│             .
│             .
├── train
│   ├── metadata.jsonl
│   ├── {image_path0}
│   ├── {image_path1}
│             .
│             .
└── validation
    ├── metadata.jsonl
    ├── {image_path0}
    ├── {image_path1}
              .
              .

> cat dataset_name/test/metadata.jsonl
{"file_name": {image_path0}, "ground_truth": "{\"gt_parse\": {ground_truth_parse}, ... {other_metadata_not_used} ... }"}
{"file_name": {image_path1}, "ground_truth": "{\"gt_parse\": {ground_truth_parse}, ... {other_metadata_not_used} ... }"}
     .
     .

The gt_parses follows the format of 
[
    {"question" : {question_sentence}, "answer" : {answer_candidate_1}}, 
    {"question" : {question_sentence}, "answer" : {answer_candidate_2}}, ...
],

for example, 
[
    {"question" : "what is the model name?", "answer" : "donut"}, 
    {"question" : "what is the model name?", "answer" : "document understanding transformer"}
].
"""

import csv
import json
import os
from tqdm import tqdm
import shutil

save_path = "term_project/CUAD_single_page_copies/vqa_dataset/"  # where dataset splits will exist
single_image_path = "term_project/CUAD_v1_single_images/"  # where documents are split into single pages and saved as images with page_#.jpg suffix
dataset_splits = ["train", "validation", "test"]

# Splits for training
for split in dataset_splits:
    dir_path = os.path.join(save_path, split, "images")
    os.makedirs(dir_path, exist_ok=True)

# CUAD single page VQA CSV
cuad_single_vqa_csv_path = "term_project/CUAD_single_page/cuad_vqa_with_page_numbers.csv"

cuad_vqa = {}
with open(cuad_single_vqa_csv_path, "r") as f:
    csv_reader = csv.DictReader(f)

    for row in tqdm(csv_reader):
        # Document and page number
        if row["page_number"] == "":
            continue

        page_num = int(float(row["page_number"]))
        doc_name = row["title"]  # title

        if doc_name not in cuad_vqa:
            cuad_vqa[doc_name] = {}

        # Image path
        image_name = doc_name.strip() + f"_page_{int(page_num)}.png"

        image_path = os.path.join(single_image_path, image_name)

        # if not os.path.isfile(image_path):
        #     print(doc_name)
        #     print(image_name)
        # assert False
        question = row["question"]
        answer = row["answer"]
        if row["is_impossible"] == "True":
            answer = "No answer"

        QA = {"question": question, "answer": answer}
        if image_name not in cuad_vqa:
            cuad_vqa[doc_name][image_name] = []

        cuad_vqa[doc_name][image_name].append(QA)

doc_names = list(cuad_vqa.keys())

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
            for image_name, QAs in cuad_vqa[split_doc].items():
                if os.path.isfile(os.path.join(single_image_path, image_name)):
                    metadata_file.write(
                        json.dumps(
                            {
                                "file_name": f"images/{image_name}",
                                "ground_truth": {
                                    "gt_parses": QAs,
                                },
                            }
                        )
                        + "\n"
                    )
                    # os.symlink(
                    #     os.path.join(single_image_path, image_name),
                    #     os.path.join(save_path, split_type, "images", image_name),
                    # )
                    shutil.copy(
                        os.path.join(single_image_path, image_name),
                        os.path.join(
                            save_path, split_type, "images", image_name
                        ),
                    )
