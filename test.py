"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json
from donut import DonutDataset


def test(args):
    pretrained_model = DonutModel.from_pretrained(
        args.pretrained_model_name_or_path
    )

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    predictions = []
    ground_truths = []
    accs = []

    evaluator = JSONParseEvaluator()
    # dataset = load_dataset(args.dataset_name_or_path, split=args.split)
    dataset = DonutDataset.load_dataset_multipage(
        args.dataset_name_or_path, split=args.split
    )

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # ground_truth = json.loads(sample["ground_truth"])

        ### Multipage ground truth
        ground_truth = dict()
        list_ques_answers = sample["question_answer"]
        gt = []
        [gt.extend(q_a_list) for q_a_list in list_ques_answers[:4]]
        ground_truth["gt_parses"] = gt
        ###############

        args.task_name = "vqa"

        num_pages = len(sample["page_name"])
        doc_name = sample["doc_name"]
        page_path = sample["page_path"]
        # C, H, W, P = 3, 2560, 1920, 5
        C, H, W, P = 3, 1280, 960, 4  ##########
        input_tensor = torch.zeros(size=(1, C, H, W, P))

        for i in range(min(num_pages, 4)):
            image_path = os.path.join(page_path, f"{doc_name}_page_{i}.jpg")
            image_tensor = Image.open(image_path)
            image_tensor = pretrained_model.encoder.prepare_input(
                image_tensor, random_padding=False
            )
            input_tensor[0, :, :, :, i] = image_tensor

        if "vqa" in args.task_name:
            output = pretrained_model.inference(
                image_tensors=input_tensor,
                prompt=f"<s_{args.task_name}><s_question>{ground_truth['gt_parses'][0]['question'].lower()}</s_question><s_answer>",
            )["predictions"][0]
        else:
            output = pretrained_model.inference(
                image=sample["image"], prompt=f"<s_{args.task_name}>"
            )["predictions"][0]

        if args.task_name == "rvlcdip":
            gt = ground_truth["gt_parse"]
            score = float(output["class"] == gt["class"])
        elif "vqa" in args.task_name:
            # Note: we evaluated the model on the official website.
            # In this script, an exact-match based score will be returned instead
            gt = ground_truth["gt_parses"]
            answers = set([qa_parse["answer"] for qa_parse in gt])
            score = float(output["answer"] in answers)
        else:
            gt = ground_truth["gt_parse"]
            score = evaluator.cal_acc(output, gt)

        accs.append(score)

        predictions.append(output)
        ground_truths.append(gt)

    scores = {
        "ted_accuracies": accs,
        "ted_accuracy": np.mean(accs),
        "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
    }
    print(
        f"Total number of samples: {len(accs)}, Tree Edit Distance (TED) based accuracy score: {scores['ted_accuracy']}, F1 accuracy score: {scores['f1_accuracy']}"
    )

    if args.save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        save_json(args.save_path, scores)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    predictions = test(args)
