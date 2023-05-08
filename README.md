<div align="center">
    
# OCR-Free Recognition Models for Document Understanding & VQA

Robin Bhoo, Mansi Gandhi, Monalisa Malani, Chirag Mehta

16-824: Visual Learning and Recognition Term Project - Spring 2023

</div>

## Abstract
Our project attempts to address the challenge of understanding long legal documents by implementing an OCR-free, multi-page document recognition model. We use a pre-trained Donut encoder and decoder, which are Transformer-based architectures, and train a meta-encoder that captures the encoder embedding of multiple pages and merges them into one. The computational limitations only allowed us to explore up to 4 pages with degraded resolution. In addition, our workaround for the computational limits inhibited the use of the pre-trained weights for the Donut backbone. However, our work shows the possibility of reasoning across multi-page documents.

Our model, a variation of the Donut model, is an OCR-free method used to perform document understanding on multi-page documents with sparse information. We have performed training and testing using the CUAD dataset. See our report submission for more details about our project and the results.

The following repository contains the code for the model we used to perform document understanding for long legal documents.

## Software installation

Reference the Donut repo for setup instructions.

### How to run

1. Download the [CUAD](https://www.atticusprojectai.org/cuad) dataset to use as the source of data.

2. Create dataset in the format `DonutDataset` expects. To do so you will have to run the files in dataset_utils. First run `pdf2images.py` to extract all the images from all the PDFs. Then run `create_text_extraction_metadata_json.py` `cuad_single_vqa_metadata_json.py` or `cuad_multi_vqa_metadata_json.py` depending on which type of dataset you want to create. These 3 scripts create dataset in the format specified here (https://github.com/lisamalani/VLR_term_project#data).

3. Run training using the config yamls here (https://github.com/lisamalani/VLR_term_project/tree/main/config). There are three types of yamls: text extraction (train_cuad.yaml), single page QA (train_vqa_cuad_single.yaml) and multi page QA (train_vqa_cuad_multi.yaml). Use the following commands to launch:

```
python train.py --config config/train_cord.yaml \
                --pretrained_model_name_or_path "naver-clova-ix/donut-base" \
                --dataset_name_or_paths '["naver-clova-ix/cord-v2"]' \
                --exp_version "test_experiment"
```

```
python test.py --dataset_name_or_path naver-clova-ix/cord-v2 --pretrained_model_name_or_path ./result/train_cord/test_experiment --save_path ./result/output.json
```
