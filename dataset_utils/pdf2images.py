import os
from pdf2image import convert_from_path
from tqdm import tqdm

cuad_dataset_dir = 'CUAD_v1/full_contract_pdf/'
save_path = 'CUAD_v1_single_images/'
parts = os.listdir(cuad_dataset_dir) # part 1, 2, 3

for part in tqdm(parts):
    part_path = os.path.join(cuad_dataset_dir,part)
    contract_type_dirs = os.listdir(part_path)
    
    for contract_type in contract_type_dirs:
        contract_type_path = os.path.join(part_path, contract_type)
        pdf_files = os.listdir(contract_type_path)
        
        for file in pdf_files:
            file_path = os.path.join(contract_type_path, file)
            images = convert_from_path(file_path)

            for i in range(len(images)):
   
                # Save pages as images in the pdf
                image_name = file.replace('.pdf', '').replace('.PDF', '').strip().strip('-')
                image_name += f"_page_{i}.png"

                image_path = os.path.join(save_path, image_name)
                images[i].save(image_path)
