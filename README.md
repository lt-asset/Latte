# LATTE: Improving Latex Recognition for Tables and Formulae with Iterative Refinement
This is the artifact of the AAAI 2025 paper "LATTE: Improving Latex Recognition for Tables and Formulae with Iterative Refinement".

## Dependency
* Python 3.10.12
* PyTorch 2.1.1
* transformers 4.35.2
* pdflatex
* ImageMagick

## Structure
* **results:** We open-source Latte's generation and refinement results on both formulae and tables datasets.
    * **formula**
        * **Latte_1.json:** Latte's generation results and evaluation on IMG2LATEX-100K
        * **Latte_2.json:** Latte's first round of refinement results and evaluation on IMG2LATEX-100K
        * **Latte_3.json:** Latte's second round of refinement results and evaluation on IMG2LATEX-100K
        * **Latte_4.json:** Latte's third round of refinement results and evaluation on IMG2LATEX-100K
    * **table**
        * **Latte_1.json:** Latte's generation results and evaluation on TAB2LATEX
        * **Latte_2.json:** Latte's first round of refinement results and evaluation on TAB2LATEX
        * **Latte_3.json:** Latte's second round of refinement results and evaluation on TAB2LATEX
        * **Latte_4.json:** Latte's third round of refinement results and evaluation on TAB2LATEX
* **tab2latex:** The TAB2LATEX dataset we collected for end-to-end LaTeX table recognition, filling the blank of missing such a dataset. We hope this dataset can benefit the future exploration of LaTeX table recognition.
    * **train.jsonl:** Training data of TAB2LATEX
    * **validate.jsonl:** Validating data of TAB2LATEX
    * **test.jsonl:** Testing data of TAB2LATEX
* **im2latex:** The IMG2LATEX-100K dataset for end-to-end LaTeX formula recognition.
    * **train.jsonl:** Training data of IMG2LATEX-100K
    * **validate.jsonl:** Validating data of IMG2LATEX-100K
    * **test.jsonl:** Testing data of IMG2LATEX-100K


## Usage

### Preparing Ground-Truth Images Data
For formular recognition using im2latex-100k, run:
```bash
cd src/data
# Rendering test data in im2latex
python render_formula_ground_truth.py --filename ../../im2latex/test.jsonl --output_dir ../../im2latex/test/ --image_width 1344 --image_height 224 --dpi 240 --base_tmp_dir /tmp/ --pad
# Rendering validation data in im2latex
python render_formula_ground_truth.py --filename ../../im2latex/validate.jsonl --output_dir ../../im2latex/validte/ --image_width 1344 --image_height 224 --dpi 240 --base_tmp_dir /tmp/ --pad
# Rendering training data in im2latex
python render_formula_ground_truth.py --filename ../../im2latex/train.jsonl --output_dir ../../im2latex/train/ --image_width 1344 --image_height 224 --dpi 240 --base_tmp_dir /tmp/ --pad
```

For table recognition using tab2latex, run:
```bash
cd src/data
# Rendering test data in tab2latex
python render_table_ground_truth.py --filename ../../tab2latex/test.jsonl --output_dir ../../tab2latex/test/ --image_width 1344 --image_height 672 --dpi 160 --base_tmp_dir /tmp/ --pad
# Rendering validation data in tab2latex
python render_table_ground_truth.py --filename ../../tab2latex/validate.jsonl --output_dir ../../tab2latex/validte/ --image_width 1344 --image_height 672 --dpi 160 --base_tmp_dir /tmp/ --pad
# Rendering training data in tab2latex
python render_table_ground_truth.py --filename ../../tab2latex/train.jsonl --output_dir ../../tab2latex/train/ --image_width 1344 --image_height 672 --dpi 160 --base_tmp_dir /tmp/ --pad
```

### Traing Generation Model
To train formula generation model, run:
```bash
cd src/generation
deepspeed --num_gpus=4 train_formula_generation.py

# Convert deepspeed checkpoints to huggingface checkpoints if you use zero0, zero1 or zero2. Using the script provided by deepspeed if you use zero3.
python zero012_to_hf_ckpt.py --ckpt_dir ../../models/ds_ckpts/global_stepXXX/ --output_dir ../../models/formula-generation --image_width 1344 --image_height 224
```

To train table generation model, run:
```bash
cd src/generation
deepspeed --num_gpus=4 train_table_generation.py

# Convert deepspeed checkpoints to huggingface checkpoints if you use zero0, zero1 or zero2. Using the script provided by deepspeed if you use zero3.
python zero012_to_hf_ckpt.py --ckpt_dir ../../models/ds_ckpts/global_stepXXX/ --output_dir ../../models/table-generation --image_width 1344 --image_height 672
```

### Inference Generation Model
To inference the formula generation model, run:
```bash
cd src/generation
python inference_formula_generation.py --model_path ../../models/formula-generation --input_path ../../im2latex/test.jsonl --image_dir ../../im2latex/test/ --output_path ../../results/formula/Latte_1.json
```

To inference the table generation model, run:
```bash
cd src/generation
python inference_table_generation.py --model_path ../../models/table-generation --input_path ../../tab2latex/test.jsonl --image_dir ../../tab2latex/test/ --output_path ../../results/table/Latte_1.json
```
