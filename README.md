# Latte: Improving Latex Recognition for Tables and Formulae with Iterative Refinement
This is the artifact of the AAAI 2025 submission "Latte: Improving Latex Recognition for Tables and Formulae with Iterative Refinement".

## Dependency
* Python 3.10.12
* PyTorch 2.1.1
* transformers 4.35.2
* pdflatex
* ImageMagick

## Structure
* **results:** We open-source Latte's generation and refinement results on both formulae and tables datasets.
    * **formulae**
        * **Latte_1.json:** Latte's generation results and evaluation on IMAGE2LATEX-100K
        * **Latte_2.json:** Latte's first round of refinement results and evaluation on IMAGETOLATEX-100K
        * **Latte_3.json:** Latte's second round of refinement results and evaluation on IMAGETOLATEX-100K
        * **Latte_4.json:** Latte's third round of refinement results and evaluation on IMAGETOLATEX-100K
    * **tables**
        * **Latte_1.json:** Latte's generation results and evaluation on TAB2LATEX
        * **Latte_2.json:** Latte's first round of refinement results and evaluation on TAB2LATEX
        * **Latte_3.json:** Latte's second round of refinement results and evaluation on TAB2LATEX
        * **Latte_4.json:** Latte's third round of refinement results and evaluation on TAB2LATEX
* **tab2latex:** The TAB2LATEX dataset we collected for end-to-end LaTeX table recognition, filling the blank of missing such a dataset. We hope this dataset can benefit the future exploration of LaTeX table recognition.
    * **train.jsonl:** Training data of TAB2LATEX
    * **validate.jsonl:** Validating data of TAB2LATEX
    * **test.jsonl:** Testing data of TAB2LATEX
