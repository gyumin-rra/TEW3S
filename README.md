# TEW3S: Timely Early Warning System for Septic Shock
Welcome to the TEW3S code repository, developed for the draft paper titled "A Development of Continuous Warning System for Timely Identification of Septic Shock." Please note that the content of this repository may change until the paper is published.

## 1. Dataset
For this research, we utilized the [MIMIC (Medical Information Mart for Intensive Care) - IV version 2.0](https://physionet.org/content/mimiciv/2.0/) dataset. Please be aware that this dataset is publicly available only for authorized users on PhysioNet.

## 2. Software Requirements
TEW3S is developed using Python, along with various Python-based modules. Below is a list of the required software environments and their versions:

| Environment Name | version |
|------------------|---------|
| python           | 3.9.16  |
| numpy            | 1.23.5  |
| matplotlib       | 3.4.3   |
| pandas           | 1.5.2   |
| sklearn          | 1.2.0   |
| IPython          | 8.10.0  |
| catboost         | 1.1.1   |
| lightgbm         | 3.2.1   |
| xgboost          | 1.7.6   |
| imblearn         | 0.10.1  |

## 3. How to Run the Code
The TEW3S building pipelines are provided in Jupyter Notebook (.ipynb) files. To run the code, please follow these steps:

1. Download all MIMIC datasets.
2. Download the Notebook files and place them in the same folder.
3. Make sure that all Notebook files are located outside the "icu" and "hosp" folders, which can be downloaded from PhysioNet.
4. Create a folder named "processed_data" in the same directory where the Notebook files are located.

All Notebook files are numbered, and they should be executed in order from 1.1 to 2.3.

## 4. Future Development Plans for This Repository
We intend to continue editing this repository until the paper is published. Our main plans for further changes in this repository include:

- Simplification and modularization of all pipelines to make them easier to run.
- Adding comments to important code sections in the pipelines.
- Improving efficiency and readability of the code.

Thank you for your interest in TEW3S. We welcome any feedback or contributions that can enhance the utility of this repository.
