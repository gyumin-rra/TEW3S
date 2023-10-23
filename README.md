# TEW3S
This is a code repository for draft paper "A Development of Continuous Warning System for Timely Identification of Septic Shock". TEW3S means Timely Early Warning System for Septic Shock, which is the resultant system proposed by the paper. The content of the respository is subject to change until the paper publication. 

## 1. Dataset
In this research, we utilized [MIMIC(Medical Information Mart for Intensive Care)-IV version 2.0](https://physionet.org/content/mimiciv/2.0/). Note that the dataset is publicly available only for credentialized users in physionet. 
## 2. Softwares
TEW3S is developed using python and the python-based modules listed as below.  
| environment name | version |
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
## 3. How to run the codes
The TEW3S building pipelines are now provided by Jupyter Notebook(.ipynb) files. Please download all MIMIC datasets and the Notebook files in same folder and make sure to locate all Notebook files outside of the icu and hosp folder which would be downloaded from the physionet. Additionally, please make a folder named as "processed_data" in the folder where the Notebook files are located.

All Notebook files are indexed with numbers, so the orders of the Notebook files goes from 1.1 to 2.3. Run the files accodrding to the order.

## 4. Future development plans for this repo
We are planning to edit this repository until the paper publication. Main plans for further changes in this repo includessimplification and modularization of all pipelines for them to be able to run by more simple procedures, the comments for important codes in the pipelines, and improvment in terms of efficiency or readibility.
