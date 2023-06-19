# LightGBM for spam classification


Language: Python
IDE used: Visual Studio Code

Instructions:

1) In order to run the code, please install the below libraries in your conda/python environment:

hyperopt==0.2.7
lightgbm==3.3.5
matplotlib==3.6.0
numpy==1.24.2
pandas==1.5.0
scikit_learn==1.2.2
seaborn==0.12.2

To reproduce results of LIGHTGBM baseline:
python3 fraud_classification.py --classifier lgbmc

To reproduce results of LIGHTGBM tuned:
python3 fraud_classification.py --classifier lgbmc-tuned

To reproduce results of Linear Regressor:
python3 fraud_classification.py --classifier logistic
