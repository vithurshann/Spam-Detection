# LightGBM for spam classification


Language: Python
IDE used: Visual Studio Code

1) Libraries used to build the code:

hyperopt==0.2.7 <br>
lightgbm==3.3.5 <br>
matplotlib==3.6.0 <br>
numpy==1.24.2 <br>
pandas==1.5.0 <br>
scikit_learn==1.2.2 <br>
seaborn==0.12.2 <br>

To produce results of LIGHTGBM baseline:
python3 fraud_classification.py --classifier lgbmc

To produce results of LIGHTGBM tuned:
python3 fraud_classification.py --classifier lgbmc-tuned

To produce results of Linear Regressor:
python3 fraud_classification.py --classifier logistic
