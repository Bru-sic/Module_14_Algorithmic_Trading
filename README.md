# Module_14_Algorithmic_Trading
Scope Implement an algorithmic trading strategy that uses machine learning to automate the trade decisions.


Author: Bruno Ivasic   
Date: 4 December 2023

# Submission  
Files associated with this solution:   
* [Main Jupyter Lab Notebook: machine_learning_trading_bot.ipynb](./Code/machine_learning_trading_bot.ipynb)

# Analysis Step 1: Tune the training algorithm by adjusting the size of the training dataset
The training window was increased from the baseline's 3 months period to 18 months (scenario 1) and then 24 months (scenario 2).

Sell (-1.0) precision (i.e. predicted sell signals that were actually correct) increased from 43% to 75% to 80% respectively suggesting the model was getting better at predicting when to sell.
Sell (-1.0) recall (i.e. actual sell signals identified correctly) dropped from 4% for the baseline to 0% in scenarios 1 and 2.

Buy (1.0) precision (i.e. predicted buy signals that were actually correct) remained constant at 56% for the baseline, scenario 1 and scenario 2.
Buy (1.0) recall (i.e. actual buy signals identified correctly) improved from 96% for the base line to 100% for both scenario 1 and scenario 2.

The predicted returns increased from 151.76% to 164.27% and then dropped to 135.95% respectively.


## Baseline scenario using 3 months training window

***** Baseline: Training:3 months; Short SMA: 4 days; Long SMA: 100 days *****

Predicted strategy return: 151.76%
![Baseline: Training:3 months; Short SMA: 4 days; Long SMA: 100 days](../Images/Baseline_t_3m_sSMA_4d_lSMA_100d.png)

 1.0    2368
-1.0    1855
Name: Signal, dtype: int64


              precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092



***** Scenario 1: Training:18 months; Short SMA: 4 days; Long SMA: 100 days *****
Predicted strategy return: 164.76%

 1.0    2368
-1.0    1855

              precision    recall  f1-score   support

        -1.0       0.75      0.00      0.00      1430
         1.0       0.56      1.00      0.72      1843

    accuracy                           0.56      3273
   macro avg       0.66      0.50      0.36      3273
weighted avg       0.64      0.56      0.41      3273

![Scenario 1: Training:18 months; Short SMA: 4 days; Long SMA: 100 days](../Images/Scenario_1_t_18m_sSMA_4d_lSMA_100d.png)



***** Scenario 2: Training:24 months; Short SMA: 4 days; Long SMA: 100 days *****
Predicted strategy return: 164.27%



 1.0    2368
-1.0    1855

              precision    recall  f1-score   support

        -1.0       0.80      0.00      0.01      1229
         1.0       0.56      1.00      0.72      1565

    accuracy                           0.56      2794
   macro avg       0.68      0.50      0.36      2794
weighted avg       0.67      0.56      0.41      2794

![Scenario 2: Training:24 months; Short SMA: 4 days; Long SMA: 100 days](../Images/Scenario_2_t_24m_sSMA_4d_lSMA_100d.png)



***** Scenario 3: Training:3 months; Short SMA: 5 days; Long SMA: 100 days *****
Predicted strategy return: 135.95%

              precision    recall  f1-score   support

        -1.0       0.39      0.01      0.03      1804
         1.0       0.56      0.98      0.71      2288

    accuracy                           0.56      4092
   macro avg       0.47      0.50      0.37      4092
weighted avg       0.48      0.56      0.41      4092


![Scenario 3: Training:3 months; Short SMA: 5 days; Long SMA: 100 days](../Images/Scenario_3_t_3m_sSMA_5d_lSMA_100d.png)


***** Scenario 4: Training:3 months; Short SMA: 6 days; Long SMA: 100 days *****
Predicted strategy return: 129.78%


              precision    recall  f1-score   support

        -1.0       0.32      0.01      0.01      1804
         1.0       0.56      0.99      0.71      2288

    accuracy                           0.56      4092
   macro avg       0.44      0.50      0.36      4092
weighted avg       0.45      0.56      0.40      4092


![Scenario 4: Training:3 months; Short SMA: 6 days; Long SMA: 100 days](../Images/Scenario_4_t_3m_sSMA_6d_lSMA_100d.png)




***** Scenario 5: Training:3 months; Short SMA: 2 days; Long SMA: 100 days *****
Predicted strategy return: 140.90%

              precision    recall  f1-score   support

        -1.0       0.41      0.01      0.03      1804
         1.0       0.56      0.99      0.71      2288

    accuracy                           0.56      4092
   macro avg       0.49      0.50      0.37      4092
weighted avg       0.49      0.56      0.41      4092

![Scenario 5: Training:3 months; Short SMA: 2 days; Long SMA: 100 days](../Images/Scenario_5_t_3m_sSMA_2d_lSMA_100d.png)


***** Scenario 6: Training:3 months; Short SMA: 7 days; Long SMA: 100 days *****
Predicted strategy return: 128.94%


              precision    recall  f1-score   support

        -1.0       0.30      0.00      0.01      1804
         1.0       0.56      0.99      0.71      2288

    accuracy                           0.56      4092
   macro avg       0.43      0.50      0.36      4092
weighted avg       0.44      0.56      0.40      4092

![Scenario 6: Training:3 months; Short SMA: 7 days; Long SMA: 100 days](../Images/Scenario_6_t_3m_sSMA_7d_lSMA_100d.png)



***** Scenario 7: Training:3 months; Short SMA: 4 days; Long SMA: 60 days *****
Predicted strategy return: 129.66%

              precision    recall  f1-score   support

        -1.0       0.40      0.07      0.12      1817
         1.0       0.56      0.91      0.69      2302

    accuracy                           0.54      4119
   macro avg       0.48      0.49      0.41      4119
weighted avg       0.49      0.54      0.44      4119


![Scenario 7: Training:3 months; Short SMA: 4 days; Long SMA: 60 days](../Images/Scenario_7_t_3m_sSMA_4d_lSMA_60d.png)



***** Scenario 8: Training:3 months; Short SMA: 4 days; Long SMA: 90 days *****
Predicted strategy return: 154.18%

              precision    recall  f1-score   support

        -1.0       0.42      0.05      0.09      1806
         1.0       0.56      0.94      0.70      2288

    accuracy                           0.55      4094
   macro avg       0.49      0.50      0.40      4094
weighted avg       0.50      0.55      0.43      4094


![Scenario 8: Training:3 months; Short SMA: 4 days; Long SMA: 90 days](../Images/Scenario_8_t_3m_sSMA_4d_lSMA_90d.png)



***** Scenario 9: Training:3 months; Short SMA: 4 days; Long SMA: 120 days *****

Predicted strategy return: 144.74%


              precision    recall  f1-score   support

        -1.0       0.41      0.05      0.09      1793
         1.0       0.56      0.94      0.70      2284

    accuracy                           0.55      4077
   macro avg       0.49      0.50      0.40      4077
weighted avg       0.49      0.55      0.43      4077



![Scenario 9: Training:3 months; Short SMA: 4 days; Long SMA: 120 days](../Images/Scenario_9_t_3m_sSMA_4d_lSMA_120d.png)



***** Scenario 10: Training:18 months; Short SMA: 2 days; Long SMA: 90 days *****
Predicted strategy return: 183.03%

              precision    recall  f1-score   support

        -1.0       0.64      0.00      0.01      1430
         1.0       0.56      1.00      0.72      1843

    accuracy                           0.56      3273
   macro avg       0.60      0.50      0.37      3273
weighted avg       0.60      0.56      0.41      3273



![Scenario 10: Training:18 months; Short SMA: 2 days; Long SMA: 90 days](../Images/Scenario_10_t_18m_sSMA_2d_lSMA_90d.png)


## Analysis Step 2: Tune the trading algorithm by adjusting the SMA input features. 

**Response:**
The short and long Simple Moving Average window sizes were adjusted with the following results:

|Scenario|Short SMA (d)|Long SMA (d)|Sell Precision|Sell Recall|Sell Precision|Sell Recall|Return|
|--|--|--|--|--|--|--|--|
|3|5|100|0.39|0.01|0.56|0.98|135.95%|
|4|6|100|0.32|0.01|0.56|0.99|129.78%|
|5|2|100|0.41|0.01|0.56|0.99|140.90%|
|6|7|100|0.30|0.00|0.56|0.99|128.94%|
|7|4|060|0.40|0.07|0.56|0.91|129.66%|
|8|4|090|0.42|0.05|0.56|0.94|154.18%|
|9|4|120|0.41|0.05|0.56|0.94|144.74%|
|10*|4|090|0.64|0.00|0.56|1.00|183.03%|

Note: * A training window of 18 months was used for scenario 10 as this appeared to be ideal based on tests in Step 1. A training window of 3 months was used for scenarios 3 to 9 to maintain a constant comparison while other variables to changed. 


A hypothesis that using a combination of the best resulting parameters from previous individual variable tests would also yield the best results did eventuate in terms of returns, having a 31.27 basis points improvement compared to the baseline from 151.76% to 183.03% for scenario 10. 


***** TO DO ***** Write your conclusions about the performance of the baseline trading algorithm in the README.md file that’s associated with your GitHub repository. Support your findings by using the PNG image that you saved in the previous step.


# Dependencies  (********TO BE UPDATED********)

############## * pip version 19.0+ (or 20.3+ for macOS) needed by TensorFlow 2 packages
######### * Tensorflow 2 version 2.5.0+
* scikit-learn version 1.2+
* hvplot
* matplotlib
RuntimeError: To use bokeh.io image export functions you need selenium ('conda install selenium' or 'pip install selenium')

# Installation / Setup   (********TO BE UPDATED ********)
RuntimeError: To use bokeh.io image export functions you need selenium ('conda install selenium' or 'pip install selenium')

The following are required:
```
pip install --upgrade pip

pip install --upgrade scikit-learn

pip install --upgrade tensorflow
```

Note: additional steps are required if using an Apple Silicon Mac (see link below)

Links to further information:
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [TensorFlow](https://www.tensorflow.org/install)
* [Guide if using an Apple Silicon Mac (with M1 and M+ architecture)](https://www.mrdbourke.com/setup-apple-m1-pro-and-m1-max-for-machine-learning-and-data-science/)


# Breifing
* [Assignment Breifing](./Briefing/Briefing.md)

---

# Citations
* 