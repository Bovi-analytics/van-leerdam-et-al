# Databricks notebook source
# MAGIC %md #Visualise Confidence levels of mean difference 

# COMMAND ----------

import pandas as pd
import numpy as np
data = {'Model': ['Logistic regression vs XgBoost', 'Logistic regression vs Best NN', 'Xgboost vs NN Without Static', 'XgBoost vs Best NN', 'NN Without Static vs NN With All Static', 'NN Without Static vs NN With Small Static'],
       'Low': [0.045795983, 0.179473896, 0.031394023, 0.11117078, 0.048457189, 0.07724952],
       'High': [0.090204017, 0.220526104, 0.078605977, 0.15282922, 0.095542811, 0.12055048]
       }
CI = pd.DataFrame(data)
CI

# COMMAND ----------

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

plt.figure(facecolor='white')
plt.hlines(y=CI['Model'], xmin = CI['Low'], xmax = CI['High'], linewidth=4, alpha =0.8)
plt.hlines(y=['NN With Small Static vs NN With All Static'], xmin = -0.025757606, xmax = 0.015757606, color = 'red', linewidth=4, alpha =0.8)
plt.title("Confidence interval difference in mean AUC bootstraps", fontsize = 20)
plt.xlabel('Mean difference in AUC')
# plt.ylabel('the models to compare')
# plt.grid()
plt.axis([-0.03, 0.23, -1, 7])

plt.show()
