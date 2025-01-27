# van-leerdam-et-al 2024

# Predicting hypocalceamia using sensor data 

This Github page provides the codes used for the paper: A predictive model for hypocalcemia in dairy cows utilizing behavioural sensor data combined with deep learning. The code consists of multiple notebooks. One notebook used for feature preparation, one used for upsampling to balance the dataset and one for the actual model formation. The codes were written on databricks in a combination of python, pyspark and a touch of R in the feature preperation file. When these files are opened on databricks they will be converted to notebooks, which might be easier to read.  

In order to give some background to the code, hereby the abstract beloning to the article. 

(Sub)clinical hypocalcaemia occurs frequently in the dairy industry, and is one of the earliest
symptoms of an impaired transition period. Calcium deficiency is accompanied by changes in
cows’ daily behavioural variables, which can be measured by sensors. The goal of this study
was to construct a predictive model to identify cows at risk of hypocalcaemia in dairy cows
using behavioural sensor data. For this study 133 primiparous and 476 multiparous cows from
8 commercial Dutch dairy farms were equipped with neck and leg sensors measuring daily
behavioural parameters, including eating, ruminating, standing, lying, and walking behaviour
of the 21 days before calving. From each cow, a blood sample was taken within 48 hours after
calving to measure their blood calcium concentration. Cows with a blood calcium concentration
≤ 2.0 mmol/L were defined as hypocalcemic. In order to create a more context based cut-off,
a second way of dividing the calcium concentrations into two categories was proposed, using a
linear mixed-effects model with a k-Means clustering. Three possible binary predictive models
were tested; a logistic regression model, a XgBoost model and a LSTM deep learning model.
The models were expanded by adding the following static features as input variables; parity (1,
2 or 3+), calving season (summer, autumn, winter, spring), day of calcium sampling relative
to calving (0, 1 or 2), body condition score and locomotion score. Of the three models, the
deep learning model performed best with an area under the receiver operating characteristic
curve (AUC) of 0.71 and an average precision of 0.47. This final model was constructed with
the addition of the static features, since they improved the model’s tuning AUC with 0.11. The
calcium label based on the cut-off categorization method proved to be easier to predict for the
models compared to the categorization method with the k-means clustering. This study provides
a novel approach for the prediction of hypocalcaemia, and an ameliorated version of the deep
learning model proposed in this study could serve as a tool to help monitor herd calcium status
and to identify animals at risk for associated transition diseases.

This article is my master thesis for veterinairy medicine and this was my first taste of programming. I would love to hear what you think of it.

The latest addition consist of code for an ontology made to make the data publicly available in the future, to contribute to open science as defined by the UNESCO recommendation on Open Science. The ontology was developed in order to make the structure and concepts of the data more comprehensible and to make it easier to extend the dataset with external data, thereby hopefully facilitating future research. The structure of the ontology is visualized in a figure which can also be found on this page.

The data of the Smarttag sensors was provided by the company Nedap (Nedap, Groenlo, the Netherlands).
