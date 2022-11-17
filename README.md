# van-leerdam-et-al

# Predicting hypocalceamia using sensor data 

This Github page provides the codes used for the paper: A predictive model for hypocalcemia in dairy cows utilizing behavioural sensor data combined with deep learning. The code consists of multiple notebooks. One notebook used for feature preparation, one used for upsampling to balance the dataset and one for the actual model formation. 

In order to give some background to the code, hereby the abstract beloning to the article. 

(Sub)clinical hypocalcemia occurs frequently in the dairy industry, and is one of the earliest symptoms of an impaired transition period. Calcium deficiency is accompanied by changes in cows' daily behavioural variables, which can be measured by sensors. 
The goal of this study was to construct a predictive model to identify cows at risk of hypocalcemia in dairy cows using behavioural sensor data. For this study 133 primiparous and 476 multiparous cows from 8 commercial Dutch dairy farms were equipped with neck and leg sensors measuring behavioural elements, including eating, ruminating, lying and walking behaviour of the 21 days before calving and the day of calving. From each cow, a blood sample was taken to measure their blood calcium value. Cows with a blood calcium value $\leq$ 2,0 mmol/L, within 48 hours after calving, were defined as hypocalcaemic. In order to correct for dependencies in the data, a second way of dividing the blood calcium status into two groups was proposed, using a linear mixed-effects model with a k-Means clustering. Three possible binary predictive models were tested; a logistic regression model, a XgBoost model and a LSTM deep learning model. The deep learning model was expanded by adding the following static features as input variables; parity (1, 2 or 3+), calving season, day of blood measurements (0, 1 or 2), BCS and Locomotion score. Of the three models, the deep learning model performed best with an AUC of 0,66 and an average precision of 0.53. This final model was constructed with the addition of the static features, since they improved the model's tuning AUC by 12\%. The calcium label with the cut-off value proved to be easier to predict for the neural network and the XgBoost, while the logistic regression model performed better using the clustered labels. An ameliorated version of the deep learning model proposed in this study could serve as a tool to monitor herd calcium status and to identify animals at risk for associated transition diseases.
