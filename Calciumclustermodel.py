# Databricks notebook source
# MAGIC %md #Calciumclustermodel

# COMMAND ----------

# DBTITLE 1,Train set
Imputed_train_set_spark = spark \
  .read \
  .option('inferschema', 'true') \
  .option("header","true") \
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/train_set")

# COMMAND ----------

# DBTITLE 1,upsampled combined val train set 
Imputed_trainValidation_up_spark = spark\
  .read \
  .option('inferschema', 'true') \
  .option("header","true")\
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/train_val_set_up")

# COMMAND ----------

# DBTITLE 1,upsampeld train set
Imputed_train_set_up_spark = spark \
  .read \
  .option('inferschema', 'true') \
  .option("header","true") \
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/train_set_up")

# COMMAND ----------

# DBTITLE 1,Validation set
Imputed_validation_set_spark = spark \
  .read \
  .option('inferschema', 'true') \
  .option("header","true") \
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/validatie_set")

# COMMAND ----------

# DBTITLE 1,Test set
Imputed_test_set_spark = spark \
  .read \
  .option('inferschema', 'true') \
  .option("header","true") \
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/test_set")

# COMMAND ----------

# MAGIC %md #Defining x and y values

# COMMAND ----------

#transforming spark dataframe into a pandas dataframe
Imputed_train_set = Imputed_train_set_spark.toPandas()
Imputed_train_set_up = Imputed_train_set_up_spark.toPandas()
Imputed_validation_set = Imputed_validation_set_spark.toPandas() 
Imputed_test_set = Imputed_test_set_spark.toPandas()
# Imputed_trainANDvalidation_up = Imputed_trainValidation_up_spark.toPandas()

# COMMAND ----------

# MAGIC %md ##Train set

# COMMAND ----------

#Extracting x and y-values as model input. Sequential sensordata is stored in a 3d matrix of 22 days per 19 features per 365 cows.
import numpy as np
import pandas as pd
#extracting all unique combinations of cow and calving moment 
Unique_Calvings = Imputed_train_set[['AnimalEartag', 'PaperRecordedCalvingDate']].drop_duplicates() 

#define sensors
feature_names =["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"]

#create empty lists per feature to store variables
AnimalEartag_list = [] 
PaperRecordedCalvingDate_list = []
CalciumValue_list = []
Calciumcluster_list = []
BCS_lijst = []
Loco_lijst = []
SensorWaardes_list = np.zeros((365,22,19)) #define matrix size for sequential features
j = 0

#itterating through the dataset in order to extract x and y for each cow 
for index, (AnimalEartag, PaperRecordedCalvingDate) in Unique_Calvings.iterrows():
    filter1 = Imputed_train_set['AnimalEartag'] == AnimalEartag
    filter2 = PaperRecordedCalvingDate == Imputed_train_set['PaperRecordedCalvingDate']
    df_calving = Imputed_train_set[filter1 & filter2]
    Loco = df_calving['FirstLocomotionScore'].max()
    BCS =  df_calving['FirstBCSScore'].max()
    cacluster = df_calving['Calciumcluster'].iloc[-1] 
    ca = df_calving['Cut_Off'].iloc[-1]
    sw = df_calving[feature_names]
  
    #convert to numpy
    sw_numpy = np.array(sw)
    #add to list 
    AnimalEartag_list.append(AnimalEartag)
    PaperRecordedCalvingDate_list.append(PaperRecordedCalvingDate)
    CalciumValue_list.append(ca)
    Calciumcluster_list.append(cacluster)
    BCS_lijst.append((BCS))
    Loco_lijst.append((Loco))
   
    SensorWaardes_list[j] = sw_numpy
    j = j + 1
 
#convert to numpy arrays
x_train = np.array(SensorWaardes_list)
y_train = np.asarray(Calciumcluster_list)
y_train2 = np.asarray(CalciumValue_list) #alternatieve y-waardes op basis van een calciumcut-off
x_BCS = np.asarray(BCS_lijst)
x_Loco = np.asarray(Loco_lijst)
x_static = np.stack((x_BCS, x_Loco), axis = 1)

# COMMAND ----------

# DBTITLE 1,Onehotencoder
#Calvingseason and parity are cathegorial variables, however the models functions on numerical variables only, therefore the variables are converted into binairy variables using the sklearn onehotencoder. Each cathegory gets its own colummn and the column of the cathegory that is true gets a one, all other get a zero.     
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
static = Imputed_train_set.groupby(['AnimalEartag', 'PaperRecordedCalvingDate']).first()
season = static['CalvingSeason'].to_numpy()
Parity = static['Parity'].to_numpy()
CalciumDaysInMilk = static['CalciumDaysInMilk'].to_numpy()
labelEnc = LabelEncoder()
x_enc = labelEnc.fit_transform(season)
x_enc = x_enc.reshape(len(x_enc), 1) 
onehotEnc = OneHotEncoder(sparse=False)
season_encoded = onehotEnc.fit_transform(x_enc)
x_enc_P = labelEnc.fit_transform(Parity)
x_enc_P = x_enc_P.reshape(len(x_enc_P), 1) 
parity_encoded = onehotEnc.fit_transform(x_enc_P)
x_static_lean = np.column_stack([season_encoded, parity_encoded, CalciumDaysInMilk])

# COMMAND ----------

# DBTITLE 1,missing value imputation van BCS en Loco 
#BCS and Locomotion score was not measured for every cow, therefore some cows have a null value. The null values are replaced by the value that is most frequent in that colummn. 
import numpy as np
from sklearn.impute import SimpleImputer
imp_freq = SimpleImputer(missing_values=0.0, strategy='most_frequent')
imp_freq.fit(x_static)
x_static = imp_freq.transform(x_static)
#all static features are added up into one numpy array 
x_Static = np.column_stack([x_static, season_encoded, parity_encoded, CalciumDaysInMilk]) 

# COMMAND ----------

# MAGIC %md ##Train set upsampeld 

# COMMAND ----------

#Since this train set is upsampled, it is impossible to filter on animaleartag and paperrecordedcalvingdate (some cows of cathegory 1 are duplicated and therefore no longer unique) therefore a new column was introduced while performing upsampling called 'Samplenumber' In order to extract an x and an y per sample, an itteration is performed based on samplenumber. 
Unique_Calvings = Imputed_train_set_up['SampleNumber'].drop_duplicates()

#defitie van de sensoren
feature_names =["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"]

#creating empty lists, pay attention to the increased matrix size and the addition of a list for samplenumber
SampleNumber_list = []
CalciumValue_list = []
Calciumcluster_list = []
SensorWaardes_list = np.zeros((534,22,19))
BCS_lijst = []
Loco_lijst = []
j = 0
for s in range(len(Unique_Calvings)):
    filter = Imputed_train_set_up['SampleNumber'] == s
    df_calving = Imputed_train_set_up[filter]
    Loco = df_calving['FirstLocomotionScore'].max()
    BCS =  df_calving['FirstBCSScore'].max()
    i = 1
    cacluster = df_calving['Calciumcluster'].iloc[-i]
    ca = df_calving['Cut_Off'].iloc[-i]
    i = i + 1 #nodig anders infiniteloop
  
    sw = df_calving[feature_names]
  
    #convert to numpy
    sw_numpy = np.array(sw)
    #add to list
    SampleNumber_list.append(s)
    CalciumValue_list.append(ca)
    Calciumcluster_list.append(cacluster)
    BCS_lijst.append((BCS))
    Loco_lijst.append((Loco))
  
    SensorWaardes_list[j] = sw_numpy
    j = j + 1

#transform to numpy array
x_train_up = np.array(SensorWaardes_list)
y_train_up = np.asarray(Calciumcluster_list)
y_train2_up =np.asarray(CalciumValue_list)
x_BCS_up = np.asarray(BCS_lijst) #BCS end dry period 
x_Loco_up = np.asarray(Loco_lijst) #locomotionscore end dry period
x_static_up = np.stack((x_BCS_up, x_Loco_up), axis = 1) #combining BCS and Locomotion in order to be able to perform upsampling efficiently

# COMMAND ----------

# DBTITLE 1,One hot encoding and imputing BCS  + Locomotionscore for the upsampeld trainset 
static = Imputed_train_set_up.groupby(['SampleNumber']).first()
season = static['CalvingSeason'].to_numpy()
Parity = static['Parity'].to_numpy()
CalciumDaysInMilk_up = static['CalciumDaysInMilk'].to_numpy()
labelEnc = LabelEncoder()
x_enc = labelEnc.fit_transform(season)
x_enc = x_enc.reshape(len(x_enc), 1) 
onehotEnc = OneHotEncoder(sparse=False)
season_encoded_up = onehotEnc.fit_transform(x_enc)
x_enc_P = labelEnc.fit_transform(Parity)
x_enc_P = x_enc_P.reshape(len(x_enc_P), 1) 
parity_encoded_up = onehotEnc.fit_transform(x_enc_P)
x_static_up = imp_freq.transform(x_static_up)
x_Static_up = np.column_stack([x_static_up, season_encoded_up, parity_encoded_up, CalciumDaysInMilk_up]) 
x_static_up_lean = np.column_stack([season_encoded_up, parity_encoded_up, CalciumDaysInMilk_up])

# COMMAND ----------

# MAGIC %md ##Train set and Validation set upsampeld combined

# COMMAND ----------

# #Since this train and validation set is upsampled, it is impossible to filter on animaleartag and paperrecordedcalvingdate (some cows of cathegory 1 are duplicated and therefore no longer unique) therefore a new column was introduced while performing upsampling called 'Samplenumber' In order to extract an x and an y per sample, an itteration is performed based on samplenumber. 
# Unique_Calvings = Imputed_trainANDvalidation_up['SampleNumber'].drop_duplicates()

# #define sensorvalues
# feature_names =["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"]

# #define empty list, bigger matrix since there are more upsampled animals 
# SampleNumber_list = []
# CalciumValue_list = []
# Calciumcluster_list = []
# SensorWaardes_list = np.zeros((730,22,19))
# BCS_lijst = []
# Loco_lijst = []
# j = 0
# for s in range(len(Unique_Calvings)):
#     filter = Imputed_trainANDvalidation_up['SampleNumber'] == s
#     df_calving = Imputed_trainANDvalidation_up[filter]
#     Loco = df_calving['FirstLocomotionScore'].max()
#     BCS =  df_calving['FirstBCSScore'].max()
#     i = 1
#     cacluster = df_calving['Calciumcluster'].iloc[-i]
#     ca = df_calving['Cut_Off'].iloc[-i]
#     i = i + 1 #nodig anders infiniteloop
  
#     sw = df_calving[feature_names]
  
#     #convert to numpy
#     sw_numpy = np.array(sw)
#     #add to list
#     SampleNumber_list.append(s)
#     CalciumValue_list.append(ca)
#     Calciumcluster_list.append(cacluster)
#     BCS_lijst.append((BCS))
#     Loco_lijst.append((Loco))
  
#     SensorWaardes_list[j] = sw_numpy
#     j = j + 1

# #converting to numpy matrix
# x_trainANDval_up = np.array(SensorWaardes_list)
# y_trainANDval_up = np.asarray(Calciumcluster_list)
# y_trainANDval2_up =np.asarray(CalciumValue_list)
# x_BCS_TV_up = np.asarray(BCS_lijst) #BCS end dry period 
# x_Loco_TV_up = np.asarray(Loco_lijst) #locomotionscore end dry period
# x_static_TV_up = np.stack((x_BCS_TV_up, x_Loco_TV_up), axis = 1) #combining BCS and Locomotion in order to be able to perform upsampling efficiently

# COMMAND ----------

# DBTITLE 1,One hot encoding and imputing BCS  + Locomotionscore for the combined upsampeld train and validation set 
# static = Imputed_trainANDvalidation_up.groupby(['SampleNumber']).first()
# season = static['CalvingSeason'].to_numpy()
# Parity = static['Parity'].to_numpy()
# CalciumDaysInMilk_up = static['CalciumDaysInMilk'].to_numpy()
# labelEnc = LabelEncoder()
# x_enc = labelEnc.fit_transform(season)
# x_enc = x_enc.reshape(len(x_enc), 1) 
# onehotEnc = OneHotEncoder(sparse=False)
# season_encoded_up = onehotEnc.fit_transform(x_enc)
# x_enc_P = labelEnc.fit_transform(Parity)
# x_enc_P = x_enc_P.reshape(len(x_enc_P), 1) 
# parity_encoded_up = onehotEnc.fit_transform(x_enc_P)
# x_static_TV_up = imp_freq.transform(x_static_TV_up)
# #all static features combined into one matrix
# x_Static_TV_up = np.column_stack([x_static_TV_up, season_encoded_up, parity_encoded_up, CalciumDaysInMilk_up]) 
# #since BCS and Locomotionscore measured by a veterinarian are the most unlikely features to be used in a real-life situation, I also made a static feature set without BCS and Locomotionscores.
# x_static_TV_up_lean = np.column_stack([season_encoded_up, parity_encoded_up, CalciumDaysInMilk_up])

# COMMAND ----------

# MAGIC %md ##Validation set

# COMMAND ----------

#extracting all unique combinations of cow and calving moment  
Unique_Calvings_Val = Imputed_validation_set[['AnimalEartag', 'PaperRecordedCalvingDate']].drop_duplicates()

#defitie van de sensoren
feature_names =["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"]

AnimalEartagV_list = []
PaperRecordedCalvingDateV_list = []
CalciumValueV_list = []
CalciumclusterV_list = []
BCS_lijst = []
Loco_lijst = []
SensorWaardesV_list = np.zeros((122,22,19))
j = 0
for index, (AnimalEartag, PaperRecordedCalvingDate) in Unique_Calvings_Val.iterrows():
    filter1 = Imputed_validation_set['AnimalEartag'] == AnimalEartag
    filter2 = PaperRecordedCalvingDate == Imputed_validation_set['PaperRecordedCalvingDate']
    df_calving_val = Imputed_validation_set[filter1 & filter2]
    i = 1
    Loco = df_calving_val['FirstLocomotionScore'].max()
    BCS =  df_calving_val['FirstBCSScore'].max()
    caV = df_calving_val['Cut_Off'].iloc[-i]
    caclusterV = df_calving_val['Calciumcluster'].iloc[-i]
    swV = df_calving_val[feature_names]
  
    #convert to numpy
    swV_numpy = np.array(swV)
    #add to list 
    AnimalEartagV_list.append(AnimalEartag)
    PaperRecordedCalvingDateV_list.append(PaperRecordedCalvingDate)
    CalciumValueV_list.append(caV)
    CalciumclusterV_list.append(caclusterV)
    BCS_lijst.append((BCS))
    Loco_lijst.append((Loco))
    
    SensorWaardesV_list[j] = swV_numpy
    j = j + 1


x_val = np.array(SensorWaardesV_list)
y_val2 = np.asarray(CalciumValueV_list)
y_val = np.asarray(CalciumclusterV_list)
x_BCS = np.asarray(BCS_lijst)
x_Loco = np.asarray(Loco_lijst)
x_static_val = np.stack((x_BCS, x_Loco), axis = 1)

# COMMAND ----------

# DBTITLE 1,Onehotencoder en Missing value imputation Validation set 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
static = Imputed_validation_set.groupby(['AnimalEartag', 'PaperRecordedCalvingDate']).first()
season = static['CalvingSeason'].to_numpy()
Parity = static['Parity'].to_numpy()
CalciumDaysInMilk = static['CalciumDaysInMilk'].to_numpy()
labelEnc = LabelEncoder()
x_enc = labelEnc.fit_transform(season)
x_enc = x_enc.reshape(len(x_enc), 1) 
onehotEnc = OneHotEncoder(sparse=False)
season_encoded = onehotEnc.fit_transform(x_enc)
x_enc_P = labelEnc.fit_transform(Parity)
x_enc_P = x_enc_P.reshape(len(x_enc_P), 1) 
parity_encoded = onehotEnc.fit_transform(x_enc_P)
x_static_val = imp_freq.transform(x_static_val)
x_Static_val = np.column_stack([x_static_val, season_encoded, parity_encoded,CalciumDaysInMilk]) 
x_static_val_lean = np.column_stack([season_encoded, parity_encoded, CalciumDaysInMilk])

# COMMAND ----------

# MAGIC %md ##Test set

# COMMAND ----------

#extracting all unique combinations of cow and calving moment
Unique_Calvings_Test = Imputed_test_set[['AnimalEartag', 'PaperRecordedCalvingDate']].drop_duplicates()

AnimalEartag_Test_list = []
PaperRecordedCalvingDate_Test_list = []
CalciumValue_Test_list = []
Calciumcluster_Test_list = []
BCS_lijst = []
Loco_lijst = []
SensorWaardes_Test_list = np.zeros((122,22,19))
k = 0
for index, (AnimalEartag, PaperRecordedCalvingDate) in Unique_Calvings_Test.iterrows():
    filter1 = Imputed_test_set['AnimalEartag'] == AnimalEartag
    filter2 = PaperRecordedCalvingDate == Imputed_test_set['PaperRecordedCalvingDate']
    df_calving_test = Imputed_test_set[filter1 & filter2]
    Loco = df_calving_test['FirstLocomotionScore'].max()
    BCS =  df_calving_test['FirstBCSScore'].max()
    i = 1
    catest = df_calving_test['Cut_Off'].iloc[-i]
    caclustertest = df_calving_test['Calciumcluster'].iloc[-i]
    a = df_calving_test['CalvingSeason'].dropna()
  
    swtest = df_calving_test[feature_names]

    #convert to numpy
    swtest_numpy = np.array(swtest)

    #add to list 
    AnimalEartag_Test_list.append(AnimalEartag)
    PaperRecordedCalvingDate_Test_list.append(PaperRecordedCalvingDate)
    CalciumValue_Test_list.append(catest)
    Calciumcluster_Test_list.append(caclustertest)
    BCS_lijst.append((BCS))
    Loco_lijst.append((Loco))
    SensorWaardes_Test_list[k] = swtest_numpy
    k = k + 1


x_test = np.asarray(SensorWaardes_Test_list)
y_test2 = np.asarray(CalciumValue_Test_list)
y_test = np.asarray(Calciumcluster_Test_list)
x_BCS = np.asarray(BCS_lijst)
x_Loco = np.asarray(Loco_lijst)
x_static_test = np.stack((x_BCS, x_Loco), axis = 1)
x_static_test = imp_freq.transform(x_static_test)

# COMMAND ----------

# DBTITLE 1,Onehotencoder en Missing value imputation Test set 
static = Imputed_test_set.groupby(['AnimalEartag', 'PaperRecordedCalvingDate']).first()
season = static['CalvingSeason'].to_numpy()
Parity = static['Parity'].to_numpy()
CalciumDaysInMilk = static['CalciumDaysInMilk'].to_numpy()
labelEnc = LabelEncoder()
x_enc = labelEnc.fit_transform(season)
x_enc = x_enc.reshape(len(x_enc), 1) 
onehotEnc = OneHotEncoder(sparse=False)
season_encoded = onehotEnc.fit_transform(x_enc)
x_enc_P = labelEnc.fit_transform(Parity)
x_enc_P = x_enc_P.reshape(len(x_enc_P), 1) 
parity_encoded = onehotEnc.fit_transform(x_enc_P)
x_Static_test = np.column_stack([x_static_test, season_encoded, parity_encoded, CalciumDaysInMilk]) 
x_static_test_lean = np.column_stack([season_encoded, parity_encoded, CalciumDaysInMilk])

# COMMAND ----------

# DBTITLE 1,combine validation and train set 
#om de resultaten van de testset te verbeteren combineren we de validatie en trainset om zo meer data te hebben om op te trainen, upgesampled zit los want daarin is de validatie set ook ge-upsampeld 
x_valANDtrain = np.vstack((x_train, x_val))
y_valANDtrain = np.hstack((y_train, y_val))
y_valANDtrain2 = np.hstack((y_train2, y_val2))

# COMMAND ----------

# MAGIC %md #Bootstrapping 

# COMMAND ----------

#XgBoost and Deep learning models have a form of randomness in their initialisation. The exact same model structure with the same training data may therefore give different results each time it is run. In order to be able to compare different models, model performance is not objective enough since it can differ with the same model and could be a high value simply because you were lucky. To compansate for this behaviour a second metric is proposed to compare models; the variance of the model. When a model has high variance, the chances are higher that a high value is obtained by luck and it could be that the second time the model is run, the exact same model results in dramaticly low performance metrics. To be able to measure varriance bootstraps are built, bootstraps are samples from the validation set with the same size as the validation set but acquired with sampling with replacement. Therefore the same model can be tested on multiple validation sets and the results can be compared and the SD calculated. 

#define function for creating bootstraps
def create_bootstrap(x_sensor,x_static,y1, y2):
    #initialise empty list for bootstraps
    bootstrap_x_sensor = []
    bootstrap_x_static = []
    bootstrap_y1 = []
    bootstrap_y2 = []
    
    #required length of bootstrap 
    len_val = x_val.shape[0]
    
    #get random observation 
    for i in range(len_val):
        # get random index
        random_idx = np.random.choice(range(len_val), 1)
		# get random observation
        random_x_sensor = x_sensor[random_idx]
        random_x_static = x_static[random_idx]
        random_y1 = y1[random_idx]
        random_y2 = y2[random_idx]
        
		# add random observation to bootstrap
        bootstrap_x_sensor.append(random_x_sensor)
        bootstrap_x_static.append(random_x_static)
        bootstrap_y1.append(random_y1)
        bootstrap_y2.append(random_y2)
        
	# convert to numpy
    bootstrap_x_sensor = np.asarray(bootstrap_x_sensor) 
    bootstrap_x_static = np.asarray(bootstrap_x_static)
    bootstrap_y1 = np.asarray(bootstrap_y1)
    bootstrap_y2 = np.asarray(bootstrap_y2)

	# return	
    return(bootstrap_x_sensor, bootstrap_x_static, bootstrap_y1, bootstrap_y2)


# COMMAND ----------

# define function to create bootstraps
def create_bootstraps(x_sensor,x_static,y1, y2, number_bootstraps):
	
# initialize bootstrap containers
    bootstrap_container_x_sensor = []
    bootstrap_container_x_static = []
    bootstrap_container_y1 = []
    bootstrap_container_y2 = []
		
	# create n bootstrap
    for i in range(number_bootstraps):
		# get bootstrap
        bootstrap_x_sensor, bootstrap_x_static, bootstrap_y1, bootstrap_y2 = create_bootstrap(x_sensor,x_static,y1, y2)
		# add to container
        bootstrap_container_x_sensor.append(bootstrap_x_sensor)
        bootstrap_container_x_static.append(bootstrap_x_static)
        bootstrap_container_y1.append(bootstrap_y1)
        bootstrap_container_y2.append(bootstrap_y2)

	# return
    return(bootstrap_container_x_sensor, bootstrap_container_x_static, bootstrap_container_y1, bootstrap_container_y2)

# COMMAND ----------

#define function to evaluate model 
def evaluate_model(model, bootstrap_container_x_sensor, bootstrap_container_x_static, bootstrap_container_y1, bootstrap_container_y2):

	# initialize evaluation container
    performance_container = []

	# loop through bootstraps
    for i in range(len(bootstrap_container_x_sensor)):

		# get X
        bootstrap_x_sensor = bootstrap_container_x_sensor[i]
        bootstrap_x_static = bootstrap_container_x_static[i]
		# get y
        bootstrap_y1 = bootstrap_container_y1[i]
        bootstrap_y2 = bootstrap_container_y2[i]
        #reshape x from 3d to 2d for machine learning and from 4d to 3d for deep learning 
        bootstrap_x_sensor = bootstrap_x_sensor.reshape((bootstrap_x_sensor.shape[0], (bootstrap_x_sensor.shape[1]*bootstrap_x_sensor.shape[2]), bootstrap_x_sensor.shape[3]))
        bootstrap_x_static = bootstrap_x_static.reshape(bootstrap_x_static.shape[0], (bootstrap_x_static.shape[1]*bootstrap_x_static.shape[2]))
        #for XgBoost convert to d-matrix, comment out when not using XgBoost!
#         bootstrap_x_sensor =  xgb.DMatrix(bootstrap_x_sensor)
		# get predictions #depending on model to evaluate, for some models static features need to be added 
        preds = model.predict([bootstrap_x_sensor, bootstrap_x_static])
        #convert to 0 or 1 value
        preds = np.where(np.squeeze(preds) < 0.5, 0, 1)
		# get metric # first choose witch y set to test 1 = clustered, 2 = cut-off
        auc = roc_auc_score(preds, bootstrap_y2)
        
		# add to container
        performance_container.append(auc)
      
	# return
    return(performance_container)


# COMMAND ----------

# MAGIC %md #Classic machine learning models

# COMMAND ----------

#flatten numpyarray naar (365,418) 
#necessairy beceause machine learning models can not funtion on 3d data 
x_Train = x_train.reshape(x_train.shape[0], (x_train.shape[1]*x_train.shape[2]))
x_Train_up = x_train_up.reshape(x_train_up.shape[0], (x_train_up.shape[1]*x_train_up.shape[2])) #shape (534,418)
x_Test = x_test.reshape(x_test.shape[0], (x_test.shape[1]*x_test.shape[2]))
x_Val = x_val.reshape(x_val.shape[0], (x_val.shape[1]*x_val.shape[2]))
# x_ValANDTrain = x_valANDtrain.reshape(x_valANDtrain.shape[0], (x_valANDtrain.shape[1]*x_valANDtrain.shape[2]))
# x_TrainANDVal_up = x_trainANDval_up.reshape(x_trainANDval_up.shape[0], (x_trainANDval_up.shape[1]*x_trainANDval_up.shape[2]))

# COMMAND ----------

 #create 50 bootstraps for machine learning (different due to flattened x-values)
Bootstrap_x_sensor_flat, Bootstrap_x_static, Bootstrap_y1, Bootstrap_y2  = create_bootstraps(x_Val, x_Static_val, y_val, y_val2, 50)


# COMMAND ----------

# MAGIC %md ##Logistic Regression 

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
#add weights for class inbalance
class_weights = {0:1, 1:3}
#define model
logisticmodel = LogisticRegression(random_state=0, solver = 'liblinear', max_iter = 400, class_weight = class_weights).fit(x_Train, y_train)

# COMMAND ----------

#Bootstrapping, evaluate model
Bootstrap_performance = evaluate_model(logisticmodel, Bootstrap_x_sensor_flat, Bootstrap_x_static, Bootstrap_y1, Bootstrap_y2 )

# COMMAND ----------

#mean and variance performance bootstrapping
Mean_performance = np.mean(Bootstrap_performance)
SD_performance = np.std(Bootstrap_performance)
# df = pd.DataFrame(Bootstrap_performance, columns = ['AUC'])
# display(df)
Mean_performance, SD_performance

# COMMAND ----------

#predict values test set
pred = logisticmodel.predict(x_Test)

# COMMAND ----------

#visualize performance
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print(classification_report(y_test, pred))

# COMMAND ----------

#calculated performance metrics 
from sklearn.metrics import average_precision_score, auc
sensitiviteit = cm[1,1]/(cm[1,0]+cm[1,1])
specificiteit = cm[0,0]/(cm[0,0]+cm[0,1])
ppv = cm[1,1]/(cm[1,1]+cm[0,1])
auc = roc_auc_score(y_test, pred)
MA = logisticmodel.score(x_Test, y_test) #mean accuracy #het percentage wat het model goed geraden heeft 
average_precision = average_precision_score(y_test, pred)
print('sensitiviteit =', sensitiviteit, 'specificiteit =', specificiteit, 'ppv =', ppv, 'AUC =', auc, 'Mean accuracy =', MA, 'average_precision =', average_precision)

# COMMAND ----------

# MAGIC %md ##XgBoost

# COMMAND ----------

#needed for ray tuner
%load_ext tensorboard

# COMMAND ----------

import xgboost as xgb
from ray import tune
#code for hyperparameter tuning XgBoost model using the Ray Tuner 
#while using databricks you need to manualy add ray to the cluster in the compute window, while adding select from pypi
#define model
def tree(config):
    dtrain = xgb.DMatrix(x_Train, label=y_train)
    dtest = xgb.DMatrix(x_Test)
    dval =  xgb.DMatrix(x_Val, label=y_val)
    results={}
    bst = xgb.train(
        config,
        dtrain,
        evals = [(dval, 'validation')],
        evals_result=results,
        verbose_eval=False,
       ) 
    accuracy = 1. - results["validation"]["error"][-1]
    auc = results["validation"]["auc"][-1]
    tune.report(mean_accuracy=accuracy, mean_auc = auc, done=True)
                
   
#define hyperparameters you would like to tune 
if __name__ == "__main__":
    config = { 
        "objective": "binary:logistic",
        "seed": 32,
        "eval_metric": ["auc", "error"],
        'max_depth': tune.randint(3,10),
        'min_chil_weight' : tune.choice([1,2,3,4]),
        'gamma' : tune.choice([0, 1]),
        'eta' : tune.loguniform(1e-4, 1e-1),  
        'scale_pos_weight' : tune.choice([((len(y_train)-sum(y_train))/sum(y_train)),1]),
    }
    

    analysis = tune.run(
        tree,
        # You can add "gpu": 0.1 to allocate GPUs
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=50)


   

# COMMAND ----------

#model architecture XgBoost
import xgboost as xgb
# binary target
#convert to special XgBoost matrix
dtrain = xgb.DMatrix(x_Train, label=y_train2)
dtest = xgb.DMatrix(x_Test)
dval =  xgb.DMatrix(x_Val, label=y_val2)
# dvaltrain = xgb.DMatrix(x_ValANDTrain, label = y_valANDtrain2)
#set parameters
param = {'max_depth': 6, "min_child_weight": 1  , 'eta':0.00257046, 'gamma': 1, 'objective': 'binary:logistic'}
param['scale_pos_weight'] = (len(y_train)-sum(y_train))/sum(y_train) #total number of positive samples devided by the total number of negative sampels, method to deal with class imbalance 
param['eval_metric'] = 'auc' 
param['seed']=32
#evaluation data for early stopping
evallist = [(dtrain, 'train'), (dval, 'validation')]
# evallist = [(dtest, 'test')] #earlystopping nu maar met de testset? 

# COMMAND ----------

num_round = 100 #define/use when not using early stopping. Early stopping is the better option most of the time since it prevents overfitting on the train set.  
boost = xgb.train(param, dtrain, early_stopping_rounds = 200, evals = evallist)

# COMMAND ----------

#bootstrapping
#create 50 bootstraps for machine learning (different due to flattened x-values)
Bootstrap_x_sensor_flat, Bootstrap_x_static, Bootstrap_y1, Bootstrap_y2  = create_bootstraps(x_Val, x_Static_val, y_val, y_val2, 50)
Bootstrap_performance = evaluate_model(boost, Bootstrap_x_sensor_flat, Bootstrap_x_static, Bootstrap_y1, Bootstrap_y2)

# COMMAND ----------

#results bootstrapping, mean auc and std 
Mean_performance = np.mean(Bootstrap_performance)
SD_performance = np.std(Bootstrap_performance)
# df = pd.DataFrame(Bootstrap_performance, columns = ['AUC'])
# display(df)
Mean_performance, SD_performance

# COMMAND ----------

#predict y-values of the testset
ypred = boost.predict(dtest)

# COMMAND ----------

#convert to 0 or 1 value
pred_test_clases = np.where(np.squeeze(ypred) < 0.5, 0, 1)

# COMMAND ----------

#compare real y-values with predicted values using confusion matrix and classification report 
cmboost = confusion_matrix(y_test2, pred_test_clases)
dispboost = ConfusionMatrixDisplay(confusion_matrix=cmboost)
dispboost.plot()
print(classification_report(y_test2, pred_test_clases))

# COMMAND ----------

#other evaluation metrics
sensitiviteit = cmboost[1,1]/(cmboost[1,0]+cmboost[1,1])
specificiteit = cmboost[0,0]/(cmboost[0,0]+cmboost[0,1])
ppv = cmboost[1,1]/(cmboost[1,1]+cmboost[0,1])
auc = roc_auc_score(y_test2, pred_test_clases)
average_precision = average_precision_score(y_test2, pred_test_clases)
print('sensitiviteit =', sensitiviteit, 'specificiteit =', specificiteit, 'ppv =', ppv, 'AUC =', auc, 'average_precision = ', average_precision)

# COMMAND ----------

#make a precision recall plot 
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, plot_precision_recall_curve
from matplotlib import pyplot as plt
#calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test2, pred_test_clases)

#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()

# COMMAND ----------

# MAGIC %md #Neural network model 

# COMMAND ----------

#packages needed for model building, while using databricks pay attention to the fact that you need a machine learning cluster instead of the deafault cluster
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout, concatenate, BatchNormalization
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# import keras_tuner as kt #add manualy to the cluster, can be used for hyperparmameter tuning sequential model, does not function with a multi input model

# COMMAND ----------

# MAGIC %md ## Sequential model

# COMMAND ----------

class_weights = {0:1, 1:1} #used to deal with classinbalance 

# COMMAND ----------

#definition of a sequential model, used for hyperparameter tuning using the keras tuner. Eventually I did not use this method for hyperparameter tuning since it is incompatible with a multi- input model.  
def DL_Model(hp):
    model = Sequential()
    model.add(LSTM(
        units = hp.Int("units", min_value = 32, max_value=512, step = 32),
        activation = "relu",
        input_shape=(x_train.shape[1], x_train.shape[2]),
            return_sequences=hp.Boolean('extra_laag')
    ))
    if hp.Boolean('extra_laag'):
        model.add(LSTM(64, activation = 'relu', return_sequences=False))
    
        
    model.add(Dropout(rate = hp.Float('dropout', min_value=0.1, max_value = 0.8, step = 0.1)))#hoeveel info moet worden vergeten uit de vorige cel 
    model.add(Dense(1, activation = "sigmoid")) #Hoeveel waardes je wil voorspellen, sigmoid zorgt voor een getal tussen o en 1  


    model.compile(optimizer= keras.optimizers.Adam(learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")), loss='binary_crossentropy', metrics = ['AUC', 'Precision', 'Recall']) 
    return model 

# COMMAND ----------

DL_Model(kt.HyperParameters()) #shows all possible hyperparameters 

# COMMAND ----------

#tuner element of the kerastuner
tuner = kt.RandomSearch(
    hypermodel=DL_Model,
    objective="val_loss",
    max_trials= 50,
    executions_per_trial=2,
    overwrite=True,
    directory="dbfs:/FileStore/shared_uploads/m.b.vanleerdam@students.uu.nl/data",
    project_name="modeltuning2",
)

# COMMAND ----------

tuner.search_space_summary()

# COMMAND ----------

#define earlystopping callback
my_callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights = True,
    )


# COMMAND ----------

tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks = [my_callbacks],  use_multiprocessing = True)

# COMMAND ----------

models = tuner.get_best_models(num_models=2)
best_model = models[0]
tuner.results_summary()

# COMMAND ----------

#basic sequential model 
#define model
model = Sequential()
model.add(LSTM(30, activation = 'relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False)) #left relu activation out since tanh en sigmoid already cause non linearity
model.add(Dropout(0.4)) 
model.add(Dense(1, activation = "sigmoid")) #Sigmoid activation causes binary result 
#compile model
model.compile(optimizer = 'Adam', loss='binary_crossentropy', metrics = ['AUC', 'Precision', 'Recall']) 
model.summary()

# COMMAND ----------

# fit model, added earlystopping and class weights
history = model.fit(x_train_up, y_train2_up, epochs=100, batch_size = 22, validation_data=(x_val, y_val2), class_weight = {0:1, 1:1}, callbacks = [my_callbacks]) 

#plot loss and val_loss for every epoch 
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()


# COMMAND ----------

#Bootstrapping 
#create 50 bootstraps for deep learning 
Bootstrap_x_sensor, Bootstrap_x_static, Bootstrap_y1, Bootstrap_y2  = create_bootstraps(x_val, x_Static_val, y_val, y_val2, 50) 
Bootstrap_performance = evaluate_model(model, Bootstrap_x_sensor, Bootstrap_x_static, Bootstrap_y1, Bootstrap_y2)

# COMMAND ----------

#results bootstrapping, mean auc and std 
Mean_performance = np.mean(Bootstrap_performance)
SD_performance = np.std(Bootstrap_performance)
df = pd.DataFrame(Bootstrap_performance, columns = ['AUC'])
display(df)
Mean_performance, SD_performance, my_callbacks.stopped_epoch

# COMMAND ----------

#predict values test set (I eventually did not use this since the functional model performed better on the validation set )
pred = model.predict(x_test)
pred_clases = np.where(np.squeeze(pred) < 0.5, 0, 1) 

# COMMAND ----------

#visualize model performance
CM = confusion_matrix(y_test2, pred_clases)
disp = ConfusionMatrixDisplay(confusion_matrix=CM)
disp.plot()
print(classification_report(y_test2, pred_clases))

# COMMAND ----------

#evaluation metrics
sensitiviteit = CM[1,1]/(CM[1,0]+CM[1,1])
specificiteit = CM[0,0]/(CM[0,0]+CM[0,1])
ppv = CM[1,1]/(CM[1,1]+CM[0,1])
auc = roc_auc_score(y_test2, pred_clases)
average_precision = average_precision_score(y_test2, pred_clases)
print('sensitiviteit =', sensitiviteit, 'specificiteit =', specificiteit, 'ppv =', ppv, 'AUC =', auc, 'average_precision = ', average_precision)

# COMMAND ----------

#precission recall plot
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, plot_precision_recall_curve
from matplotlib import pyplot as plt
#calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, pred_clases)

#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()

# COMMAND ----------

# MAGIC %md ### Hyperparameter tuning sequential model

# COMMAND ----------

#define class weight options
cw1 = {0:1, 1:1}  
cw2 = {0:1, 1:2}
cw3 = {0:1, 1:3}
cw4 = {0:1, 1:4}

# COMMAND ----------

#def parameters to tune
p_dropout = [0, 0.1, 0.2, 0.3, 0.4]
p_units = [10, 20, 30, 40, 50, 60, 70, 80,90,100]
p_BN = [1,0]
p_BS = [12, 22, 32, 42]
p_cw = [cw1, cw2, cw3, cw4]

# COMMAND ----------


# A functional model was used for hyperparameter tuning, works at the same way as a sequential model but is compatinble with multi-input and therefore needed when adding static features.  
import random
#define inputs, in this case single input model
sensor_input = keras.Input(shape=(22,19))
#create empty list
Results = []
#run multiple models but with a different random configuration of parameters each time
for i in range(2):
    units = random.choice(p_units)
    BN = random.choice(p_BN)
    dropout = random.choice(p_dropout)
    BS = random.choice(p_BS)
    Class_weights = random.choice(p_cw)
#define LSTM:

    LSTM_output = LSTM(units)(sensor_input) #relu toevoegen? 

    if BN == 1:
        BN_output = BatchNormalization()(LSTM_output)
    else: 
        BN_output = LSTM_output

#dropout layer
    Dropout_output = Dropout(dropout)(BN_output)

#define final layer
    prediction =Dense(1, activation = 'sigmoid')(Dropout_output)

#define model
    Model = keras.Model(
    inputs = sensor_input,
    outputs =prediction)
#compile model
    Model.compile(optimizer = 'Adam', loss='binary_crossentropy',  metrics = ['AUC', 'Precision', 'Recall']) 
#fit model
    Model.fit(x_train_up, y_train2_up, batch_size = BS, epochs=100, class_weight = Class_weights, validation_data=(x_val, y_val2), callbacks = [my_callbacks],  use_multiprocessing = True )
#predict values validation set 
    pred = Model.predict(x_val)
    pred_clases = np.where(np.squeeze(pred) < 0.5, 0, 1) 
#add predictions to list and the associated hyperparameters    
    Results.append({
        'f1_score': f1_score(y_val2, pred_clases),
        'AUC': roc_auc_score(y_val2, pred_clases), 
        'Average Precision': average_precision_score(y_val2, pred_clases), 
        'p_units': units,
        'p_dropout': dropout,
        'p_BN' : BN,
        'p_BS' : BS,
        'p_cw' : Class_weights
        })
#convert to DF
Results = pd.DataFrame(Results)
#sort modelresults
Results = Results.sort_values(by='AUC', ascending=False)
display(Results)

# COMMAND ----------

# MAGIC %md ##Functional model 

# COMMAND ----------

#Multi-input functional model with both sequential as static input
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
#define inputs
sensor_input = keras.Input(shape=(22,19))
static_input = keras.Input(shape=(10))

#define LSTM 
LSTM_output = LSTM(30)(sensor_input)
#add batch normalisation 
BN_output = BatchNormalization()(LSTM_output)
#combine with static features
all_features = concatenate([BN_output, static_input])

#define MLP-layer  
MLP_output = Dense(80, activation = 'relu')(all_features)

#add dropout 
Dropout_output = Dropout(rate = 0.0)(MLP_output)

#define final layer 
prediction = Dense(1, activation = 'sigmoid')(Dropout_output)

#define model
Model = keras.Model(
    inputs = [sensor_input, static_input],
    outputs = prediction)
#compile model
Model.compile(optimizer = 'Adam', loss='binary_crossentropy',  metrics = ['AUC', 'Precision', 'Recall'])
Model.summary()

# COMMAND ----------

#fit model, add class_weights and earlystopping as callback (my_callbacks has been previously defined)
history = Model.fit([x_train_up, x_Static_up], y_train2_up, batch_size = 22, epochs=100, class_weight = {0:1, 1:3}, validation_data=([x_val, x_Static_val], y_val2), callbacks = [my_callbacks], use_multiprocessing = True )
#plot loss and val_loss curve to keep track of model functioning
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# COMMAND ----------

#Bootstrapping 
#create 50 bootstraps for deep learning
Bootstrap_x_sensor, Bootstrap_x_static, Bootstrap_y1, Bootstrap_y2  = create_bootstraps(x_val, x_Static_val, y_val, y_val2, 50) 
Bootstrap_performance = evaluate_model(Model, Bootstrap_x_sensor, Bootstrap_x_static, Bootstrap_y1, Bootstrap_y2)
Bootstrap_performance

# COMMAND ----------

#results bootstrapping, mean auc and std 
Mean_performance = np.mean(Bootstrap_performance)
SD_performance = np.std(Bootstrap_performance)
df = pd.DataFrame(Bootstrap_performance, columns = ['AUC'])
display(df)
Mean_performance, SD_performance, my_callbacks.stopped_epoch

# COMMAND ----------

#testing final model on the test set 
pred = Model.predict([x_test, x_Static_test])
pred_clases = np.where(np.squeeze(pred) < 0.5, 0, 1) 
CM = confusion_matrix(y_test2, pred_clases)
disp = ConfusionMatrixDisplay(confusion_matrix=CM)
disp.plot()
print(classification_report(y_test2, pred_clases))

# COMMAND ----------

#evaluation metrics 
sensitiviteit = CM[1,1]/(CM[1,0]+CM[1,1])
specificiteit = CM[0,0]/(CM[0,0]+CM[0,1])
ppv = CM[1,1]/(CM[1,1]+CM[0,1])
auc = roc_auc_score(y_test2, pred_clases) 

average_precision = average_precision_score(y_test2, pred_clases)
print('sensitiviteit =', sensitiviteit, 'specificiteit =', specificiteit, 'ppv =', ppv, 'AUC =', auc, 'average_precision = ', average_precision)

# COMMAND ----------

# MAGIC %md ###hyperparameter tuning functional model 

# COMMAND ----------

#define possible class-weights
cw1 = {0:1, 1:1} 
cw2 = {0:1, 1:2}
cw3 = {0:1, 1:3}

# COMMAND ----------

#define possible hyperparameters
p_dropout = [0, 0.1, 0.2, 0.3, 0.4]
p_units = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
p_units2 = [10, 20, 30, 40, 50, 60, 70, 80]
p_BN = [0,1]
p_BS = [12, 22, 32, 42]
p_cw = [cw1, cw2, cw3]

# COMMAND ----------

#hyperparameter tuning using random search for multi-input functional model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
import random
#define inputs
sensor_input = keras.Input(shape=(22,19))
static_input = keras.Input(shape=(10))
#create empty list for modelresults
results = []
#define number of possible combination to try
for i in range(100):
    #choose random parameters
    units = random.choice(p_units)
    unitsMLP = random.choice(p_units2)
    BN = random.choice(p_BN)
    dropout = random.choice(p_dropout)
    BS = random.choice(p_BS)
    Class_weights = random.choice(p_cw)

    #define LSTM:

    LSTM_output = LSTM(units)(sensor_input)
    #add batchnormalisation depending on hyperparameter choice
    if BN == 1:
        BN_output = BatchNormalization()(LSTM_output)
    else: 
        BN_output = LSTM_output
    #combine with static features
    all_features = concatenate([BN_output, static_input])

    #define MLP layer 
    MLP_output = Dense(unitsMLP, activation = 'relu')(all_features)

    #add dropout
    Dropout_output = Dropout(dropout)(MLP_output)

    #define final layer
    prediction =Dense(1, activation = 'sigmoid')(Dropout_output)

    #define model
    Model = keras.Model(
    inputs = [sensor_input, static_input],
    outputs =prediction)
    #compile model
    Model.compile(optimizer = 'Adam', loss='binary_crossentropy',  metrics = ['AUC', 'Precision', 'Recall'])
    #fit model, add earlystopping and validation data 
    Model.fit([x_train_up, x_Static_up], y_train2_up, batch_size = BS, epochs=100, class_weight = Class_weights, validation_data=([x_val,x_Static_val], y_val2), callbacks = [my_callbacks],  use_multiprocessing = True )
    #predict y-values validation set
    pred = Model.predict([x_val,x_Static_val])
    pred_clases = np.where(np.squeeze(pred) < 0.5, 0, 1) 
    #append model results to list and the hyperparameters used to make the model 
    results.append({
        'f1_score': f1_score(y_val2, pred_clases),
        'AUC': roc_auc_score(y_val2, pred_clases), 
        'Average Precision': average_precision_score(y_val2, pred_clases), 
        'p_units': units,
        'p_units2': unitsMLP,
        'p_dropout': dropout,
        'p_BN' : BN,
        'p_BS' : BS,
        'p_cw' : Class_weights               
        })
#convert list to dataframe
results = pd.DataFrame(results)
#sort by AUC
results = results.sort_values(by='AUC', ascending=False)
display(results)
