# Databricks notebook source
# MAGIC %md
# MAGIC # Upsampling of the train set 

# COMMAND ----------

Data_set_from_r = spark \
  .read \
  .option('inferschema', 'true') \
  .option("header","true") \
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/metcluster")

# COMMAND ----------

# MAGIC %md #Adding missing rows 

# COMMAND ----------

panda_data_set = Data_set_from_r.toPandas()
#convert pyspark dataframe to panda dataframe

# COMMAND ----------

#Only keep scores for the end of dry period, because scores from different time points are incomparable 
panda_data_set['FirstLocomotionScore'].mask(panda_data_set['FirstLocomotionType'] != 'LocomotionScoreEndDryPeriod', 0, inplace=True)
panda_data_set['FirstBCSScore'].mask(panda_data_set['FirstBCSType'] != 'BCSEndDryPeriod', 0, inplace=True)

# COMMAND ----------

import pandas as pd
#Only keep scores for the end of dry period, because scores from different time points are incomparable 
unique_calvings = panda_data_set[['AnimalEartag', 'PaperRecordedCalvingDate']].drop_duplicates() 
Cluster_lijst = []
CalciumValue_lijst = []
season_lijst = []
calciumdays_lijst = []
parity_lijst = []

#filter through dataset 
for index, (AnimalEartag, PaperRecordedCalvingDate) in unique_calvings.iterrows():
  filter1 = panda_data_set['AnimalEartag'] == AnimalEartag
  filter2 = PaperRecordedCalvingDate == panda_data_set['PaperRecordedCalvingDate']
  gefilterde_set = panda_data_set[filter1 & filter2]
  season = gefilterde_set['CalvingSeason'].iloc[-1]
  Calcium_cluster = gefilterde_set['Calciumcluster'].iloc[-1]
  Calcium_value = gefilterde_set['CalciumValue'].iloc[-1]
  Calcium_days = gefilterde_set['CalciumDaysInMilk'].iloc[-1]
  parity = gefilterde_set['Parity'].iloc[-1]
  Cluster_lijst.append((Calcium_cluster)) 
  season_lijst.append((season))
  calciumdays_lijst.append(( Calcium_days))
  CalciumValue_lijst.append((Calcium_value))
  parity_lijst.append((parity))
  
#add to list and dataframe  
Calciumcluster_df = pd.DataFrame (Cluster_lijst, columns = ['Calciumcluster'])    
Calciumvalue_df =  pd.DataFrame (CalciumValue_lijst, columns = ['CalciumValue'])  
calcium_df = pd.concat([Calciumcluster_df,Calciumvalue_df], axis =1)
calcium_df['CalvingSeason'] = season_lijst
calcium_df['CalciumDaysInMilk'] = calciumdays_lijst
calcium_df['Parity'] = parity_lijst

# COMMAND ----------

#Some cows had two rows for the same day, solved by group-by
data_set_bijna_compleet = panda_data_set.groupby(['AnimalEartag','PaperRecordedCalvingDate','TransitionDaysInMilk'], as_index = False) \
.sum(["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"])
#groupby caused columns to go missing, re-added by a left merge with the old dataframe 
data_set_bijna_compleet = pd.merge(data_set_bijna_compleet, panda_data_set, on = ['AnimalEartag', 'PaperRecordedCalvingDate', 'TransitionDaysInMilk', "WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay", "HerdIdentifier", "LactationNumber", "FirstLocomotionScore", 'FirstBCSScore', 'DryOffBCS', 'CalciumValue', 'KetosisValueOne', 'KetosisValueTwo', 'BCSEndDryMinusDryOff', 'DryPeriodLength', 'LocomotionDaysInMilk', 'BCSDaysInMilk', 'CalciumDaysInMilk', 'KetosisOneDaysInMilk', 'KetosisTwoDaysInMilk', "Year", "AnimalIdentifier"], how = 'left')


# COMMAND ----------

#extract BCS and Loco scores
grouped_set = panda_data_set.groupby(['AnimalEartag', 'PaperRecordedCalvingDate']).max(['FirstBCSScore', 'FirstLocomotionScore'])
BCSandLoco = grouped_set[['FirstBCSScore', 'FirstLocomotionScore']]

# COMMAND ----------

#insert rows, every cow must have 22 rows and 19 features 
import pandas as pd
import numpy as np
unique_calvings = data_set_bijna_compleet[['AnimalEartag', 'PaperRecordedCalvingDate']].drop_duplicates() #609 unique cows
unique_calvings.reset_index(drop=True, inplace=True)
calcium_df.reset_index(drop=True, inplace=True)
BCSandLoco.reset_index(drop = True, inplace = True)

columns_to_transform = ["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"]

new_column = [*range(-21,1)]

unique_calvings['TransitionDaysInMilk'] = pd.Series([new_column for x in range(len(unique_calvings.index))]).values
unique_calvings = unique_calvings.join(calcium_df, how = 'left')
unique_calvings_clus = unique_calvings.join(BCSandLoco, how = 'left')
right_df = unique_calvings_clus.groupby(['AnimalEartag', 'PaperRecordedCalvingDate','Calciumcluster', 'CalciumValue', 'CalvingSeason', 'CalciumDaysInMilk', 'Parity', 'FirstBCSScore', 'FirstLocomotionScore']).TransitionDaysInMilk.apply(lambda x: pd.DataFrame(x.values[0])).reset_index()
right_df = right_df.drop('level_9', axis=1)
right_df.columns = ['AnimalEartag', 'PaperRecordedCalvingDate', 'Calciumcluster', 'CalciumValue', 'CalvingSeason', 'CalciumDaysInMilk', 'Parity', 'FirstBCSScore', 'FirstLocomotionScore', 'TransitionDaysInMilk']
panda_set_compleet = pd.merge(right_df, data_set_bijna_compleet, on = ['AnimalEartag', 'PaperRecordedCalvingDate',  'TransitionDaysInMilk'], how = 'left')
panda_set_compleet['Calciumcluster'] = right_df['Calciumcluster']
panda_set_compleet['CalciumValue'] = right_df['CalciumValue']
panda_set_compleet['CalvingSeason'] = right_df['CalvingSeason']
panda_set_compleet['Parity'] = right_df['Parity']
panda_set_compleet['CalciumDaysInMilk'] = right_df['CalciumDaysInMilk']
panda_set_compleet['FirstLocomotionScore'] = right_df['FirstLocomotionScore']
panda_set_compleet['FirstBCSScore'] = right_df['FirstBCSScore']
panda_set_compleet = panda_set_compleet.drop(['Calciumcluster_y', 'Calciumcluster_x', 'CalciumValue_x', 'CalciumValue_y', 'CalvingSeason_x', 'CalvingSeason_y', 'CalciumDaysInMilk_x', 'CalciumDaysInMilk_y', 'Parity_x', 'Parity_y', 'FirstLocomotionScore_x', 'FirstLocomotionScore_y', 'FirstBCSScore_x', 'FirstBCSScore_y'], axis =1)
panda_set_compleet[columns_to_transform] = panda_set_compleet[columns_to_transform].replace(0, np.nan) #zodat 0 null wordt


# COMMAND ----------

#add column with cut-off value #cut-off value based on: Prevalence of subclinical hypocalcemia in dairy herds by Reinhardt
panda_set_compleet['Cut_Off'] = np.where(panda_set_compleet['CalciumValue']<= 2.0, '1', '0')

# COMMAND ----------

#plot for calciumdistribution per cluster, colour matches previous figure on clusters
from matplotlib import pyplot as plt
panda_set_compleet['Calciumcluster'] = panda_set_compleet['Calciumcluster'].astype('category')
cluster_0 = panda_set_compleet[panda_set_compleet.Calciumcluster==0].CalciumValue
cluster_1 = panda_set_compleet[panda_set_compleet.Calciumcluster==1].CalciumValue

plt.xlabel("CalciumValue")
plt.ylabel("Number Of Cows")
# plt.title("Blood calciumvalues per group provided by cluster")

plt.hist([cluster_0, cluster_1], bins =80, rwidth=0.95, color=['lightblue','orange'],label=['Healty = 0','Hypocalcaemia = 1'])
plt.legend()

plt.savefig('/tmp/CalciumDistributionCluster')
dbutils.fs.cp("file:/tmp/CalciumDistributionCluster.png","dbfs:/FileStore/shared_uploads/m.b.vanleerdam@students.uu.nl/data/FigureCalciumDistributionCluster")

# COMMAND ----------

#plot for overall distribution calcium, cut-off groups are in colour
from matplotlib import pyplot as plt
panda_set_compleet['Cut_Off'] = panda_set_compleet['Cut_Off'].astype('int')
value_0 = panda_set_compleet[panda_set_compleet.Cut_Off==0].CalciumValue
value_1 = panda_set_compleet[panda_set_compleet.Cut_Off==1].CalciumValue
 
plt.xlabel("CalciumValue")
plt.ylabel("Number Of Cows")
plt.title("Blood calciumvalues per cut_off group")
 
plt.hist([value_0, value_1], bins = 100, rwidth=0.95, color=['red','green'],label=['Healty = 0','Hypocalcaemia = 1'])
plt.legend()

# COMMAND ----------

#example of the difference of a feature between two groups 
cluster_0 = panda_set_compleet[panda_set_compleet.Calciumcluster==0].EatingTimeMinutesPerDay
cluster_1 = panda_set_compleet[panda_set_compleet.Calciumcluster==1].EatingTimeMinutesPerDay

plt.xlabel("EatingTimeMinutesPerDay")
plt.ylabel("Number Of Cows")
plt.title("EatingTimeMinutesPerDay per group")

plt.hist([cluster_0, cluster_1], rwidth=0.95, color=['blue','orange'],label=['Cluster = 0','Cluster = 1'])
plt.legend()

# COMMAND ----------

# MAGIC %md #Devide train, test and validation set 

# COMMAND ----------

#random shuffeling otherwise the cows are ordered chronological which causes bias
np.random.seed(89)
grouped = panda_set_compleet.groupby(['AnimalEartag', 'PaperRecordedCalvingDate'])
a=np.arange(grouped.ngroups)
np.random.shuffle(a)
panda_set_shuffeld = panda_set_compleet[grouped.ngroup().isin(a[:])]

# COMMAND ----------

#We cannot use random way of splitting dataset into train and test as
#the sequence of events is important for time series.
#So let us take first 60% values = 365 koeien van de 609 for train and the remaining 40% for testing and validation 122 koeien each 
# split into train and test sets
train_size = 365*22
test_size = 122*22+365*22
validation_size = 122*22
train_set, test_set, validation_set = panda_set_shuffeld.iloc[0:train_size,:], panda_set_shuffeld.iloc[train_size:test_size,:], panda_set_shuffeld.iloc[test_size:len(panda_set_shuffeld),:]
train_val = pd.concat([train_set, validation_set])

# COMMAND ----------

# DBTITLE 1,upsampling of the train set 
# Class count
unique_train = train_set[['AnimalEartag', 'PaperRecordedCalvingDate', 'Calciumcluster', 'CalciumValue']].groupby(['AnimalEartag', 'PaperRecordedCalvingDate']).first()
count_class_0, count_class_1 = unique_train['Calciumcluster'].value_counts()
#devide per class
cluster_0 = unique_train[unique_train.Calciumcluster==0]
cluster_1 = unique_train[unique_train.Calciumcluster==1]
# #upsampling
cluster_1_over = cluster_1.sample(count_class_0, replace=True)
upsampled_set = pd.concat([cluster_1_over, cluster_0], axis=0)
#random shuffeling to avoid bias due to chronological order
upsampled_set = upsampled_set.sample(frac=1).reset_index(drop=False)
samplenumber = [*range(0,534)]
upsampled_set['SampleNumber'] = samplenumber
#re-adding features 
Upsampled_train_set = pd.merge(upsampled_set, panda_set_compleet, on = ['AnimalEartag', 'PaperRecordedCalvingDate', 'Calciumcluster', 'CalciumValue'], how = 'left')


# COMMAND ----------

# DBTITLE 1,upsampling train_val
# Class count
unique_train_val = train_val[['AnimalEartag', 'PaperRecordedCalvingDate', 'Calciumcluster', 'CalciumValue']].groupby(['AnimalEartag', 'PaperRecordedCalvingDate']).first()
count_class_0, count_class_1 = unique_train_val['Calciumcluster'].value_counts()
#devide per class
cluster_0 = unique_train_val[unique_train_val.Calciumcluster==0]
cluster_1 = unique_train_val[unique_train_val.Calciumcluster==1]
# #upsampling
cluster_1_over = cluster_1.sample(count_class_0, replace=True)
upsampled_set = pd.concat([cluster_1_over, cluster_0], axis=0)
#random shuffeling zodat de dieren niet op chronologische volgorde ingedeeld worden in de sets
upsampled_set = upsampled_set.sample(frac=1).reset_index(drop=False)
samplenumber = [*range(0,730)]
upsampled_set['SampleNumber'] = samplenumber
# #terugtoevoegen features
Upsampled_train_val_set = pd.merge(upsampled_set, panda_set_compleet, on = ['AnimalEartag', 'PaperRecordedCalvingDate', 'Calciumcluster', 'CalciumValue'], how = 'left')

# COMMAND ----------

# MAGIC %md #Normalise sensor data

# COMMAND ----------

# DBTITLE 1,Scaling the features so they lie between 0 and 1 
columns_to_select = ["TransitionDaysInMilk", "WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"]

def norm_to_zero_one(df):
     return (df - df.min()) * 1.0 / (df.max() - df.min())
  
transdays = Upsampled_train_set["TransitionDaysInMilk"]

Upsampled_train_set[columns_to_select] = Upsampled_train_set[columns_to_select].groupby("TransitionDaysInMilk").apply(norm_to_zero_one)
Upsampled_train_set["TransitionDaysInMilk"] = transdays
Scaled_train_set_up = spark.createDataFrame(Upsampled_train_set)

#also used on test and validation set 
transdays_test = test_set["TransitionDaysInMilk"]
test_set[columns_to_select] = test_set[columns_to_select].groupby("TransitionDaysInMilk").apply(norm_to_zero_one)
test_set["TransitionDaysInMilk"] = transdays_test

transdays_validatie = validation_set["TransitionDaysInMilk"]
validation_set[columns_to_select] = validation_set[columns_to_select].groupby("TransitionDaysInMilk").apply(norm_to_zero_one)
validation_set["TransitionDaysInMilk"] = transdays_validatie

#when combining the train and validation set and perform upsampling on both
transdays_train_val = Upsampled_train_val_set["TransitionDaysInMilk"]

Upsampled_train_val_set[columns_to_select] = Upsampled_train_val_set[columns_to_select].groupby("TransitionDaysInMilk").apply(norm_to_zero_one)
Upsampled_train_val_set["TransitionDaysInMilk"] = transdays
Scaled_train_val_set_up = spark.createDataFrame(Upsampled_train_val_set)


# COMMAND ----------

# MAGIC %md #Missing Value Imputation

# COMMAND ----------

# MAGIC %md ##Trainset

# COMMAND ----------

# DBTITLE 1,impute mean per sensor per day 
from sklearn.base import BaseEstimator, TransformerMixin
feature_names =["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"]
class WithinGroupMeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_var):
        self.group_var = group_var
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # the copy leaves the original dataframe intact
        X_ = X.copy()
        for col in X_.columns:
            if X_[col].dtypes == 'float64':
                X_.loc[(X[col].isna()) & X_[self.group_var].notna(), col] = X_[self.group_var].map(X_.groupby(self.group_var)[col].mean())
                X_[col] = X_[col].fillna(X_[col].mean())
        return X_
 

imp = WithinGroupMeanImputer(group_var='TransitionDaysInMilk')
parity =Upsampled_train_set['Parity']
imputed_train_set_up = imp.fit(Upsampled_train_set[feature_names])

imputed_train_set_up = imp.transform(Upsampled_train_set[["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay", 'TransitionDaysInMilk']])
imputed_train_set_up = imputed_train_set_up.join(Upsampled_train_set[['AnimalEartag', 'PaperRecordedCalvingDate', 'SampleNumber', 'CalciumDaysInMilk', 'Cut_Off', 'CalvingSeason', 'Calciumcluster', 'FirstLocomotionScore', 'FirstBCSScore']])
imputed_train_set_up['Parity'] = parity
Imputed_train_set_up_spark = spark.createDataFrame(imputed_train_set_up)

# COMMAND ----------

parity_val = validation_set['Parity']
imputed_validation_set = imp.transform(validation_set[["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay", 'TransitionDaysInMilk']])
Imputed_validation_set = imputed_validation_set.join(validation_set[['AnimalEartag', 'PaperRecordedCalvingDate',  'CalciumDaysInMilk', 'Cut_Off', 'CalvingSeason', 'Calciumcluster', 'FirstLocomotionScore', 'FirstBCSScore']])
Imputed_validation_set['Parity'] = parity_val
test_parity = test_set['Parity']
imputed_test_set = imp.transform(test_set[["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay", 'TransitionDaysInMilk']])
imputed_test_set = imputed_test_set.join(test_set[['AnimalEartag', 'PaperRecordedCalvingDate', 'CalciumDaysInMilk', 'Cut_Off', 'CalvingSeason', 'Calciumcluster', 'FirstLocomotionScore', 'FirstBCSScore']])
imputed_test_set['Parity'] = test_parity
Imputed_validation_set_spark = spark.createDataFrame(Imputed_validation_set)
Imputed_test_set_spark = spark.createDataFrame(imputed_test_set)
#Upsampled_train_val_set
parity_trainval = Upsampled_train_val_set['Parity']
imputed_trainvalidation_set = imp.transform(Upsampled_train_val_set[["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay", 'TransitionDaysInMilk']])
Imputed_train_validation_set = imputed_trainvalidation_set.join(Upsampled_train_val_set[['AnimalEartag', 'PaperRecordedCalvingDate', 'SampleNumber', 'CalciumDaysInMilk', 'Cut_Off', 'CalvingSeason', 'Calciumcluster', 'FirstLocomotionScore', 'FirstBCSScore']])
Imputed_train_validation_set['Parity'] = parity_trainval
Imputed_trainValidation_up_spark = spark.createDataFrame(Imputed_train_validation_set)
