# Databricks notebook source
# MAGIC %md
# MAGIC # Feature preparation for calcium predictive models

# COMMAND ----------

# MAGIC %md
# MAGIC # Labels

# COMMAND ----------

CalvingData = spark \
  .read \
  .option("inferSchema","true") \
  .option("header","true") \
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Labels")

# COMMAND ----------

from pyspark.sql import functions as f

# COMMAND ----------

# DBTITLE 1,Filter for cows with an available bloodsample on day 0,1,2 after calving 
Calcium012 = CalvingData.filter((f.col('CalciumDaysInMilk') == 0) | (f.col('CalciumDaysInMilk') == 1) | (f.col('CalciumDaysInMilk') == 2))

# COMMAND ----------

# MAGIC %md
# MAGIC # Features

# COMMAND ----------

display(dbutils.fs.ls("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Features"))

# COMMAND ----------

SensorData = spark \
.read \
.option("header","true") \
.csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Features/") 

# COMMAND ----------

# DBTITLE 1,Select only features for the 21 days before calving and the day of calving 
dfFilteredSensorData = SensorData \
  .withColumn("TransitionDaysInMilk", SensorData.TransitionDaysInMilk.cast("integer")) \
  .filter(f.col("TransitionDaysInMilk").between(-21, 0))

# COMMAND ----------

# DBTITLE 1,drop unnecessary features
dfFilteredSensorData2 = dfFilteredSensorData \
  .filter((dfFilteredSensorData.Sensor != 'EatingTimeMinutesPer15Minutes') & 
         (dfFilteredSensorData.Sensor != 'EatingTimeMinutesPer2Hours') &
         (dfFilteredSensorData.Sensor != 'InactiveTimeMinutesPer15Minutes') &
         (dfFilteredSensorData.Sensor != 'InactiveTimeMinutesPer2Hours') &
         (dfFilteredSensorData.Sensor != 'LegActivityStepsPer15Minutes') &
         (dfFilteredSensorData.Sensor != 'LegActivityStepsPer2Hours') &
         (dfFilteredSensorData.Sensor != 'LyingBoutLengthLowerQntMinutesPerDay') &
         (dfFilteredSensorData.Sensor != 'LyingBoutLengthMaxMinutesPerDay') &
         (dfFilteredSensorData.Sensor != 'LyingBoutLengthMedMinutesPerDay') &
         (dfFilteredSensorData.Sensor != 'LyingBoutLengthMinMinutesPerDay') & 
         (dfFilteredSensorData.Sensor != 'LyingBoutLengthUpperQntMinutesPerDay') &
         (dfFilteredSensorData.Sensor != 'LyingBoutsPer15Minutes') &
         (dfFilteredSensorData.Sensor != 'LyingTimeMinutesPer15Minutes') &
         (dfFilteredSensorData.Sensor != 'LyingTimeMinutesPer2Hours') &
         (dfFilteredSensorData.Sensor != 'RuminationTimeMinutesPer15Minutes') &
         (dfFilteredSensorData.Sensor != 'RuminationTimeMinutesPer2Hours') &
         (dfFilteredSensorData.Sensor != 'StandingTimeMinutesPer15Minutes') &
         (dfFilteredSensorData.Sensor != 'StandingTimeMinutesPer2Hours') & 
         (dfFilteredSensorData.Sensor != 'WalkingTimeMinutesPer15Minutes') &
         (dfFilteredSensorData.Sensor != 'WalkingTimeMinutesPer2Hours'))

#drop other unnecessary columns 
dfFilteredSensorData2 = dfFilteredSensorData2.drop('TransitionAnimalEartag','DaysInMilk','EventDate')

# COMMAND ----------

# MAGIC %md
# MAGIC #Features and Labels combined

# COMMAND ----------

Joined_dataset = Calcium012 \
  .join(dfFilteredSensorData2, ["AnimalEartag", "HerdIdentifier", "CalvingDate", "Parity", "LactationNumber", "CalvingSeason"])

# COMMAND ----------

# DBTITLE 1,sensorvalue, sd and avg convert to integer 
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
Joined_dataset = Joined_dataset \
  .withColumn("SensorValue", \
              col("SensorValue") \
                .cast(IntegerType()))    
Joined_dataset = Joined_dataset \
  .withColumn("AvgSensorValue", \
              col("AvgSensorValue") \
                .cast(IntegerType())) 
Joined_dataset = Joined_dataset \
  .withColumn("SDSensorValue", \
              col("SDSensorValue") \
                .cast(IntegerType()))  
Joined_dataset.printSchema()
(Joined_dataset)

# COMMAND ----------

# MAGIC %md ##Histograms per feature 

# COMMAND ----------

#extract each feature
WalkingTimeMinutesPerDay = Joined_dataset.filter((f.col('Sensor') == "WalkingTimeMinutesPerDay"))
EatingBoutLengthMinutesPerBout = Joined_dataset.filter((f.col('Sensor') == "EatingBoutLengthMinutesPerBout"))
EatingInterBoutLengthMinutes = Joined_dataset.filter((f.col('Sensor') == "EatingInterBoutLengthMinutes"))
EatingNumberOfBoutsPerDay = Joined_dataset.filter((f.col('Sensor') == "EatingNumberOfBoutsPerDay"))
EatingTimeMinutesPerDay = Joined_dataset.filter((f.col('Sensor') == "EatingTimeMinutesPerDay"))
InactiveBoutLengthMinutesPerDay = Joined_dataset.filter((f.col('Sensor') == "InactiveBoutLengthMinutesPerDay"))
InactiveBoutsPerDay = Joined_dataset.filter((f.col('Sensor') == "InactiveBoutsPerDay"))
InactiveInterboutLengthMinutesPerDay = Joined_dataset.filter((f.col('Sensor') == "InactiveInterboutLengthMinutesPerDay"))
InactiveTimeMinutesPerDay = Joined_dataset.filter((f.col('Sensor') == "InactiveTimeMinutesPerDay"))
LegActivityStepsPerDay = Joined_dataset.filter((f.col('Sensor') == "LegActivityStepsPerDay"))
LyingBoutLengthMinutesPerDay = Joined_dataset.filter((f.col('Sensor') == "LyingBoutLengthMinutesPerDay"))
LyingBoutsPerDay = Joined_dataset.filter((f.col('Sensor') == "LyingBoutsPerDay"))
LyingTimeMinutesPerDay = Joined_dataset.filter((f.col('Sensor') == "LyingTimeMinutesPerDay"))
RuminationBoutLengthMinutesPerBout = Joined_dataset.filter((f.col('Sensor') == "RuminationBoutLengthMinutesPerBout"))
RuminationInterBoutLengthMinutes = Joined_dataset.filter((f.col('Sensor') == "RuminationInterBoutLengthMinutes"))
RuminationNumberOfBoutsPerDay = Joined_dataset.filter((f.col('Sensor') == "RuminationNumberOfBoutsPerDay"))
RuminationTimeMinutesPerDay = Joined_dataset.filter((f.col('Sensor') == "RuminationTimeMinutesPerDay"))
StandingTimeMinutesPerDay = Joined_dataset.filter((f.col('Sensor') == "StandingTimeMinutesPerDay"))
StandupsPerDay = Joined_dataset.filter((f.col('Sensor') == "StandupsPerDay"))

# COMMAND ----------

display(WalkingTimeMinutesPerDay)

# COMMAND ----------

display(EatingBoutLengthMinutesPerBout)

# COMMAND ----------

display(EatingInterBoutLengthMinutes)

# COMMAND ----------

display(EatingNumberOfBoutsPerDay)

# COMMAND ----------

display(EatingTimeMinutesPerDay)

# COMMAND ----------

display(InactiveBoutLengthMinutesPerDay)

# COMMAND ----------

display(InactiveBoutsPerDay)

# COMMAND ----------

display(InactiveInterboutLengthMinutesPerDay)

# COMMAND ----------

display(InactiveTimeMinutesPerDay)

# COMMAND ----------

display(LegActivityStepsPerDay)

# COMMAND ----------

display(LyingBoutLengthMinutesPerDay)

# COMMAND ----------

display(LyingBoutsPerDay)

# COMMAND ----------

display(LyingTimeMinutesPerDay)

# COMMAND ----------

display(RuminationTimeMinutesPerDay)

# COMMAND ----------

display(RuminationBoutLengthMinutesPerBout)

# COMMAND ----------

display(RuminationInterBoutLengthMinutes)

# COMMAND ----------

display(RuminationNumberOfBoutsPerDay)

# COMMAND ----------

display(StandingTimeMinutesPerDay)

# COMMAND ----------

display(StandupsPerDay)

# COMMAND ----------

# MAGIC %md ##Each feature gets its own column 

# COMMAND ----------

# DBTITLE 1,pivot table 
pivotJoined_dataset = Joined_dataset \
  .groupBy(["AnimalEartag", "HerdIdentifier", "CalvingDate", "Parity", "LactationNumber", "CalvingSeason", "FirstLocomotionScoreDate", "FirstLocomotionScore", "FirstLocomotionType", "FirstBCSDate", "FirstBCSScore", "FirstBCSType", "DryOffDate", "DryOffBCS", "CalciumDate", "CalciumValue", "KetosisDateOne", "KetosisValueOne", "KetosisDateTwo", "KetosisValueTwo", "PaperRecordedCalvingDate", "BCSEndDryMinusDryOff", "DryPeriodLength", "LocomotionDaysInMilk", "BCSDaysInMilk", "CalciumDaysInMilk", "KetosisOneDaysInMilk", "KetosisTwoDaysInMilk", "AnimalIdentifier", "Year" , "TransitionDaysInMilk"]) \
  .pivot("Sensor", ["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay" ]) \
  .sum('SensorValue')
#remove AVG and SD, not needed

# COMMAND ----------

pivotJoined_dataset\
  .repartition(1)\
  .write\
  .mode('overwrite') \
  .option("header","true")\
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/pivotJoined")

# COMMAND ----------

pivotJoined_dataset = spark \
  .read \
  .option('inferschema', 'true') \
  .option("header","true") \
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/pivotJoined")

# COMMAND ----------

#sort on animal per day 
pivotJoined_dataset = pivotJoined_dataset.sort(['AnimalEartag', 'CalvingDate', 'TransitionDaysInMilk'])

# COMMAND ----------

# DBTITLE 1,Remove extreme outlier 
#cow recieved calcium infusion just before bloodsampling 
pivotJoined_dataset = pivotJoined_dataset \
  .filter(pivotJoined_dataset.CalciumValue < 3.4)

# COMMAND ----------

#graph of calcium distribution
display(pivotJoined_dataset)

# COMMAND ----------

# DBTITLE 1,Convert features to integers
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("WalkingTimeMinutesPerDay", \
              col("WalkingTimeMinutesPerDay") \
                .cast(IntegerType()))    
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("EatingBoutLengthMinutesPerBout", \
              col("EatingBoutLengthMinutesPerBout") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("EatingInterBoutLengthMinutes", \
              col("EatingInterBoutLengthMinutes") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("EatingNumberOfBoutsPerDay", \
              col("EatingNumberOfBoutsPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("EatingTimeMinutesPerDay", \
              col("EatingTimeMinutesPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("InactiveBoutLengthMinutesPerDay", \
              col("InactiveBoutLengthMinutesPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("InactiveBoutsPerDay", \
              col("InactiveBoutsPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("InactiveInterboutLengthMinutesPerDay", \
              col("InactiveInterboutLengthMinutesPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("InactiveTimeMinutesPerDay", \
              col("InactiveTimeMinutesPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("LegActivityStepsPerDay", \
              col("LegActivityStepsPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("LyingBoutLengthMinutesPerDay", \
              col("LyingBoutLengthMinutesPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("LyingBoutsPerDay", \
              col("LyingBoutsPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("LyingTimeMinutesPerDay", \
              col("LyingTimeMinutesPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("RuminationBoutLengthMinutesPerBout", \
              col("RuminationBoutLengthMinutesPerBout") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("RuminationInterBoutLengthMinutes", \
              col("RuminationInterBoutLengthMinutes") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("RuminationNumberOfBoutsPerDay", \
              col("RuminationNumberOfBoutsPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("RuminationTimeMinutesPerDay", \
              col("RuminationTimeMinutesPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("StandingTimeMinutesPerDay", \
              col("StandingTimeMinutesPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset = pivotJoined_dataset \
  .withColumn("StandupsPerDay", \
              col("StandupsPerDay") \
                .cast(IntegerType())) 
pivotJoined_dataset.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC #Scattermatrix

# COMMAND ----------

# DBTITLE 1,Scattermatrix features and calcium
from pandas.plotting import scatter_matrix
Variables_for_scatter = ["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay", "CalciumValue"]

numeric_data = pivotJoined_dataset.select(Variables_for_scatter).toPandas()
axs = scatter_matrix(numeric_data, figsize=(23, 23));

# Rotate axis labels and remove axis ticks
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

# COMMAND ----------

# MAGIC %md #Add calciumclusters

# COMMAND ----------

#Convert pyspark to R
pivotJoined_dataset.createOrReplaceTempView("R_scaled")

# COMMAND ----------

# MAGIC %r
# MAGIC require(SparkR)
# MAGIC Data_set_r <- sql("select * from R_scaled")
# MAGIC Data_set_r <- as.data.frame(Data_set_r)
# MAGIC head(Data_set_r)

# COMMAND ----------

# MAGIC %r
# MAGIC #lineair mixed effect model, built in order to compensate for dependencies in the data > farm, parity, day of sampling 
# MAGIC install.packages("lme4")
# MAGIC install.packages("ggplot2")
# MAGIC library(ggplot2)
# MAGIC library(lme4)
# MAGIC library(dplyr)
# MAGIC AnalysisDataCluster <- Data_set_r %>% 
# MAGIC dplyr::select(
# MAGIC     CalciumValue,
# MAGIC     CalciumDaysInMilk,
# MAGIC     Parity,
# MAGIC     AnimalEartag,
# MAGIC     HerdIdentifier)
# MAGIC 
# MAGIC #define model
# MAGIC ClusterLMER<-lme4::lmer(
# MAGIC                   CalciumValue ~ CalciumDaysInMilk * Parity +  (1| HerdIdentifier), 
# MAGIC                   data = AnalysisDataCluster,
# MAGIC                   REML = FALSE
# MAGIC                   )
# MAGIC 
# MAGIC #predict calcium
# MAGIC AnalysisDataCluster$PredictCalcium <- predict(ClusterLMER, newdata = AnalysisDataCluster)
# MAGIC AnalysisDataCluster$CalciumResidual <- residuals(ClusterLMER)
# MAGIC #k-means clustering
# MAGIC AnalysisDataCluster$CalciumCluster <- as.factor(kmeans(AnalysisDataCluster[,c("CalciumValue","CalciumResidual")],centers=2)$cluster)
# MAGIC #plot calcium vs residuals with clustering 
# MAGIC ggplot(AnalysisDataCluster,
# MAGIC aes(x=CalciumValue, y=CalciumResidual, colour = CalciumCluster))  +
# MAGIC geom_point() +
# MAGIC scale_color_manual(breaks = c('1','2'),
# MAGIC                    values=c("tan1", "cadetblue3")) +
# MAGIC theme_classic()

# COMMAND ----------

# MAGIC %r
# MAGIC #when numbers of observations are limited, a binary variable is easier to predict. Therefore the clustering is used as a binary variable
# MAGIC Calciumcluster <- AnalysisDataCluster$CalciumCluster 
# MAGIC #add clusters to dataframe
# MAGIC data_set_r2 <- cbind(Data_set_r, Calciumcluster)
# MAGIC #change name of cluster 2 to 0 not at risk and 1 stays 1 at risk of hypocalceamia
# MAGIC levels(data_set_r2$Calciumcluster) <- c(1,0)
# MAGIC summary(data_set_r2$Calciumcluster)

# COMMAND ----------

# MAGIC %r
# MAGIC #convert dataframe from r back to python 
# MAGIC data_set_levels <- as.DataFrame(data_set_r2)
# MAGIC 
# MAGIC createOrReplaceTempView(data_set_levels,"r_back_to_py")

# COMMAND ----------

Data_set_from_r = sql("SELECT * FROM r_back_to_py") 

# COMMAND ----------

# MAGIC %md #save dataset 

# COMMAND ----------

Data_set_from_r\
  .repartition(1)\
  .write\
  .option("header","true")\
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/metcluster")

# COMMAND ----------

Data_set_from_r = spark \
  .read \
  .option('inferschema', 'true') \
  .option("header","true") \
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/metcluster")

# COMMAND ----------

# MAGIC %md #Add missing rows 

# COMMAND ----------

panda_data_set = Data_set_from_r.toPandas()
#convert pyspark dataframe to panda dataframe

# COMMAND ----------

#Only keep scores for the end of dry period, because scores from different time points are incomparable 
panda_data_set['FirstLocomotionScore'].mask(panda_data_set['FirstLocomotionType'] != 'LocomotionScoreEndDryPeriod', 0, inplace=True)
panda_data_set['FirstBCSScore'].mask(panda_data_set['FirstBCSType'] != 'BCSEndDryPeriod', 0, inplace=True)

# COMMAND ----------

import pandas as pd
#extract unique cows and the static features that belong to that cow and are never missing 
unique_calvings = panda_data_set[['AnimalEartag', 'PaperRecordedCalvingDate']].drop_duplicates() 
Cluster_lijst = []
CalciumValue_lijst = []
season_lijst = []
calindays_lijst = []
parity_lijst = []

#filter through dataset 
for index, (AnimalEartag, PaperRecordedCalvingDate) in unique_calvings.iterrows():
    filter1 = panda_data_set['AnimalEartag'] == AnimalEartag
    filter2 = PaperRecordedCalvingDate == panda_data_set['PaperRecordedCalvingDate']
    gefilterde_set = panda_data_set[filter1 & filter2]
    Parity = gefilterde_set['Parity'].iloc[-1]
    Calindays = gefilterde_set['CalciumDaysInMilk'].iloc[-1]
    season = gefilterde_set['CalvingSeason'].iloc[-1] 
    Calcium_cluster = gefilterde_set['Calciumcluster'].iloc[-1]
    Calcium_Value = gefilterde_set['CalciumValue'].iloc[-1]
    Cluster_lijst.append((Calcium_cluster)) 
    CalciumValue_lijst.append((Calcium_Value))
    season_lijst.append((season))
    calindays_lijst.append(Calindays)
    parity_lijst.append(Parity)
#add to list and dataframe    
Calcium_df = pd.DataFrame (Cluster_lijst, columns = ['Calciumcluster'])  
Calcium_df['CalciumValue'] = CalciumValue_lijst
Calcium_df['CalvingSeason'] = season_lijst
Calcium_df['Parity'] = parity_lijst
Calcium_df['CalciumDaysInMilk'] = calindays_lijst

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
Calcium_df.reset_index(drop=True, inplace=True)
BCSandLoco.reset_index(drop = True, inplace = True)

columns_to_transform = ["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"]

new_column = [*range(-21,1)]

unique_calvings['TransitionDaysInMilk'] = pd.Series([new_column for x in range(len(unique_calvings.index))]).values
unique_calvings = unique_calvings.join(Calcium_df, how = 'left')
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
display(panda_set_compleet)

# COMMAND ----------

#add column with cut-off value #cut-off value based on: Prevalence of subclinical hypocalcemia in dairy herds by Reinhardt
panda_set_compleet['Cut_Off'] = np.where(panda_set_compleet['CalciumValue']<= 2.0, '1', '0')


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

# COMMAND ----------

# MAGIC %md #Normalise sensor data

# COMMAND ----------

# DBTITLE 1,Scaling the features so they lie between 0 and 1 
columns_to_select = ["TransitionDaysInMilk", "WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay"]

def norm_to_zero_one(df):
     return (df - df.min()) * 1.0 / (df.max() - df.min())
  
transdays = train_set["TransitionDaysInMilk"]

train_set[columns_to_select] = train_set[columns_to_select].groupby("TransitionDaysInMilk").apply(norm_to_zero_one)
train_set["TransitionDaysInMilk"] = transdays
Scaled_train_set = spark.createDataFrame(train_set)

#also for test and validation 
transdays_test = test_set["TransitionDaysInMilk"]
test_set[columns_to_select] = test_set[columns_to_select].groupby("TransitionDaysInMilk").apply(norm_to_zero_one)
test_set["TransitionDaysInMilk"] = transdays_test

transdays_validatie = validation_set["TransitionDaysInMilk"]
validation_set[columns_to_select] = validation_set[columns_to_select].groupby("TransitionDaysInMilk").apply(norm_to_zero_one)
validation_set["TransitionDaysInMilk"] = transdays_validatie



# COMMAND ----------

# MAGIC %md #Missing Value Imputation

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

imputed_train_set = imp.fit(train_set[feature_names])

imputed_train_set = imp.transform(train_set[["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay", 'TransitionDaysInMilk']])
Imputed_train_set = imputed_train_set.join(train_set[['AnimalEartag', 'PaperRecordedCalvingDate', 'Parity', 'CalciumDaysInMilk', 'Cut_Off', 'CalvingSeason', 'Calciumcluster', 'FirstLocomotionScore', 'FirstBCSScore']])

Imputed_train_set_spark = spark.createDataFrame(Imputed_train_set)

# COMMAND ----------

imputed_validation_set = imp.transform(validation_set[["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay", 'TransitionDaysInMilk']])
Imputed_validation_set = imputed_validation_set.join(validation_set[['AnimalEartag', 'PaperRecordedCalvingDate', 'Parity', 'CalciumDaysInMilk', 'Cut_Off', 'CalvingSeason', 'Calciumcluster', 'FirstLocomotionScore', 'FirstBCSScore']])
imputed_test_set = imp.transform(test_set[["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay", 'TransitionDaysInMilk']])
imputed_test_set = imputed_test_set.join(test_set[['AnimalEartag', 'PaperRecordedCalvingDate', 'Parity', 'CalciumDaysInMilk', 'Cut_Off', 'CalvingSeason', 'Calciumcluster', 'FirstLocomotionScore', 'FirstBCSScore']])
Imputed_validation_set_spark = spark.createDataFrame(Imputed_validation_set)
Imputed_test_set_spark = spark.createDataFrame(imputed_test_set)
display(Imputed_validation_set)


# COMMAND ----------

# MAGIC %md #NA Values 

# COMMAND ----------

# DBTITLE 1,Amount of  NA values per sensor (absolute value)
from pyspark.sql.functions import col,when,count
df_Columns=["WalkingTimeMinutesPerDay", "EatingBoutLengthMinutesPerBout", "EatingInterBoutLengthMinutes", "EatingNumberOfBoutsPerDay", "EatingTimeMinutesPerDay", "InactiveBoutLengthMinutesPerDay", "InactiveBoutsPerDay", "InactiveInterboutLengthMinutesPerDay", "InactiveTimeMinutesPerDay", "LegActivityStepsPerDay", "LyingBoutLengthMinutesPerDay","LyingBoutsPerDay", "LyingTimeMinutesPerDay", "RuminationBoutLengthMinutesPerBout", "RuminationInterBoutLengthMinutes", "RuminationNumberOfBoutsPerDay","RuminationTimeMinutesPerDay", "StandingTimeMinutesPerDay", "StandupsPerDay" ]
display(Data_set_from_r.select([count(when(col(c).isNull(), c)).alias(c) for c in df_Columns]))


# COMMAND ----------

(panda_set_compleet[df_Columns].isnull().sum().sum())/(panda_set_compleet[df_Columns].count().sum())*100

# COMMAND ----------

# DBTITLE 1,Amount of  NA values per sensor (percentage)
from pyspark.sql.functions import lit
display(Data_set_from_r.select([(count(when(col(c).isNull(), c))/count(lit(1))).alias(c) for c in df_Columns]))
