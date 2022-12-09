# Databricks notebook source
# MAGIC %md #Filling Ontology with data 

# COMMAND ----------

#In order to open source the data used in the paper about the prediction of hypocalcemia an ontology was created. The data used is preprocessed in the notebook featurepreb and includes sensorvalues, calcium measurements and the translation of the calcium values into clusters and cut-off categories. 

# COMMAND ----------

# DBTITLE 1,Loading the data 
Dataset = spark \
  .read \
  .option('inferschema', 'true') \
  .option("header","true") \
  .csv("wasbs://gpluse-cluster-2@bovianalytics.blob.core.windows.net/Projects/SenseOfSensors/CsvData/van-leerdam-et-al/Meike/adaptedDF_ontology")

# COMMAND ----------

#if needed code for converting dataset from pyspark to panda 
panda_data_set = Dataset.toPandas()

# COMMAND ----------

display(panda_data_set)

# COMMAND ----------

#Convert date to string, ontology will not function with date  
panda_data_set['CalvingDate'] = panda_data_set['CalvingDate'].astype(str)
panda_data_set['DayOfMeasurement'] = panda_data_set['DayOfMeasurement'].astype(str)
panda_data_set['DayOfCalciumMeasurement'] = panda_data_set['DayOfCalciumMeasurement'].astype(str)

# COMMAND ----------

# MAGIC %md
# MAGIC ### POPULATE ONTOLOGY SCHEMA

# COMMAND ----------

#%pip install rdflib
#%pip install openpyxl

# COMMAND ----------

from rdflib import Graph, OWL
from rdflib import Namespace
from rdflib import URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
import pandas as pd
import uuid 


# COMMAND ----------

namespace = Namespace("http://www.purl.org/sensor_cattle#")
graph = Graph()
panda_data_set =panda_data_set.fillna('')
count = 0
nrows = panda_data_set.shape[0]

for _, row in panda_data_set.iterrows():
    # populate ontology for every event
    count += 1
    if count % 10 == 0:
        print("rij %i of %i (%0.1f)" %  (count, nrows, (count/nrows * 100)), end='\r')

    
    # create Animal
    animal_instance = URIRef(namespace + str(row['AnimalEartag']))
    graph.add((animal_instance, RDF.type, namespace.DairyCow))
    graph.add((animal_instance, namespace.AnimalEartag, Literal(str(row['AnimalEartag']))))
    graph.add((animal_instance, namespace.Parity, Literal(str(row['Parity']))))
    graph.add((animal_instance, namespace.CalvingSeason, Literal(str(row['CalvingSeason']))))
    graph.add((animal_instance, namespace.CalvingDate, Literal(row['CalvingDate'],datatype=XSD.dateTimeStamp)))

    # create Herd
    if row['HerdIdentifier']:
        herd_instance = URIRef(namespace + str(row['HerdIdentifier']))
        graph.add((herd_instance, RDF.type, namespace.Herd))
        graph.add((herd_instance, namespace.HerdIdentifier, Literal(str(row['HerdIdentifier']))))

        graph.add((animal_instance, namespace.livesIn, herd_instance))


    measurement_instance1 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance1, RDF.type, namespace.Measurement))
    graph.add((measurement_instance1, namespace.AverageMinutesBetweenBouts, Literal(str(row['EatingInterBoutLengthMinutes']))))
    graph.add((measurement_instance1, namespace.measuredBy, namespace.Eating))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance1))
    graph.add((measurement_instance1, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))


    measurement_instance2 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance2, RDF.type, namespace.Measurement))
    graph.add((measurement_instance2, namespace.AverageMinutesPerBout, Literal(str(row['EatingBoutLengthMinutesPerBout']))))
    graph.add((measurement_instance2, namespace.measuredBy, namespace.Eating))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance2))
    graph.add((measurement_instance2, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance3 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance3, RDF.type, namespace.Measurement))
    graph.add((measurement_instance3, namespace.MinutesPerDay, Literal(str(row['EatingTimeMinutesPerDay']))))
    graph.add((measurement_instance3, namespace.measuredBy, namespace.Eating))
    graph.add((measurement_instance3, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance3))

    measurement_instance4 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance4, RDF.type, namespace.Measurement))
    graph.add((measurement_instance4, namespace.NumberOfBouts, Literal(str(row['EatingNumberOfBoutsPerDay']))))
    graph.add((measurement_instance4, namespace.measuredBy, namespace.Eating))
    graph.add((measurement_instance4, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance4))


    measurement_instance5 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance5, RDF.type, namespace.Measurement))
    graph.add((measurement_instance5, namespace.AverageMinutesBetweenBouts, Literal(str(row['RuminationInterBoutLengthMinutes']))))
    graph.add((measurement_instance5, namespace.measuredBy, namespace.Ruminating))
    graph.add((measurement_instance5, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance5))

    measurement_instance6 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance6, RDF.type, namespace.Measurement))
    graph.add((measurement_instance6, namespace.AverageMinutesPerBout, Literal(str(row['RuminationBoutLengthMinutesPerBout']))))
    graph.add((measurement_instance6, namespace.measuredBy, namespace.Ruminating))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance6))
    graph.add((measurement_instance6, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance7 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance7, RDF.type, namespace.Measurement))
    graph.add((measurement_instance7, namespace.MinutesPerDay, Literal(str(row['RuminationTimeMinutesPerDay']))))
    graph.add((measurement_instance7, namespace.measuredBy, namespace.Ruminating))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance7))
    graph.add((measurement_instance7, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance8 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance8, RDF.type, namespace.Measurement))
    graph.add((measurement_instance8, namespace.NumberOfBouts, Literal(str(row['RuminationNumberOfBoutsPerDay']))))
    graph.add((measurement_instance8, namespace.measuredBy, namespace.Ruminating))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance8))
    graph.add((measurement_instance8, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance9 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance9, RDF.type, namespace.Measurement))
    graph.add((measurement_instance9, namespace.NumberOfBouts, Literal(str(row['LyingBoutsPerDay']))))
    graph.add((measurement_instance9, namespace.measuredBy, namespace.Lying))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance9))
    graph.add((measurement_instance9, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance10 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance10, RDF.type, namespace.Measurement))
    graph.add((measurement_instance10, namespace.MinutesPerDay, Literal(str(row['LyingTimeMinutesPerDay']))))
    graph.add((measurement_instance10, namespace.measuredBy, namespace.Lying))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance10))
    graph.add((measurement_instance10, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance11 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance11, RDF.type, namespace.Measurement))
    graph.add((measurement_instance11, namespace.AverageMinutesPerBout, Literal(str(row['LyingBoutLengthMinutesPerDay']))))
    graph.add((measurement_instance11, namespace.measuredBy, namespace.Lying))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance11))
    graph.add((measurement_instance11, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance12 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance12, RDF.type, namespace.Measurement))
    graph.add((measurement_instance12, namespace.MinutesPerDay, Literal(str(row['InactiveTimeMinutesPerDay']))))
    graph.add((measurement_instance12, namespace.measuredBy, namespace.Inactivity))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance12))
    graph.add((measurement_instance12, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance13 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance13, RDF.type, namespace.Measurement))
    graph.add((measurement_instance13, namespace.NumberOfBouts, Literal(str(row['InactiveBoutsPerDay']))))
    graph.add((measurement_instance13, namespace.measuredBy, namespace.Inactivity))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance13))
    graph.add((measurement_instance13, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance14 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance14, RDF.type, namespace.Measurement))
    graph.add((measurement_instance14, namespace.AverageMinutesPerBout, Literal(str(row['InactiveBoutLengthMinutesPerDay']))))
    graph.add((measurement_instance14, namespace.measuredBy, namespace.Inactivity))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance14))
    graph.add((measurement_instance14, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance15 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance15, RDF.type, namespace.Measurement))
    graph.add((measurement_instance15, namespace.AverageMinutesBetweenBouts,Literal(str(row['InactiveInterboutLengthMinutesPerDay']))))
    graph.add((measurement_instance15, namespace.measuredBy, namespace.Inactivity))
    graph.add((measurement_instance15, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance15))

    measurement_instance16 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance16, RDF.type, namespace.Measurement))
    graph.add((measurement_instance16, namespace.MinutesPerDay, Literal(str(row['WalkingTimeMinutesPerDay']))))
    graph.add((measurement_instance16, namespace.measuredBy, namespace.Walking))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance16))
    graph.add((measurement_instance16, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance17 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance17, RDF.type, namespace.Measurement))
    graph.add((measurement_instance17, namespace.NumberOfSteps, Literal(str(row['LegActivityStepsPerDay']))))
    graph.add((measurement_instance17, namespace.measuredBy, namespace.Walking))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance17))
    graph.add((measurement_instance17, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance18 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance18, RDF.type, namespace.Measurement))
    graph.add((measurement_instance18, namespace.MinutesPerDay, Literal(str(row['StandingTimeMinutesPerDay']))))
    graph.add((measurement_instance18, namespace.measuredBy, namespace.Standing))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance18))
    graph.add((measurement_instance18, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance19 = URIRef(namespace + str(uuid.uuid4()))
    graph.add((measurement_instance19, RDF.type, namespace.Measurement))
    graph.add((measurement_instance19, namespace.NumberOfBouts, Literal(str(row['StandupsPerDay']))))
    graph.add((measurement_instance19, namespace.measuredBy, namespace.Standing))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance19))
    graph.add((measurement_instance19, namespace.DayOfMeasurement, Literal(str(row['DayOfMeasurement']))))

    measurement_instance20 = URIRef(namespace + 'CalciumMeasurement' + str(row['AnimalEartag']))
    graph.add((measurement_instance20, RDF.type, namespace.BloodCalcium))
    graph.add((measurement_instance20, namespace.CalciumValue, Literal(str(row['CalciumValue']))))
    graph.add((measurement_instance20, namespace.CutOff, Literal(str(row['Cut_Off']))))
    graph.add((measurement_instance20, namespace.Cluster, Literal(str(row['Calciumcluster']))))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance20))
    graph.add((measurement_instance20, namespace.DayOfCalciumMeasurement, Literal(str(row['DayOfCalciumMeasurement']))))

    measurement_instance21 = URIRef(namespace  +str(row['AnimalEartag'])+str(row['FirstBCSScore']))

    graph.add((measurement_instance21, RDF.type, namespace.BodyConditionScore))
    graph.add((measurement_instance21, namespace.ScoreBCS, Literal(str(row['FirstBCSScore']))))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance21))

    measurement_instance22 = URIRef(namespace +str(row['AnimalEartag'])+str(row['FirstLocomotionScore']))
    graph.add((measurement_instance22, RDF.type, namespace.LocomotionScore))
    graph.add((measurement_instance22, namespace.ScoreLocomotion, Literal(str(row['FirstLocomotionScore']))))
    graph.add((animal_instance, namespace.hasMeasurement, measurement_instance22))


# COMMAND ----------

# to save
# graph.serialize(destination='sensor_data.ttl', format='ttl')

# COMMAND ----------

# MAGIC %md #Queries to check the ontology 

# COMMAND ----------

query = """
PREFIX ns1: <http://www.purl.org/sensor_cattle#>
SELECT  ?cowID  ?HerdIdentifier ?DayOfMeasurement  ?sensor ?property ?value
WHERE {
    ?cows ns1:hasMeasurement ?measurement;
          ns1:AnimalEartag ?cowID;
          ns1:livesIn ?herd.
    ?herd ns1:HerdIdentifier ?HerdIdentifier.
    ?measurement  a ns1:Measurement;
        ns1:measuredBy ?sensor;
        ns1:DayOfMeasurement ?DayOfMeasurement;
        ?property ?value.
   FILTER (?property IN (ns1:MinutesPerDay,ns1:NumberOfBouts,ns1:AverageMinutesPerBout,ns1:AverageMinutesBetweenBouts,ns1:NumberOfSteps))
}
 LIMIT  5000
"""

qres = graph.query(query)

print(qres)

df = pd.DataFrame(
    data=([None if x is None else x.toPython() for x in row] for row in qres),
    columns=[str(x) for x in qres.vars],
)

df['sensor'] = df['sensor'].str.replace('http://www.purl.org/sensor_cattle#','')
df['property'] = df['property'].str.replace('http://www.purl.org/sensor_cattle#','')
display(df)

# COMMAND ----------

query2 = """
PREFIX ns1: <http://www.purl.org/sensor_cattle#>SELECT  ?cowID  ?HerdIdentifier   ?value
WHERE {
    ?cows ns1:hasMeasurement ?measurement;
          ns1:AnimalEartag ?cowID;
          ns1:livesIn ?herd.
    ?herd ns1:HerdIdentifier ?HerdIdentifier.
    ?measurement a ns1:BodyConditionScore; 
        ns1:ScoreBCS ?value.
}
LIMIT  10000
"""


qres2 = graph.query(query2)

print(qres2)

df2 = pd.DataFrame(
    data=([None if x is None else x.toPython() for x in row] for row in qres2),
    columns=[str(x) for x in qres2.vars],
)

display(df2)

# COMMAND ----------

query3 = """
PREFIX ns1: <http://www.purl.org/sensor_cattle#>SELECT  ?cowID  ?HerdIdentifier  ?DayOfCalciumMeasurement ?CalciumValue ?Cluster ?CutOff 
WHERE {
    ?cows ns1:hasMeasurement ?measurement;
          ns1:AnimalEartag ?cowID;
          ns1:livesIn ?herd.
    ?herd ns1:HerdIdentifier ?HerdIdentifier.
    ?measurement  a ns1:BloodCalcium;
    ns1:DayOfCalciumMeasurement ?DayOfCalciumMeasurement ;
    ns1:CalciumValue ?CalciumValue ;
    ns1:Cluster ?Cluster;
    ns1:CutOff ?CutOff .
}

"""


qres3 = graph.query(query3)

print(qres3)

df3= pd.DataFrame(
    data=([None if x is None else x.toPython() for x in row] for row in qres3),
    columns=[str(x) for x in qres3.vars],
)

display(df3)
