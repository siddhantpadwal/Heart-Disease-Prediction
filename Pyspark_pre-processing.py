# Databricks notebook source
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType, BooleanType, DateType

# COMMAND ----------

configs = {"fs.azure.account.auth.type": "OAuth",
"fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
"fs.azure.account.oauth2.client.id": "704ad7b4-55af-4001-9a8b-481519945703",
"fs.azure.account.oauth2.client.secret": 'HCs8Q~2RoOhUwSk856PMFX3rODrcrrmHeM13obXq',
"fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/6f784768-51e6-44e0-9460-2282ff44b0ec/oauth2/token"}

# COMMAND ----------

dbutils.fs.mount(
source = "abfss://rawdata@team9dbda.dfs.core.windows.net", # contrainer@storageacc
mount_point = "/mnt/team9_01",
extra_configs = configs)

# COMMAND ----------

dbutils.fs.ls('/mnt/team9_01')

# COMMAND ----------

!pip install pyspark

# !pip install pyspark

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[*]').getOrCreate()
spark

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

df = spark.read.csv(r"/mnt/team9_01/Raw_Data_01/data_1.csv", header=True, inferSchema=True)
df.show(10)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# Calculate total number of rows
num_rows = df.count()
print(num_rows)

# COMMAND ----------


#Calculate total number of rows
num_rows = df.count()

# Create a dictionary to store column names and their respective null percentages
null_percentages_dict = {}

# Iterate over each column
for column in df.columns:
    # Count the number of null values in the column
    null_count = df.where(col(column).isNull()).count()
    # Calculate the null percentage
    null_percentage = (null_count / num_rows) * 100
    # Store column name and its null percentage in the dictionary
    null_percentages_dict[column] = null_percentage

# Print the null percentages for each column
for column, percentage in null_percentages_dict.items():
    print(f"Column '{column}' has {percentage:.2f}% null values.")



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, first
from pyspark.sql.window import Window

# COMMAND ----------

from pyspark.sql.functions import col, count, first

# Calculate the mode of the 'State' column
state_mode = df.groupBy('State').count().orderBy(col("count").desc(), col('State')).select(first('State')).collect()[0][0]

# COMMAND ----------

from pyspark.sql.functions import col, count, first

# Calculate the mode of the 'State' column
Food_Habit_mode = df.groupBy('Food_Habits').count().orderBy(col("count").desc(), col('Food_Habits')).select(first('Food_habits')).collect()[0][0]


# COMMAND ----------

print(state_mode)

# COMMAND ----------

print(Food_Habit_mode)

# COMMAND ----------

from pyspark.sql.functions import col, count, first

df = df.fillna({'State': 'Unknown', 'Food_Habits': 'Healthy Food'})


# COMMAND ----------

#Calculate total number of rows
num_rows = df.count()

# Create a dictionary to store column names and their respective null percentages
null_percentages_dict = {}

# Iterate over each column
for column in df.columns:
    # Count the number of null values in the column
    null_count = df.where(col(column).isNull()).count()
    # Calculate the null percentage
    null_percentage = (null_count / num_rows) * 100
    # Store column name and its null percentage in the dictionary
    null_percentages_dict[column] = null_percentage

# Print the null percentages for each column
for column, percentage in null_percentages_dict.items():
    print(f"Column '{column}' has {percentage:.2f}% null values.")

# COMMAND ----------

# pip install numpy

# COMMAND ----------

# pip install pandas

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Assuming 'df' is your DataFrame and 'string_columns' is your list of string columns
string_columns = ['State', 'Gender', 'BloodSugureLevel', 'ECGResults', 'PainDuringExercise', 'TypeOfBloodDisorder', 'HeartDiseaseStatus', 'PeakExercisePatterns', 'Food_Habits']

indexer = StringIndexer(inputCol=string_columns[0], outputCol="StateIndex")

indexed_columns = []

for col_name in string_columns:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "Index")
    indexed_columns.append(indexer)

indexed_data = Pipeline(stages=indexed_columns).fit(df).transform(df)
indexed_data.show()



# COMMAND ----------

df.printSchema()

# COMMAND ----------

df = df.drop("State")

# COMMAND ----------

df.show()

# COMMAND ----------

from pyspark.sql.functions import when, col

# Create a dictionary for each column where keys are original categories and values are encoded values
reverse_fasting_blood_sugar_mapping = {'greater than 120mg/dl': 1, 'less than 120mg/dl': 0}
reverse_resting_ecg_mapping = {'normal': 0, 'having ST-T wave abnormality': 1, 'left ventricular hypertrophy': 2}
reverse_exercise_induced_angina_mapping = {'yes': 1, 'no': 0}
reverse_peak_exercise_st_segment_mapping = {'upsloping': 1, 'flat': 2, 'downsloping': 3}
reverse_thal_mapping = {'normal': 3, 'fixed defect': 6, 'reversible defect': 7}

# Use the when function to replace original categories with encoded values
df = df.withColumn('BloodSugureLevel', when(col('BloodSugureLevel') == 'greater than 120mg/dl', 1).otherwise(0))
df = df.withColumn('ECGResults', when(col('ECGResults') == 'normal', 0).when(col('ECGResults') == 'having ST-T wave abnormality', 1).otherwise(2))
df = df.withColumn('PainDuringExercise', when(col('PainDuringExercise') == 'yes', 1).otherwise(0))
df = df.withColumn('PeakExercisePatterns', when(col('PeakExercisePatterns') == 'upsloping', 1).when(col('PeakExercisePatterns') == 'flat', 2).otherwise(3))
df = df.withColumn('TypeOfBloodDisorder', when(col('TypeOfBloodDisorder') == 'normal', 3).when(col('TypeOfBloodDisorder') == 'fixed defect', 6).otherwise(7))

# COMMAND ----------

df.show()

# COMMAND ----------

from pyspark.sql.functions import when, col

# Create a dictionary where keys are old values and values are new values
reverse_gender_mapping = {'male': 1.0, 'female': 0.0}

# Use the when function to replace old values with new values
df = df.withColumn('Gender', when(col('Gender') == 'male', 1.0).otherwise(0.0))

df.show(5)

# COMMAND ----------

from pyspark.sql.functions import when, col

# Create a dictionary where keys are old values and values are new values
reverse_food_habits_mapping = {'Junk Food': 1, 'Healthy Food': 0}

# Use the when function to replace old values with new values
df = df.withColumn('Food_Habits', when(col('Food_Habits') == 'Junk Food', 1).otherwise(0))

df.show(5)

# COMMAND ----------

from pyspark.sql.functions import when, col

# Use the when function to replace 'absent' with 0 and 'present' with 1 in the 'HeartDiseaseStatus' column
df = df.withColumn('HeartDiseaseStatus', when(col('HeartDiseaseStatus') == 'present', 1).otherwise(0))

df.show(5)

# COMMAND ----------

df.count(), len(df.columns)

# COMMAND ----------

df = df.dropDuplicates()
df.count(), len(df.columns)

# COMMAND ----------

dbutils.fs.mount(
source = "abfss://loadfile@team9dbda.dfs.core.windows.net", # contrainer@storageacc
mount_point = "/mnt/team9_02",
extra_configs = configs)

# COMMAND ----------

df.coalesce(1).write.mode("overwrite").option("header", "true").csv("/mnt/team9_02/Load_File_02")

# COMMAND ----------

dbutils.fs.mount(
source = "abfss://cleaneddata@team9dbda.dfs.core.windows.net", # contrainer@storageacc
mount_point = "/mnt/team9_03",
extra_configs = configs)

# COMMAND ----------

filenames = dbutils.fs.ls('/mnt/team9_02/Load_File_02')
name = ''

for filename in filenames:
    if filename.name.endswith('.csv'):
        name = filename.name

dbutils.fs.cp('/mnt/team9_02/Load_File_02/'+name,'/mnt/team9_03/Cleaned_Data_02/heart_disease_cleaned_02.csv')

# COMMAND ----------

df.printSchema()

# COMMAND ----------


