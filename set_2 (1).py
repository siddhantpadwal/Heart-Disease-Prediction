 Databricks notebook source
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType, BooleanType, DateType

# COMMAND ----------

# MAGIC %fs
# MAGIC ls "/mnt/team9_3"

# COMMAND ----------

df = spark.read.format("csv").option("header","true").load("/mnt/team9_3/Cleaned_Data_02/heart_disease_cleaned_02.csv")
df.show()

# COMMAND ----------

# Print schema command
df.printSchema()

# COMMAND ----------

#  |-- Age: integer (nullable = true)
#  |-- Gender: double (nullable = false)
#  |-- CaseNumber: integer (nullable = true)
#  |-- chest: double (nullable = true)
#  |-- NormalBloodPressure: double (nullable = true)
#  |-- CholesterolLevel: double (nullable = true)
#  |-- BloodSugureLevel: integer (nullable = false)
#  |-- ECGResults: integer (nullable = false)
#  |-- MaxHeartRateAchieved: double (nullable = true)
#  |-- PainDuringExercise: integer (nullable = false)
#  |-- ExerciseHeartChange: double (nullable = true)
#  |-- PeakExercisePattern: double (nullable = true)
#  |-- MajorVesselsCount: double (nullable = true)
#  |-- TypeOfBloodDisorder: integer (nullable = false)
#  |-- HeartDiseaseStatus: integer (nullable = false)
#  |-- PeakExercisePatterns: integer (nullable = false)
#  |-- Food_Habits: integer (nullable = false)

# COMMAND ----------

df = df.withColumn("Age", df["Age"].cast("integer")) \
       .withColumn("Gender", df["Gender"].cast("double")) \
       .withColumn("CaseNumber", df["CaseNumber"].cast("integer")) \
       .withColumn("chest", df["chest"].cast("double")) \
       .withColumn("NormalBloodPressure", df["NormalBloodPressure"].cast("double")) \
       .withColumn("CholesterolLevel", df["CholesterolLevel"].cast("double")) \
       .withColumn("BloodSugureLevel", df["BloodSugureLevel"].cast("integer")) \
       .withColumn("ECGResults", df["ECGResults"].cast("integer")) \
       .withColumn("MaxHeartRateAchieved", df["MaxHeartRateAchieved"].cast("double")) \
       .withColumn("PainDuringExercise", df["PainDuringExercise"].cast("integer")) \
       .withColumn("ExerciseHeartChange", df["ExerciseHeartChange"].cast("double")) \
       .withColumn("PeakExercisePattern", df["PeakExercisePattern"].cast("double")) \
       .withColumn("MajorVesselsCount", df["MajorVesselsCount"].cast("double")) \
        .withColumn("TypeOfBloodDisorder", df["TypeOfBloodDisorder"].cast("integer")) \
       .withColumn("HeartDiseaseStatus", df["HeartDiseaseStatus"].cast("integer")) \
       .withColumn("PeakExercisePatterns", df["PeakExercisePatterns"].cast("integer")) \
       .withColumn("Food_Habits", df["Food_Habits"].cast("integer")) 

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# Logistic Regression

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Assuming 'df' is your Spark DataFrame and 'class' is the target column

# Define the feature columns
feature_cols = [col for col in df.columns if col != 'HeartDiseaseStatus']

# Combine all features into one vector named "features"
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Scale features using MinMaxScaler
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel = scaler.fit(df)
df = scalerModel.transform(df)

# Split the data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2])

# Define the model
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='HeartDiseaseStatus')

# Train the model
lrModel = lr.fit(train_data)

# Make predictions
predictions = lrModel.transform(test_data)

# Compute accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="HeartDiseaseStatus", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test Accuracy = %g" % (accuracy))


# COMMAND ----------

# Decision Tree

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Assuming 'df' is your Spark DataFrame and 'class' is the target column

# Define the feature columns
feature_cols = [col for col in df.columns if col != 'HeartDiseaseStatus']

# Combine all features into one vector named "assembledFeatures"
assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembledFeatures1")
df = assembler.transform(df)

# Split the data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=0)

# Scale features using StandardScaler
scaler = StandardScaler(inputCol="assembledFeatures1", outputCol="scaledFeatures5", withStd=True, withMean=False)
scalerModel = scaler.fit(train_data)
train_data = scalerModel.transform(train_data)
test_data = scalerModel.transform(test_data)

# Define the model
dt = DecisionTreeClassifier(featuresCol='scaledFeatures5', labelCol='HeartDiseaseStatus')

# Train the model
dtModel = dt.fit(train_data)

# Make predictions
predictions = dtModel.transform(test_data)

# Compute accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="HeartDiseaseStatus", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test Accuracy = %g" % (accuracy * 100))


# COMMAND ----------

# Random Forest

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Assuming 'df' is your Spark DataFrame and 'class' is the target column

# Define the feature columns
feature_cols = [col for col in df.columns if col != 'HeartDiseaseStatus']

# Combine all features into one vector named "assembledFeatures"
assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembledFeatures_1")
df = assembler.transform(df)

# Split the data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=0)

scaler = StandardScaler(inputCol="assembledFeatures_1", outputCol="scaledFeatures_1", withStd=True, withMean=False)
scalerModel = scaler.fit(train_data)
train_data = scalerModel.transform(train_data)
test_data = scalerModel.transform(test_data)

rf = RandomForestClassifier(featuresCol='scaledFeatures_1', labelCol='HeartDiseaseStatus', numTrees=10, impurity='entropy')

# Train the model
rfModel = rf.fit(train_data)

# Make predictions
predictions = rfModel.transform(test_data)

# Compute accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="HeartDiseaseStatus", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(accuracy)

print("Random Forest Test Accuracy = %g" % (accuracy * 100))

# COMMAND ----------

# GBT

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="HeartDiseaseStatus", outputCol="indexedLabel").fit(df)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
assembler = VectorAssembler(
    inputCols=[x for x in df.columns if x != 'HeartDiseaseStatus'],
    outputCol="features")

# Train a GBT model.
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="features", maxIter=10)

# Chain indexers and GBT in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler, gbt])

# Train model. This also runs the indexers.
model = pipeline.fit(df)

# Make predictions.
predictions = model.transform(df)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g" % accuracy)

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(predictions)
print("Test Precision = %g" % precision)

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(predictions)
print("Test Recall = %g" % recall)

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
f1 = evaluator.evaluate(predictions)
print("Test F1 Score = %g" % f1)

# COMMAND ----------


