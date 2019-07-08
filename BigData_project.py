from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import NaiveBayes


def predict(model, test, metricName):
    predictions = model.transform(test)
    if (metricName == "ROC Area"):
        evaluator = BinaryClassificationEvaluator()
    else:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=metricName)
    predict_value = evaluator.evaluate(predictions)
    return predict_value

def printROC(model):
    trainingSummary = model.summary
    roc = trainingSummary.roc.toPandas()

    plt.plot(roc['FPR'],roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def LoadingData ():
    spark = SparkSession.builder.appName('ml-bank').getOrCreate()
    df = spark.read.csv('spambase.data.csv', header = False, inferSchema = True)
    
    categoricalColumns = ['_c55','_c56']
    
    stages = []
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]

    label_stringIdx = StringIndexer(inputCol = '_c57', outputCol = 'label')
    stages += [label_stringIdx]

    numericCols = ['_c1','_c2','_c3','_c4','_c5','_c6','_c7','_c8','_c9','_c10','_c11','_c12','_c13','_c14','_c15','_c16','_c17','_c18','_c19',\
                   '_c20','_c21','_c22','_c23','_c24','_c25','_c26','_c27','_c28','_c29','_c30','_c31','_c32','_c33','_c34','_c35','_c36','_c37',\
                   '_c38','_c39','_c40','_c41','_c42','_c43','_c44','_c45','_c46','_c47','_c48','_c49','_c50','_c51','_c52','_c53','_c54']

    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]
    
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = ['label', 'features'] + numericCols + categoricalColumns
    df = df.select(selectedCols)  
    
    train, test = df.randomSplit([0.7, 0.3], seed = 2018)

    return train, test

def LogisticRegressionCl (train):
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
    lrModel = lr.fit(train)

    return lrModel

def DecisionTreeCl(train):
    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
    dtModel = dt.fit(train)
    
    return dtModel

def NaiveBayesCl(train):
    # create the trainer and set its parameters
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    # train the model
    nbModel = nb.fit(train)

    return nbModel

if __name__ == "__main__":
    
    test, train = LoadingData()
      
    measurments = ['f1', 'weightedPrecision', 'weightedRecall', 'accuracy', 'ROC Area']

    out = pd.DataFrame({
        'f1':[0.1,0.1,0.1],
        'weightedPrecision':[0.1,0.1,0.1],
        'weightedRecall':[0.1,0.1,0.1],
        'accuracy':[0.1,0.1,0.1],
        'ROC Area':[0.1,0.1,0.1]
        })
    
    LR = LogisticRegressionCl (train)
    printROC(LR)
    for k in measurments:
        out[k][0] = predict(LR,test,k)

    DT = DecisionTreeCl(train)
    for k in measurments:
        out[k][1] = predict(DT, test, k)

    NB = NaiveBayesCl(train)
    for k in measurments:
        out[k][2] = predict(NB, test, k)

    out.index = ['LogisticRegression', 'DecisionTree', 'NaiveBayes']
    out.to_csv('filename.csv')
