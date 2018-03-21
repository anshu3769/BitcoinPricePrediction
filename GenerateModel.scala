/* Generation of Prediction Model for predicting Bitcoin price
based on number of tweets and number of google searches on previous
day. This model user Linear Regression to correlate the features
 */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD


object GenerateModel {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("JoinData");

    val sc = new SparkContext(conf);

    val spark = SparkSession
      .builder()
      .appName("JoinData")
      .config("spark.some.config.option", "some-value")
      .getOrCreate();


    val rawDataRDD = sc.textFile("/Users/Anshu/Documents/joinedData.csv");

    /*val NumOfDataPoints = rawDataRDD.count();
    println("Data points are "+ NumOfDataPoints); */


    /*Split the RDD into two RDDs (Testing and Training Data with 80 percent being
    *training data and the remaining data is for testing the model)*/
    val splitTrainingAndTesting = rawDataRDD.randomSplit(Array(0.8,0.2),2);

    /*Parse the training RDD*/
    /* Label = BitcoinPrice
       Feature Added = Number of bitcoin searches on google
       Feature to be added = Number of tweets with keyword "bitcoin"
     */
    val parsedTrainingRDD = rawDataRDD.map {line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble , Vectors.dense(parts(1).toDouble , parts(2).toDouble ))
    }.cache() ;

    /*Parse the testing data */
    /*
    val parsedTestingRDD = splitTrainingAndTesting(1).map {line =>
      val parts = line.split(',')
      LabeledPoint(parts(1).toDouble , Vectors.dense(parts(2).toDouble, parts(3).toDouble))
    }.cache() ; */

    val cnt = parsedTrainingRDD.count();
    println("Count is " + cnt);

    parsedTrainingRDD.foreach(println);



    /* Paramaters for model generation */
    val numIterations = 100
    val stepSize = 0.0001
    /*val model = LinearRegressionWithSGD.train(parsedTrainingRDD, numIterations, stepSize); */

    val algorithm = new LinearRegressionWithSGD()
    algorithm.setIntercept(true)
    algorithm.optimizer
      .setNumIterations(numIterations)
      .setStepSize(stepSize)

    /* DONE : Model trained for number of bitcoin  searches on google
    * TO DO : Train the model on number of tweets and number of tweets
    * FEATURE REMOVED AFTER ANALYSIS : The model does not fit on number of bitcoin transactions.*/

    val model = algorithm.run(parsedTrainingRDD);

    /* Coefficient and intercept of the model */
    println("weights: %s, intercept: %s".format(model.weights, model.intercept));

    println("MODEL IS::  ")
    println(model);

    /*val lrModel = lr.fit(parsedTrainingRDD); */

    /* Evaluate model on training examples and compute training error */


    val valuesAndPreds = parsedTrainingRDD.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    /*Save the predicted values */
    valuesAndPreds.saveAsTextFile("/Users/Anshu/Documents/PredictedValue");

    valuesAndPreds.foreach(println);

    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((p - v), 2) }.mean()
    println("training Mean Squared Error = " + MSE)
    /*val predictionAndLabel = valuesAndPreds.zip(parsedTestingRDD.map(._label)) */

    // Save and load model
    model.save(sc, "/Users/Anshu/Documents/scalaLinearRegressionWithSGDModelNew1");
    val sameModel = LinearRegressionModel.load(sc, "/Users/Anshu/Documents/scalaLinearRegressionWithSGDModelNew1");

    println("Print here")
    /* To predict the value of bitcoin when number of tweets are 105.01K and google searches are 600K. */
    println(sameModel.predict(Vectors.dense(105.01, 600)));


  }
}
