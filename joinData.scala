import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD


object JoinData {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("JoinData");

    val sc = new SparkContext(conf);

    val spark = SparkSession
      .builder()
      .appName("JoinData")
      .config("spark.some.config.option", "some-value")
      .getOrCreate();

    import spark.implicits._

    /* Number of transactions per day*/
    val txnDF = spark.read.csv("/Users/Anshu/Documents/formattedTransactionData.csv");

    /* Daily Bitcoin Price */
    val btcDF = spark.read.csv("/Users/Anshu/Documents/formattedBitcoinDataNew.csv");

    /* Daily number of tweets with "bitcoin" as a keyword */
    val tweetDF = spark.read.csv("/Users/Anshu/Documents/tweets.csv");

    /* Number of daily google searches about bitcoin */
    val googleTrendsDF = spark.read.csv("/Users/Anshu/Downloads/9monthsData.csv");


    /* Name the columns */
    val transactionDF = txnDF.selectExpr("_c0 as Date","_c1 as NumberOfTransactions","_c2 as ChangeInNumberOfTransaction");

    val transactionDFNew = transactionDF.selectExpr("Date","cast(NumberOfTransactions as Double) NumberOfTransactions");


    val bitcoinDF = btcDF.selectExpr("_c0 as Date","_c1 as BitcoinPrice","_c2 as ChangeInBitcoinPrice");

    val bitcoinDFNew = bitcoinDF.selectExpr("Date","cast(BitcoinPrice as Double) BitcoinPrice");

    /*val bitcoinDF = btcDF.selectExpr("_c0 as Date","_c1 as BitcoinPrice"); */

    val tweetFeedDF = tweetDF.selectExpr("_c0 as Date","_c1 as NumberOfTweets");

    val googleTrend = googleTrendsDF.selectExpr("_c0 as Date","_c1 as GoogleTrends");

    val googleTrends = googleTrend.selectExpr("Date","cast(GoogleTrends as Double) GoogleTrends");

    transactionDFNew.printSchema();

     /* txnMax = transactionDFNew.select("NumberOfTransactions").rdd.map(r =>r.getAs[Double]("NumberOfTransactions")).max(); */
     /*val txnMax = transactionDFNew.select("NumberOfTransactions").rdd.max(); */
    /* val priceMax = bitcoinDFNew.select("BitcoinPrice").rdd.map(r =>r.getAs[Double]("BitcoinPrice")).max();
     val googleSearchMax = googleTrends.select("GoogleTrends").rdd.map(r=>r.getAs[Double]("GoogleTrends")).max(); */
    /*val txnMax = transactionDFNew.select(max("NumberOfTransactions")).getAs[Int]; */
   /*println("PRINT VALUES");
    println(txnMax);
    println("   ");
    println(priceMax);
    println("   ");
    println(googleSearchMax); */



    /*Join transaction and price data */
    val joinTransactionAndPrice = transactionDF.join(bitcoinDFNew,Seq("Date"));

    //Join transaction-price data with tweet data
   /* val joinTransactionPriceAndTweet  = joinTransactionAndPrice.join(tweetFeedDF,Seq("Date")); */

    /* Join transaction-price-tweet data with Google trends */
    val joinTransactionPriceTweetAndTrends = joinTransactionAndPrice.join(googleTrends,Seq("Date"));

    joinTransactionPriceTweetAndTrends.show(2);
    joinTransactionPriceTweetAndTrends.printSchema();

    val correlation = joinTransactionPriceTweetAndTrends.stat.corr("BitcoinPrice","GoogleTrends");



    println("CORREALTION" + correlation);

    joinTransactionPriceTweetAndTrends.write.format("csv").save("/Users/Anshu/Documents/joinedDataNew46.csv");

    val rawDataRDD = sc.textFile("/Users/Anshu/Documents/joinedDataNew46.csv");

    val NumOfDataPoints = rawDataRDD.count();
    println("Data points are "+ NumOfDataPoints);

    //val header = rawDataRDD.first;

    //val withoutHeaderRDD = rawDataRDD.filter(_(0) != header(0));

    /*Split the RDD into two RDDs (Testing and Training Data with 80 percent being
    *training data and the remaining data is for testing the model)*/
    val splitTrainingAndTesting = rawDataRDD.randomSplit(Array(0.8,0.2),2)

     /*Parse the training RDD*/
    /* Label = BitcoinPrice
       Feature Added = Number of bitcoin searches on google
       Feature to be added = Number of tweets with keyword "bitcoin"
     */
    val parsedTrainingRDD = splitTrainingAndTesting(0).map {line =>
                                          val parts = line.split(',')
                                          LabeledPoint(parts(3).toDouble , Vectors.dense(parts(4).toDouble ))
    }.cache() ;

    /*Parse the testing data */
    val parsedTestingRDD = splitTrainingAndTesting(1).map {line =>
      val parts = line.split(',')
      LabeledPoint(parts(3).toDouble , Vectors.dense(parts(4).toDouble ))
    }.cache() ;

    val cnt = parsedTrainingRDD.count();
    println("Count is " + cnt);

    parsedTrainingRDD.foreach(println);

    /* Paramaters for model generation */
    val numIterations = 200
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

    valuesAndPreds.foreach(println);

    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("training Mean Squared Error = " + MSE)
    /*val predictionAndLabel = valuesAndPreds.zip(parsedTestingRDD.map(._label)) */

    // Save and load model
    model.save(sc, "/Users/Anshu/Documents/scalaLinearRegressionWithSGDModelNew")
    val sameModel = LinearRegressionModel.load(sc, "/Users/Anshu/Documents/scalaLinearRegressionWithSGDModelNew")
  }
}