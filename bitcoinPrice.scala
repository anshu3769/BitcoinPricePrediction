import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions._

import org.apache.spark.sql.functions._

object BitcoinPrice {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("BitcoinPrice");

    val sc = new SparkContext(conf);

    val spark = SparkSession
      .builder()
      .appName("BitcoinPrice")
      .config("spark.some.config.option", "some-value")
      .getOrCreate();

    import spark.implicits._

    val bitcoinDF = spark.read.option("header","true").csv("/Users/Anshu/Downloads/BitcoinPrice.csv");

    /* myDF.show(5); */
    /*val bitcoinDF = myDF.selectExpr("Date as Date","Close Price as BitCoinPrice"); */

    val separateTimeAndDateColumnDF = bitcoinDF
      .withColumn("Time",split(col("Date")," ")
        .getItem(1))
      .withColumn("Date",split(col("Date")," ")
        .getItem(0)) ;

    val dropTimeColumnDF = separateTimeAndDateColumnDF.drop("Time")
/*
    val windowSpec = Window.orderBy("Date")

    val transformedBitCoinDF = dropTimeColumnDF
      .withColumn("ChangeInPrice", $"BitCoinPrice" - when((lag("BitCoinPrice", 1)
        .over(windowSpec)).isNull, $"BitCoinPrice")
        .otherwise(lag("BitCoinPrice", 1)
          .over(windowSpec))); */

    dropTimeColumnDF.show(5);
    dropTimeColumnDF.printSchema();
    dropTimeColumnDF.write.option("header","true").format("csv").mode("overwrite").save("/Users/Anshu/Documents/formattedBitcoinDataNew.csv");
    //transformedBitCoinDF.write.format("csv").mode("overwrite").save("/Users/Anshu/Documents/formattedBitcoinDataNew.csv");
  }
}