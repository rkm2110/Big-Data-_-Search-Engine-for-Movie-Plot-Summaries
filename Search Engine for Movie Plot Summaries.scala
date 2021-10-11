// Databricks notebook source
//feetching the movie data
val input_movies = sc.textFile("/FileStore/tables/movie_metadata.tsv")
val input_summaries = sc.textFile("/FileStore/tables/plot_summaries.txt")
//fetching the list of stopwords from a ext file
val input_stopWords = sc.textFile("/FileStore/tables/stopwords.txt")

// COMMAND ----------

//splitting stopwords and converting into set
val mapped_stopWords = input_stopWords.flatMap(a => a.split(",")).collect().toSet


// COMMAND ----------

// mapping summaries and movies from input files
val mapped_summaries = input_summaries.map(a => a.toLowerCase.split("""\s+""")).map(a => a.map(b => b).filter(word => mapped_stopWords.contains(word) == false)).map(a=>(a(0),a.drop(1))).cache()
val mapped_movies = input_movies.map(_.split("\t")).map(a => (a(0), a(2)))

// COMMAND ----------

// converting into collection
mapped_summaries.collect()
mapped_movies.collect()

// COMMAND ----------

//Calculating Term frequency of all the words
val tf = mapped_summaries.flatMap(x => x._2.map(y => ((x._1, y), 1))).reduceByKey((x,y) => x+y).map(x => (x._1._2, (x._1._1, x._2)))
//Calculating document frequency
val df = tf.map(x => (x._1, 1)).reduceByKey((x,y) => x+y).map(x => (x._1, (x._2, math.log(N/x._2))))

//calculating tf-Idf
var tfIdf = df.join(tf).map(x => (x._2._2._1, (x._1, x._2._2._2, x._2._1._1, x._2._1._2, x._2._2._2 * x._2._1._2)))
tfIdf = mapped_movies.join(tfIdf).map(x => x._2).cache()

// COMMAND ----------

//fetching the list of queries from the search file
val query = sc.textFile("/FileStore/tables/search_word.txt").collect

// COMMAND ----------

//performing search
for (searchWord <- query) {
  println("Query : "+ searchWord)
  println("Searching...")
  println()
  var queryWords = searchWord.split(" ").map(_.toLowerCase.trim)
  //for multiple query words
  if (queryWords.length > 1) {
    var cosTf = sc.parallelize(queryWords).map(x => (x, 1)).reduceByKey((x,y) => x+y)
    var dfWord = tfIdf.map(x => x._2).map(x => (x._1, (x._3, x._4)))
    var cosTfIdf = cosTf.leftOuterJoin(dfWord).map(x => (x._1, if (x._2._2.isEmpty) 0 else x._2._1 * x._2._2.get._2))

    var mergedDocWt = tfIdf.map(x => (x._2._1, (x._1, x._2._5))).join(cosTfIdf).map(x => x._2).map(x => (x._1._1, x._1._2, x._2)) 
    var dot = mergedDocWt.map(x => (x._1, (x._2 * x._3, x._3 * x._3, x._2 * x._2))).reduceByKey((x,y) => ((x._1 + y._1, x._2 + y._2, x._3 + y._3)))
    var doc_set = dot.map(x => (x._1, x._2._1/(math.sqrt(x._2._2) * math.sqrt(x._2._3)))).sortBy(-_._2).map(_._1).take(10)
    println("Top 10 query results :")
    doc_set.foreach {println}
  }
  //for single query word
  else { 
    var doc_set = tfIdf.filter(x => x._2._1 == queryWords.head).sortBy(-_._2._5).map(_._1).take(10)
   println("Top 10 query results :")
    doc_set.foreach {println}
  }
  println("-------------------------------------------------------")
}
