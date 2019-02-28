from pyspark import SparkConf, SparkContext
import os
from pyspark.mllib.recommendation import Rating, ALS
import pandas as pd
from collections import defaultdict

from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import re
import time

if __name__ == '__main__':
    conf = SparkConf().setAppName('Recommand App')
    sc = SparkContext(conf=conf)

    rawData = sc.textFile('/Users/xiewenkang/PycharmProjects/sparkML/ml-latest-small/ratings.csv')
    rawRating = rawData.filter(lambda line: 'm' not in line).map(lambda x: x.split(','))
    rating = rawRating.map(lambda line: Rating(int(line[0]), int(line[1]), float(line[2])))
    model = ALS.train(ratings=rating, rank=50, iterations=10, lambda_=0.01)
    userFeature = model.userFeatures()

    raw = pd.read_csv('/Users/xiewenkang/PycharmProjects/sparkML/ml-latest-small/ratings.csv')
    userId = set(raw['userId'])

    movieId = set(raw['movieId'])

    titles_dict = sc.textFile('/Users/xiewenkang/PycharmProjects/sparkML/ml-latest-small/movies.csv').filter\
        (lambda line: '(' in line).map(lambda line: (int(line.split(',')[0]), line.split(',')[1])).collectAsMap()

    userProducts = rating.map(lambda  rating:(rating.user, rating.product))
    predictions = model.predictAll(userProducts).map(lambda rate: ((rate.user, rate.product), rate.rating))
    ratingAndPredictions = rating.map(lambda rate:((rate.user, rate.product), rate.rating)).join(predictions)

    predictedAndTrue = ratingAndPredictions.map(lambda x: (x[1][0], x[1][1]))
    regressionMetrics = RegressionMetrics(predictedAndTrue)

    print('RMSE =  %f' % regressionMetrics.meanSquaredError)
    print('MSE = %f' % regressionMetrics.rootMeanSquaredError)

    for user in sorted(userId):
        result = model.recommendProducts(user, 10)
        moviesForUser = rating.groupBy(lambda x: x.user).mapValues(list).lookup(user)
        print("=================================================================================")
        print("The moive '%d' have been rated, predict result, and the distance shows below: " % user)
        for user_rating in sorted(moviesForUser[0], key=lambda x: x.rating, reverse=True):
            movie = user_rating.product
            predict_res = model.predict(user_rating.user, movie)
            print("movie:%s\tactual:%.2f\tpredict:%.2f\tdistance:%.2f" % (
                titles_dict[user_rating.product], user_rating.rating, predict_res,
                abs(user_rating.rating - predict_res)))
        print("===================================================")


        print("To user '%d' recommand 10 movie list:" % result[0].user)
        for i in range(len(result)):
            print('Rate %i: %s\trating:%.2f' % (
            i, titles_dict[result[i].product], model.predict(user, result[i].product)))

        print()
        time.sleep(1)

