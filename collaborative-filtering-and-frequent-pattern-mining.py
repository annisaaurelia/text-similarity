# encoding=utf8
from flask import Flask, jsonify, render_template, request, url_for, redirect
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SQLContext
from pyspark import SparkContext
from pyspark.mllib.fpm import FPGrowth
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle, requests, ast
import numpy as np
import pandas as pd
import sys
import json
import math

reload(sys)
sys.setdefaultencoding('utf8')

PER_PAGE = 10
ZOMATO = 'example/zomato.json'
PICKLE = 'example/lsc.p'
STOPWORD = 'stopword-ind.txt'

app = Flask(__name__)

zomato = pd.read_json(ZOMATO)
zomato = zomato.fillna('na')

model = ""
with open(PICKLE) as m:
  model = pickle.load(m)

stopwords = []
with open(STOPWORD) as s:
  for w in s:
    w = w.strip()
    stopwords.append(w)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

@app.route('/api/trainme', methods=['POST'])
def trainme():
  with open(PICKLE, 'rwb') as m:
    model = pickle.load(m)

  return "Yes Please"

@app.route('/api/search/<query>')
@app.route('/api/search/<query>/<page>')
def search(query, page = 1):
  query = stemmer.stem(str(query))
  query = ' '.join([w for w in query.split() if w not in stopwords])
  prediction = model.predict([query])

  try:
    page = int(page)
  except Exception as e:
    page = 1

  if page < 1:
    page = 1

  i = (page - 1) * PER_PAGE
  j = PER_PAGE * page
  results = zomato.iloc[prediction[i:j]]
  return jsonify(results.to_dict(orient='records'))

@app.route("/<query>")
@app.route("/<query>/<page>")
def search_page(query, page=1):
  # WHY???
  # print '[DEBUG] SEND REQUEST TO - %s' % (url_for('search', query=q, _external=True),)
  # results = requests.get(url_for('search', query=q, _external=True))
  # print results.json()

  query = stemmer.stem(str(query))
  query = ' '.join([w for w in query.split() if w not in stopwords])
  prediction = model.predict([query])

  try:
    page = int(page)
  except Exception as e:
    page = 1

  if page < 1:
    page = 1

  i = (page - 1) * PER_PAGE
  j = PER_PAGE * page
  results = zomato.iloc[prediction[i:j]]
  results = results.to_dict(orient='records')

  for r in results:
    r['alamat'] = ast.literal_eval(r['lokasi'])['alamat']
    r['similar'] = get_similar(r['restaurant_id'])

  return render_template('search-result.html', results=results)

@app.route("/<int:user_id>/<query>/sort")
def sortingSearch(user_id,query):
  temp=query
  query = stemmer.stem(str(query))
  query = ' '.join([w for w in query.split() if w not in stopwords])
  prediction = model.predict([query])
  results = zomato.iloc[prediction[0:20]]
  results = results.to_dict(orient='records')
  results = sort_by_rating(user_id,results)
  return render_template('sorted-result.html',  query=temp, user_id=user_id, results=results)

# ----- USING SPARK ENVIRONMENT ----------
@app.route("/<int:user_id>")
def index(user_id):
  top_ratings = get_top_ratings(user_id)
  q = request.args.get('q', '').strip()
  if (q != ''):
    return redirect(url_for('search_page', query=q))
  return render_template('index.html', ratings=top_ratings)

def get_top_ratings(userId):
  listRestaurant = df.select('restaurant_id').distinct().rdd.map(lambda r: r[0])
  # mengetahui berapa rating yang diberikan oleh userid terhadap restaurant_id
  join = listRestaurant.map(lambda x: (userId, x)).collect()
  unseen_rating = sqlContext.createDataFrame(join, ["user_id", "restaurant_id"])
  predictions = sorted(modelCF.transform(unseen_rating).collect(), key=lambda r: float('-inf') if math.isnan(r[2]) else r[2],  reverse=True)[0:9]
  top=[]
  for p in predictions:
    search = zomato[zomato['restaurant_id'] == p[1]]
    if not search.empty:
        top.append({'nama':search['nama'].values[0].upper(),'rating':search['rating'].values[0],'link':search['link'].values[0]})
  return top

def sort_by_rating(userId, results):
    id =[]
    for r in results:
        id.append(r['restaurant_id'])
    listRestaurant = sc.parallelize(id)
    # mengetahui berapa rating yang diberikan oleh userid terhadap restaurant_id
    join = listRestaurant.map(lambda x: (userId, x)).collect()
    unseen_rating = sqlContext.createDataFrame(join, ["user_id", "restaurant_id"])
    predictions = sorted(modelCF.transform(unseen_rating).collect(), key=lambda r: float('-inf') if math.isnan(r[2]) else r[2],  reverse=True)
    top=[]
    for p in predictions:
        search = zomato[zomato['restaurant_id'] == p[1]]
        if not search.empty:
            top.append({'alamat':ast.literal_eval(search['lokasi'].values[0])['alamat'],'similar':get_similar(search['restaurant_id'].values[0]),'nama':search['nama'].values[0].upper(),'rating':search['rating'].values[0],'link':search['link'].values[0],'harga':search['harga'].values[0] })
    return top

def cf():
    # COLLABORATIVE FILTERING
    global sc, sqlContext, df, modelCF
    sc = SparkContext(appName="lapearl")
    sqlContext = SQLContext(sc)
    # please change the file location
    lines = sc.textFile("file:///home/annisa/spark-1.6.0/bin/example/ratings.csv")
    parts = lines.map(lambda row: row.split(","))
    ratingsRDD = parts.map(lambda p: Row(rating=float(p[0]), restaurant_id=int(p[1]), user_id=int(p[2])))
    df = sqlContext.createDataFrame(ratingsRDD)
    (training, test) = df.randomSplit([0.6, 0.4], seed=0)
    # Build the recommendation model using ALS on the training data
    als = ALS(rank=5, maxIter=5, regParam=0.01, userCol="user_id", itemCol="restaurant_id", ratingCol="rating")
    modelCF = als.fit(training)

def freqItem():
    # FREQUENT PATTERN MINING
    global itemSet
    # please change the file location
    data = sc.textFile("file:///home/annisa/spark-1.6.0/bin/example/restaurants.txt")
    transactions = data.map(lambda line: line.strip().split(' '))
    model = FPGrowth.train(transactions, minSupport=0.0003, numPartitions=10)
    itemSet = model.freqItemsets().collect()

def get_similar(restaurant_id):
    similar=name=[]
    for fi in itemSet:
        for i in fi[0]:
            if len(fi[0]) > 1 and str(i) == str(restaurant_id):
                similar = fi[0]
                break
    for s in similar: 
        if(int(s) != restaurant_id):
            search = zomato[zomato['restaurant_id'] == int(s)]
            if not search.empty:
              name.append({'nama':search['nama'].values[0].upper(), 'link':search['link'].values[0]})
    return name

if __name__ == "__main__":
    cf()
    freqItem();
    app.run()