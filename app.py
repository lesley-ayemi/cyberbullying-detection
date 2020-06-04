from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def home():
	return render_template("home.html")

@app.route('/predict', methods = ['POST'])
def predict():
	#return render_template("result.html")
	

	df= pd.read_csv("data2.csv")

	df_data = df[["class", "comments"]]
	df_x = df_data["comments"]
	df_y = df_data["class"]

	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus)

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.3, random_state=42)

	from sklearn.linear_model import LogisticRegression
	clf = LogisticRegression()
	clf.fit(X_train, y_train)
	clf.score(X_test, y_test)

	# # #load the vectorizer
	# my_vectorizer = open("comment_vectorizer.pkl", "rb")
	# vector = joblib.load(my_vectorizer)
	# #load the model
	# my_model = open("myFinalModel.pkl","rb")
	# model_clf = joblib.load(my_model)


	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('home.html', name = data, prediction = my_prediction, user_comment = comment)

if __name__ == '__main__':
	app.run(debug = True)