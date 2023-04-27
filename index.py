from flask import Flask, render_template, request, session, url_for, Response
import pandas as pd
import numpy as np
from werkzeug.utils import redirect
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import linear_model

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn import metrics
from sklearn.metrics import pairwise_distances
import sklearn.metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

from math import sqrt
from sklearn.metrics import classification_report

import pygal

from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



from random import randint
import time
import json

app = Flask(__name__)
global Linear, RoandomForest ,ADA,Bagging, Gradient


def f(x_train, x_test, y_train, y_test):
    global X_trains, X_tests, y_trains, y_tests
    X_trains = pd.DataFrame(x_train)
    X_tests = pd.DataFrame(x_test)
    y_trains = pd.DataFrame(y_train)
    y_tests = pd.DataFrame(y_test)



    return X_trains, X_tests, y_trains, y_tests


def scores(score):
    global score1
    score1 = []
    # if sc
    return score1


@app.route('/')# constractors in flask reperesntation
def index():
    return render_template('index.html')


@app.route('/upload')
def registration():
    return render_template('uploaddataset.html')

@app.route('/bar_chart')
def bar_charts():
    line_chart = pygal.Bar()
    line_chart.title = 'Accuracy Scores of Different Algorithms'
    line_chart.add('Linear Regression', [pridect1])
    line_chart.add('Random Forest Regressor', [accuracyscore1])
    line_chart.add('Ada Boost Regressor', [accuracyscore2])
    line_chart.add('Bagging Regressor', [accuracyscore3])
    line_chart.add('Gradient Boost Regressor', [accuracyscore4])
    graph_data = line_chart.render_data_uri()
    return render_template('bar_chart.html',data=graph_data)


labels = [
    'Linear', 'RoandomForest ', 'ADA',
    'Bagging', 'Gradient'
]

values = [
    949.45,752.43, 1853.92,602.26, 17.19

]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", ]


@app.route("/data")
def chart_data(data=None):
    data_set = []

    for x in range(0, 12):
        y = randint(1, 12)
        data_set.append(y)

    data = {}

    data['set'] = data_set

    js = json.dumps(data)

    resp = Response(js, status=200, mimetype='application/json')

    return resp


@app.route("/bar_chart")
def hello(data=None):
    data = {}
    data['title'] = 'Chart'
    print(data['title'])
    print(data)

    return render_template('index1.html', data=data)


@app.route('/uploaddataset', methods=["POST", "GET"])
def uploaddataset_csv_submitted():
    if request.method == "POST":
        csvfile = request.files['csvfile']
        result = csvfile.filename
        file = "D:\\forecasting of crime\\" + result
        print(file)
        session['filepath'] = file

        return render_template('uploaddataset.html', msg='sucess')
    return render_template('uploaddataset.html')


@app.route('/view', methods=["POST", "GET"])
def view():
    global df
    session_var_value = session.get('filepath')

    print("session variable is=====" + session_var_value)
    df = pd.read_csv(session_var_value)
    # print(df)
    x = pd.DataFrame(df)

    return render_template("view.html", data=x.to_html())

    # return render_template('view.html', name=session_var_value, data=df.to_html())

@app.route('/traintest')
def traintestvalue():
    return render_template('traintestdataset.html')

@app.route('/traintestdataset', methods=["POST", "GET"])
def traintestdataset_submitted():
    if request.method == "POST":
        value = request.form['traintestvalue']
        print("train test value is=============" + value)
        value1 = float(value)
        print(value1)
        filepath = session.get('filepath')
      #  df1 = pd.read_csv(filepath)
        df1=df
        df1 = df1.fillna(0)

        X= df1.drop(['PERID', 'Criminal'], axis=1)#data preprcosessing
        y= df1['Criminal']#target variable(predict the criminal)
        # print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=value1)

        f(X_train, X_test, y_train, y_test)

        X_train1 = pd.DataFrame(X_train)
        X_trainlen = len(X_train)

        y_test1 = pd.DataFrame(y_test)
        y_testlen = len(y_test)


        return render_template('traintestdataset.html', msg='sucess', data=X_train1.to_html(),X_trainlenvalue=X_trainlen, y_testlenval=y_testlen)
    return render_template('traintestdataset.html')


@app.route('/modelperformance')
def modelperformances():
    return render_template('modelperformance.html')


@app.route('/modelperformance',methods=["POST","GET"])
def selected_model_submitted():
    global pridect1,accuracyscore1,accuracyscore2,accuracyscore3,accuracyscore4
    if request.method == "POST":
        selectedalg = int(request.form['algorithm'])

        print(X_trains)
        print(X_tests)

        print(y_trains)
        print(y_tests)

        if (selectedalg == 1):
            #model=tree.DecisionTreeRegressor()
            model=linear_model.LinearRegression()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            pridect1=model.score(X_trains, y_trains)
            #accuracyscore = accuracy_score(y_tests, y_pred)
            #print(pridect)
            return render_template('modelperformance.html', msg="pridect", score=pridect1,
                                   model="LinearRegression")

        if (selectedalg == 2):
            model = RandomForestRegressor()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore1 = model.score(X_trains, y_trains)
            #accuracyscore = accuracy_score(y_tests, y_pred)
            print(accuracyscore1)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore1, model="RandomForestRegressor")

        if (selectedalg == 3):
            model = AdaBoostRegressor()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore2 = model.score(X_trains, y_trains)
            #accuracyscore = accuracy_score(y_tests, y_pred)
            print(accuracyscore2)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore2, model="AdaBoostRegressor")

        if (selectedalg == 4):
            model = BaggingRegressor()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore3 = model.score(X_trains, y_trains)
            #accuracyscore = accuracy_score(y_tests, y_pred)
            print(accuracyscore3)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore3, model="BaggingRegressor")

        if (selectedalg == 5):
            model = GradientBoostingRegressor()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore4 = model.score(X_trains, y_trains)
            #accuracyscore = accuracy_score(y_tests, y_pred)
            print(accuracyscore4)
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore4, model="GradientBoostingRegressor")
    return render_template('modelperformance.html')


@app.route('/prediction')
def predictions():
    return render_template('prediction.html')


@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        list1 = []
        Open = request.form['Open']
        print(Open)
        #Turnover = request.form['Turnover']
#        list1.extend([Open]) #Tunnover
        print(list1)
        list1 = list(Open.split(","))

        model = linear_model.LinearRegression()
        model.fit(X_trains, y_trains)
        # y_pred = model.predict(X_tests)
        predi = model.score(X_trains, y_trains)
        # accuracyscore = accuracy_score(y_tests, y_pred)
        print(predi)
        model = AdaBoostRegressor()
        model.fit(X_trains, y_trains)
        predi = model.predict([list1])
        print(predi)
        pre = predi
        print(pre)

        return render_template('prediction.html', msg='predictsucess',predvalue=predi)
    return render_template('prediction.html')



if __name__ == '__main__':
    app.secret_key = ".."
    app.run()




