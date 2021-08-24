from flask import Flask, render_template, request, session, url_for, redirect, flash
import pymysql

#=========
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy
import tflearn

import json
import pickle

from sklearn import preprocessing
import pandas as pd
import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.model_selection import KFold
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import warnings
import sys
#from sklearn.utils.extmath import np.dot
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
#=========Database Connection===
connection = pymysql.connect(host="localhost", user="root", password="root", database="chatbot")
cursor = connection.cursor()
#============start=======chatbot============


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
Corpus = pd.read_csv(r"algodata.csv",encoding='latin-1',error_bad_lines=False)
Tfidf_vect2 = TfidfVectorizer(max_features=100)
Tfidf_vect2.fit(Corpus['text'])
classesobt=['drugs','narcissist','socialanxiety']

Corpus = pd.read_csv(r"algodata.csv",encoding='latin-1',error_bad_lines=False)
Tfidf_vect2 = TfidfVectorizer(max_features=100)
Tfidf_vect2.fit(Corpus['text'])
Train_X_Tfidf1 = Tfidf_vect2.transform(Corpus['text'])
filename='Naive.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#chatbot
with open("narcissit.json", encoding="utf8") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            # print(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model11 = tflearn.DNN(net)

try:
    model11.load("model.tflearn")
except:

    model11.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model11.save("model.tflearn")


#==============end====chatbot=================


app = Flask(__name__)
app.secret_key = 'random string'

def dbConnection():
    try:
        connection = pymysql.connect(host="localhost", user="root", password="root", database="chatbot")
        return connection
    except:
        print("Something went wrong in database Connection")


@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        mobile = request.form.get("mobile")
        gender = request.form.get("gender")
        dob = request.form.get("dob")
        username = request.form.get("username")
        password = request.form.get("password")
        cursor.execute('SELECT * FROM userdetails WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        # print(account)
        if account:
            msg = 'user already exist'
            return redirect(url_for('index'))
        else:
            cursor.execute("insert into userdetails(fullname,gender,mobile,email,dob,username,password) values('" + name + "','" + gender + "','" + mobile + "','" + email + "','" + dob + "','" + username + "','" + password + "')")
            connection.commit()
            # return render_template('/index')
            msg = 'user already exist'
            return redirect(url_for('index'))
    else:
        #return render_template('index.html')
        return redirect(url_for('index'))


@app.route('/login', methods=["GET","POST"])
def login():
    msg = ''
    if request.method == "POST":
        session.pop('user',None)
        username = request.form.get("username")
        password = request.form.get("password")
        cursor.execute('SELECT * FROM userdetails WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        #print(account)
        if account:
            session['user'] = account[1]
            #return render_template('home.html')
            return redirect(url_for('chatbot'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    #return render_template('index.html', msg=msg)
    return redirect(url_for('index'))


#logout code
@app.route('/logout')
def logout():
    session.pop('user')
    return redirect(url_for('index'))


@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))


#bot start======
@app.route('/chatbot')
def chatbot():
    if 'user' in session:
        return render_template('chatbot.html', user=session['user'])
    return redirect(url_for('index'))

import wikipedia
def answerfromwiki(text): 
    complete_content = wikipedia.summary(text)
    print(complete_content)
    return complete_content
valobt=0
allstringis=''
@app.route("/get")
def get_bot_response():
    global valobt,allstringis
    valobt+=1
    #valobtained=valobtained+1
    userText = request.args.get('msg')
    results = model11.predict([bag_of_words(userText, words)])
    print(results)
    results_index = numpy.argmax(results)
    print(results_index)
    print(results[0][results_index])
    oppred=results[0][results_index]
    outdatagot = " \n"
    print(oppred)
    if oppred>0.3:
        
        outis=''    
        tag = labels[results_index]
        print(tag)
        #outdatagot = ""
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                outdatagot = random.choice(responses)
                print(outdatagot)
        outis=outdatagot
            
        print(outis)    
        if valobt%7==0:
            results1 = loaded_model.predict(Tfidf_vect2.transform([allstringis]))
            probais=loaded_model.predict_proba(Tfidf_vect2.transform([allstringis]))[:, 1]
            print(results,results1,probais)
            print('result1 is',classesobt[results1[0]])
            if probais>0.5:
            
                outis=classesobt[results1[0]]
                cursor.execute("insert into output(value) values('"+outis+"')")
                connection.commit()
        return str(' '+outis)
    else:
        outdatagot=outdatagot+" Answer from wikipedia \n"+answerfromwiki(userText)
        return str(outdatagot)



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)
model11.load("model.tflearn")
#bot end======-------------

#---------admin--------
@app.route('/admin',methods=["GET","POST"])
def admin():
    return render_template('admin.html')


@app.route('/adminlogin', methods=["GET","POST"])
def adminlogin():
    if request.method == "POST":
        try:
            #session.pop('user',None)
            username = request.form.get("username")
            #print(username)
            password = request.form.get("password")
            #print(password)
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (username, password))
            result = cursor.fetchone()
            if result:
                session['admin_name'] = result[1]
                session['adminid'] = result[0]
                flash("Successfully login")
                return redirect(url_for('adminhome'))
            else:
                flash("(Invalid mail id and password)")
                return redirect(url_for('admin'))
        except Exception as e:
            print(e)
            print("Exception occured at login")
            flash("Something went wrong")
            return redirect(url_for('admin'))
    return render_template('admin')


@app.route('/adminhome',methods=["GET","POST"])
def adminhome():
    if 'adminid' in session:
        admin_name = session['admin_name']
        return render_template('adminhome.html', uname=admin_name)
    return redirect(url_for('admin'))


@app.route('/stats',methods=["GET","POST"])
def stats():
    if 'adminid' in session:
        admin_name = session['admin_name']
        con = dbConnection()
        cursor = con.cursor()
        cursor.execute('SELECT * FROM userdetails')
        result = cursor.fetchall()
        return render_template('stats.html', result=result, uname=admin_name)
    return redirect(url_for('admin'))

from collections import Counter
@app.route('/userstats',methods=["GET","POST"])
def userstats():
    if 'adminid' in session:
        admin_name = session['admin_name']
        con = dbConnection()
        cursor = con.cursor()
        cursor.execute('SELECT value FROM output')
        result = list(cursor.fetchall())
        list1=[]
        for ik in result:
            list1.append(ik[0])

        #list1 =result[0]# ['sad', 'happy', 'depressed', 'sad', 'depressed','sad', 'happy', 'depressed']
        counts = dict(Counter(list1))
        print(counts)

        return render_template('userstats.html', counts=counts, uname=admin_name)
    return redirect(url_for('admin'))



#logout code
@app.route('/alogout')
def alogout():
    session.pop('admin_name')
    session.pop('adminid')
    return redirect(url_for('admin'))

#----- admin ------------


if __name__ == '__main__':
    #app.run(debug="True")
    app.run('0.0.0.0')
