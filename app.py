#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import time
from sklearn import metrics 
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from feature import generate_data_set
# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv("phishing.csv")
#droping index column
data = data.drop(['Index'],axis = 1)
# Splitting the dataset into dependant and independant fetature

X = data.drop(["class"],axis =1)
y = data["class"]

# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
gbc.fit(X,y)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", xx= -1)

@app.route('/dashboard')
def dashboard():
   return render_template('dashboard.html', xx= -1)

@app.route('/index')
def index1():
   return render_template('index.html')


@app.route('/home')
def home():
   return render_template('index.html')


@app.route('/aboutus')
def aboutus():
   return render_template('aboutus.html')

@app.route('/register')
def registration():
   return render_template('register.html')


@app.route('/login')
def login():
   return render_template('login.html')



@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,30) 
        y_pred =gbc.predict(x)[0]

        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        for index,data in enumerate(X.columns):
        	print(data,': ',x[0][index])
        

        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]

        safe_per = str("{0:.0f}".format(y_pro_non_phishing*100))
        unsafe_per = str("{0:.0f}".format(y_pro_phishing*100))

        data = {'Safe ('+safe_per+' %)':int(safe_per), 'Unsafe ('+unsafe_per+' %)':int(unsafe_per)}
        courses = list(data.keys())
        values = list(data.values())
        fig = plt.figure(figsize = (10, 5))
        plt.bar(courses, values, color =['green','red'])

        plt.xlabel("URL Statistics")
        plt.ylabel("Percentage")
        plt.title("Result")
        filename = 'static/output/'+str(time.time()).split('.')[0]+'.png'
        plt.savefig(filename)

        
        return render_template('dashboard.html',xx =round(y_pro_non_phishing,2),url=url,filename=filename )
        # else:
        #     pred = "It is {0:.2f} % unsafe to go ".format(y_pro_non_phishing*100)
        #     return render_template('index.html',x =y_pro_non_phishing,url=url )
    return render_template("dashboard.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)