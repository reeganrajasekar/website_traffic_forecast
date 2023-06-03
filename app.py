from flask import Flask, url_for, render_template, request, redirect, flash
import csv
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__, template_folder = "template")

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict',methods = ['POST', 'GET'])
def predict():
  title = request.form.get("title")
  data =[]
  f = request.files['file']
  f.save("pre.csv")
  with open("pre.csv", mode ='r')as file:
    csvFile = csv.reader(file)
    end=""
    for lines in csvFile:
      if lines[0]!='Day' and lines[1]!='Views':
        data.append([int(lines[0]),int(lines[1])])
        end=int(lines[0])
  X = np.array(data)[:,0].reshape(-1,1)
  y = np.array(data)[:,1].reshape(-1,1)
  i=0
  to_predict_x= []
  while(i<16):
    to_predict_x.append(end+i)
    i=i+1
  to_predict_x= np.array(to_predict_x).reshape(-1,1)
  real_x=[]
  real_y=[]
  for j in data:
    real_x.append(j[0])
    real_y.append(j[1])
  regsr=LinearRegression()
  regsr.fit(X,y)
  predicted_y= regsr.predict(to_predict_x)
  return render_template("predict.html",title=title,x=to_predict_x,y=predicted_y,real_x=real_x,real_y=real_y)


if __name__ == '__main__':
   app.run(debug=True)
