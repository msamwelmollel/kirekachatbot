#  <style>
# * {
#   box-sizing: border-box;
# }
# .menu {
#   float: left;
#   width: 20%;
# }
# .menuitem {
#   padding: 8px;
#   margin-top: 7px;
#   border-bottom: 1px solid #f1f1f1;
# }
# .main {
#   float: left;
#   width: 60%;
#   padding: 0 20px;
#   overflow: hidden;
# }
# .right {
#   background-color: lightblue;
#   float: left;
#   width: 20%;
#   padding: 10px 15px;
#   margin-top: 7px;
# }

# @media only screen and (max-width:800px) {
#   /* For tablets: */
#   .main {
#     width: 80%;
#     padding: 0;
#   }
#   .right {
#     width: 100%;
#   }
# }
# @media only screen and (max-width:500px) {
#   /* For mobile phones: */
#   .menu, .main, .right {
#     width: 100%;
#   }
# }
# </style>




from flask import Flask,request, url_for, redirect, render_template, jsonify

#import pandas as pd
#questionanswer.pkl filter=lfs diff=lfs merge=lfs -text
import pickle
#import numpy as np
#import numpy as np 
import joblib
import string
#from nltk.corpus import stopwords
#import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline

# import logging

# app.logger.addHandler(logging.StreamHandler(sys.stdout))
# app.logger.setLevel(logging.ERROR)


def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]
model = pickle.load(open('questionanswer.pkl', 'rb'))
# Initalise the Flask app
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(model.predict([userText.lower()])[0])


if __name__ == "__main__":
    app.run(port = 5000, debug=True)



# model = joblib.load('joblibquestionandanswer.pkl')


# # Loads pre-trained model

# #model = 'joblibquestionandanswer'





# # filename = 'questionanswer.pkl'

# # def read_filemodel():
# #     return pickle.load(open(filename, 'rb'))

# # x = read_filemodel()


# # with open(filename , 'rb') as f:
# #     model = pickle.load(f)






# @app.route('/')
# #@app.route('/', methods=['GET', 'POST'])
# def home():
#     return render_template("home.html")

# @app.route('/predict',methods=['POST'])
# def predict():

#     #model = pickle.load(open(filename, 'rb'))
#     int_features = [x for x in request.form.values()]
#     #print('**********')
#     #print(int_features)
#     #print(int_features)
#     #int_features = str(request.form.values())
#     final = ' '.join([str(elem) for elem in int_features])
#     #data_unseen =    final #pd.DataFrame([final], columns = cols)
#     #prediction = model.predict(['What are you doing'])[0] # predict_model(model, data=data_unseen, round = 0)
#     #print(len(final))
#     #print('*******')
#     #print([final])
#     #print([final]))
#     prediction = model.predict([final])[0]
# #     prediction =   'all is well'
#     #prediction = int(prediction.Label[0])
#     return render_template('home.html',pred='Kireka anasema: {}'.format(prediction))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     #model = pickle.load(open(filename, 'rb'))
#     #data_unseen = data #pd.DataFrame([data])
#     #prediction = model.predict(['What are you doing'])[0] # predict_model(model, data=data_unseen)
#     output =    'prediction'   #.Label[0]
#     return jsonify(output)

# if __name__ == '__main__':
    
# #     filename = 'questionanswer.pkl'

# #     def read_filemodel():
# #         return pickle.load(open('questionanswer.pkl', 'rb'))

# #     x = read_filemodel()
# #     model = pickle.load(open('questionanswer.pkl', 'rb'))
# #     def cleaner(x):
# #     return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]

# #     Pipe = Pipeline([
# #         ('bow',CountVectorizer(analyzer=cleaner)),
# #         ('tfidf',TfidfTransformer()),
# #         ('classifier',DecisionTreeClassifier())
# #     ])
# #     pickle.dump(Pipe, open(filename, 'wb'))
# #     model = pickle.load(open('questionanswer.pkl', 'rb'))#
    
#     app.run(port = 5000, debug=True)




# #print(model.predict(['What are you doing'])[0])
