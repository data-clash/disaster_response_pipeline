import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_msg_clean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals   
    #count per genre
    genre_count = df.groupby('genre').count()['message']
    
    #genre names
    genre_names = list(genre_count.index)
    
    #genre percent
    gen_per = round(100*genre_count/genre_count.sum(), 2)
    #for ind, i in enumerate(gen_per):
    #    gen_per[ind] = "{}%".format(i)
    #g = list(map("{}%".format, gen_per))
    #print(gen_per)
    gen = list(genre_count.index)
    cat_num = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    cat_num = cat_num.sort_values(ascending = False)
    cat = list(cat_num.index)

    # create visuals  
    '''graphs = [
        
    ]'''
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cat,
                    y=cat_num
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": gen_per,
                  "y": genre_names
                },                               
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": gen,
                "values": genre_count
              }
            ],
            "layout": {
              "title": "Count and Percent of Messages by Genre"
            }
        }
    ]    

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    print(ids)
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    print(graphJSON)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
   
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(model.predict([query])[0])
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True, threaded=False)


if __name__ == '__main__':
    main()