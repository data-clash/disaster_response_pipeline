import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import TransformerMixin,BaseEstimator
from nltk.tokenize import word_tokenize,sent_tokenize

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
from nltk.stem.wordnet import WordNetLemmatizer

def load_data(database_filepath):
    """
    Load cleaned data from database_filepath
    INPUT
    database_filepath --filepath to csv dataset   
    OUTPUT
    X - message column to predict Y values
    Y - list of columns to be predicted
    category_names - name of Y column names
    """
    
    # load data from database
    database_filepath = 'sqlite:///' + database_filepath
    #engine = create_engine('sqlite:///disasterResponse.db')
    engine = create_engine(database_filepath)
    df = pd.read_sql('SELECT * FROM disaster_msg_clean',con=engine,index_col='id')    
    
    #select message column
    X = df['message'].dropna()

    #skip first three columns : message, original, genre and select the rest of the columns
    Y = df.iloc[:,3:].dropna()
    
    #name of Y column names 
    category_names = Y.columns.values
    
    return X,Y,category_names

def tokenize(text):
    """
    Tokenize the text
    
    Input:
        text -> Text message to tokenize
    Output:
        clean_tokens -> List of tokens extracted from text
    """
    
    #declare url regex to detect a url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'  

    
    #replace the url with text : urlplaceholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #normalize text
    #text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())      
   
    #tokenize text
    tokens = word_tokenize(text)
    #words = word_tokenize(text)    
    
    # stopword list 
    #stop_words = stopwords.words("english")
    #words = [w for w in words if w not in stopwords.words("english")]
    #lemmatize text
    lemmatizer = WordNetLemmatizer()

    #generate a list of clean tokens 
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    #stemmed = [PorterStemmer().stem(w) for w in words]
    #lemmatizing
    #clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]   
        
        
    return clean_tokens


def build_model():
    """
    Model construction
    
    """
    #create pipeline
    pipeline = Pipeline([                 
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),              
                ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])     
   
    #model parameters for GridSearchSV
    parameters = {'classifier__estimator__max_depth': [10, None],
              'classifier__estimator__min_samples_leaf':[2, 10],
              'tfidf__use_idf': (True, False)}

    # call gridsearch on above parameters. 
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=10)
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model and print f-score, precision and recall values
    
    Input:
        model -> Classifier Model
        X_test -> X test features
        Y_test -> Predicted Y test value
        category_name -> category names of Y column
    Output:
        Print average f-score, precision and recall
    """
    #predict values based on X_test
    y_pred = model.predict(X_test)
    
    #Create a dataframe to store category, f_score,precision and recall
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    
    #initialize index for the dataframe
    num = 0   

    #save the f_score,precision,recall per Y column in a dataframe
    for cat in category_names:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[cat], y_pred[:,num], average='weighted')
        results.loc[num] = [cat,f_score, precision, recall]    
        num = num + 1
    
    #print average scores for the model results
    print('Mean f_score:', results['f_score'].mean())
    print('Mean precision:', results['precision'].mean())
    print('Mean recall:', results['recall'].mean())


def save_model(model, model_filepath):
    """ 
    Save model's best_estimator_ in a pickle file
    
    Input:
        model -> Classifier Model
        model_filepath -> Path to save pickle file        
    Output:
        save the model's best estimator in pickle file
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()