import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
from tools.helper import tokenize, compute_text_length
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report

def load_data(database_filepath):
    """Loads a SQLite DB and returns the messages which were previously processed using process_data.py

    Args:
        database_filepath: Path to the sqlite db.

    Returns:
        X and Y of the dataset, plus labels of Y

    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('{}'.format(database_filepath[database_filepath.rindex('/')+1:].replace('.db','')), engine)
    X = df.message
    Y = df.drop(['id','message','original','genre'], axis=1)
    return X, Y, list(Y)

def build_model():
    """Builds the model.

    Args:
        n/a

    Returns:
        Model to be used.

    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('length', Pipeline([
                ('count', FunctionTransformer(compute_text_length, validate=False))
            ]))]
        )),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'features__text__vect__ngram_range':[(1,2),(2,2)],
        'clf__estimator__n_estimators':[50, 100]
    }
    return GridSearchCV(pipeline, parameters, cv=2)

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model and prints a classification report.

    Args:
        model: The model to be evaluated.
        X_test: Test data.
        Y_test: Test data labels.
        category_names: The target names for the classification report.

    Returns:
        n/a

    """
    Y_pred = model.predict(X_test)
    for i in range(0, len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))
    #print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), target_names=category_names))


def save_model(model, model_filepath):
    """Save the model to disk.

    Args:
        model: The model to be exported.
        model_filepath: The destination for the export.

    Returns:
        n/a

    """
    pickle.dump(model, open('{}'.format(model_filepath), 'wb'))


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