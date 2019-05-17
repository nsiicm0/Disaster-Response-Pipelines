# Disaster Response Pipeline Project

## About
This project was developed during the Udacity Data Science Nanodegree. It builds upon a predefined template.

## Pipeline

The project contains a pipeline which follows the following process:

![Process Pipeline][res/pipeline]

A dataset consisting of messages and their multi-class labels are ingested during the ETL process (realized within the data/process_data.py script).
The resulting data from the ETL process will then be stored within a sqlite database.

The data from the sqlite database will be used to train the machine learning model. Before being converted to train and test data, the dataset is tokenized and lemmatized in order to remove irrelevant words. The training uses a cross-validation model with grid search to optimize parameters. (realized within the model/train_classifier.py script)

The trained model will be used to predict messages, that a user can enter within the flask web app. The web app can be launched using the run.py script.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Structure

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

```