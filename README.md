# UFC Fight Predictor

The aim of this project is to create an interactive application that uses a machine learning model to make predictions on UFC fights using fighter statistics. This application would serve as an entertainment tool to increase traffic to a UFC website/blog. The application can be found at https://ufc-fight-predictor.herokuapp.com.

## File Descriptions

- superseded: Folder containing old notebooks and .py files.
- ufc_scrape:
  - ufc_scrape:
    - __init__.py, items.py, middlewares.py, pipelines.py, settings.py: Default Scrapy files.
    - spiders:
      - fighters_spider.py: spider to scrape data on the fights of each fighter.
      - fights_spider.py: spider to scrape data on each UFC event.
- data:
  - data_fighters.py: scraped data of each fighter's fight details.
  - data_fights.py: scraped data of each UFC event.
- Classes.py: contains grid search classes to be usedin modelling.
- preprocessing.ipynb: notebook for initial exploration of scraped data and preprocessing for modelling.
- first_app.py: python file for creating the streamlit application.
- model.pkl: final model saved using pickle.
- scaler.pkl: StandardScaler transformation saved using pickle.
- Procfile: file for Heroku that configures services to be run in the dynpo.
- requirements.txt: file for Heroku that states the python library requirements for the application to be run.

## Methods Used

- Web scraping
- Data exploration
- Data visualisation
- Machine learning
- Cloud computing
- Front end development
- Deploying to web

## Technologies

- Python
- Scrapy
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- Keras
- Google Cloud Platform
- Streamlit
- Heroku

## Executive Summary

The UFC maintains a statistics website (www.ufcstats.com) that contains information on every fighter, fight and event that has occured in the organistation's history. To obtain the information needed for modelling, two scrapy spiders were made. One of these spiders scraped the detailed breakdown of each fight and one scraped the details of each UFC event. The website actually has a summary of each fighters career statistics, however this could not be used as it would not be reflective of the fighter's performance at the time of the fight. To prevent data leakage it was important to only use data of a fighter that was available prior to the fight occuring. For example, a fighter's current career statistics should not be used to train the model on the outcome of a fight that occured 5 fights ago, because those statistics would have been different at the time of the fight. This was one of the major hurdles of the project and it was why the detailed breakdown of each fight had to be scraped, so that a fighter's statistics at the point of each past fight could be calculated.

Once the data had been through a fair amount of work in pandas, I had each fighter's pre-fight statistics for each of their fights and the data was ready for modelling. Many models were created and their hyperparameters were tuned, the best performers from each model type are given below with their performance on the validation dataset:

<h5 align="center">Model Performances</h5>
<p align="center">
  <img src="https://github.com/ravimalde/ufc_fight_predictor/blob/master/images/model%20evaluation.png" title="Model Performances" width=500>
</p>

The stacking model comprised of a support vector machine, random forest and xgboost model was the best performer (see diagram below for model architecture). The most important features in the dataset are also given beneath.

<img src="https://github.com/ravimalde/ufc_fight_predictor/blob/master/images/feature_importance.png" width=850 align=middle>

The model achieved an accuracy of 0.65 on the test dataset. The nature of combat sports, particularly MMA, is that they are very unpredictable and upsets are frequent. The whole outcome of a fight can change in a fraction of a second so although this acuracy isn't impressive on paper, I'm fairly happy with the outcome and I'm confident that this could be improved upon in the future with more data.

To create the application I used Streamlit (www.streamlit.com). This tool allows for the creation of interactive machine learning applications in an extremely pythonic way. The layout of the application I designed allows the user to choose from any two fighters in the organisation and predict who the winner will be and display what the confidence is in that outcome occuring. It also presents all of the fighter statistics in matplotlib plots so that the user can easily see a visual representation of how the fighter's compare.

Lastly, the application was deployed to the web using Heroku. This is a cloud platform as a service 
