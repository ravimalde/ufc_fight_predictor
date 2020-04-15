# UFC Fight Predictor

The aim of this project was to create an interactive application that uses a machine learning model to make predictions on UFC fights using fighter statistics. This application would serve as an entertainment tool to increase traffic to a UFC website/blog. The application can be found at https://ufc-fight-predictor.herokuapp.com.

- Email: ravidmalde@gmail.com
- LinkedIn: www.linkedin.com/in/ravi-malde
- Medium: www.medium.com/@ravimalde

## Table of Contents

1. [ File Descriptions ](#file_description)
2. [ Methods Used ](#methods_used)
3. [ Technologies Used ](#technologies_used)
4. [ Executive Summary ](#executive_summary)
  * [ Web Scraping ](#web_scraping)
  * [ Data Cleaning and Feature Engineering ](#cleaning)
  * [ Modelling ](#modelling)
  * [ Developing Application ](#developing_application)
  * [ Deploying Application ](#deploying_application)
  * [ Limitations ](#limitations)
  * [ Future Work ](#future_work)

<a name="file_description"></a>
## File Descriptions

- preprocessing.ipynb: notebook for initial exploration of scraped data and preprocessing for modelling.
- modelling.ipynb: notebook where the models were made, tuned and evaluated.
- Classes.py: contains classes for hyperparameter tuning in modelling notebook.
- first_app.py: python file for creating the streamlit application.
- superseded: Folder containing old notebooks and .py files.
- ufc_scrape:
  - ufc_scrape:
    - items.py, middlewares.py, pipelines.py, settings.py, init.py: default Scrapy files.
    - spiders:
      - fighters_spider.py: spider to scrape data on the fights of each fighter.
      - fights_spider.py: spider to scrape data on each UFC event.
- data:
  - data_fighters.csv: scraped data of each fighter's fight details.
  - data_fights.csv: scraped data of each UFC event.
  - data_for_application.csv: data to be used in application.
  - data_cleaned: cleaned data for modelling.
- model.pkl: final model saved using pickle.
- scaler.pkl: RobustScaler transformation saved using pickle.
- Procfile: file for Heroku that configures services to be run in the dynpo.
- requirements.txt: file for Heroku that states the python library requirements for the application to be run.

<a name="methods_used"></a>
## Methods Used

- Web scraping
- Data exploration, visualisation and cleaning
- Feature engineering
- Machine learning
- Cloud computing
- Front end development
- Deploying to web

<a name="technologies_used"></a>
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

<a name="executive_summary"></a>
## Executive Summary

As stated at the beginning of this document, the aim of the project was to create a web application where users could choose two fighters and use a machine learning model to predict the outcome of the fight. It's purpose is primarly as an entertainment tool that could be placed on a UFC blog/website to increase traffic. Despite it being an entertainment tool, accuracy was still regarded as important because making correct predictions would increase the likelihood of it being shared amongst social circles. It was also designed to be interactive and give the user as much information as possible about the fighters.

<a name="web_scraping"></a>
### Web Scraping

The UFC maintains a [statistics website](www.ufcstats.com) that contains information on every fighter, fight and event that has occured in the organistation's history. To extract this data for modelling, two scrapy spiders were created:

- **fighter_spider** : Scraped the detailed breakdown of each fighter's bouts (such as the number of punches thrown by each fighter) and their measurements such as height and reach. Examples of the pages scraped can be found [here](http://www.ufcstats.com/fight-details/b46f2f007b622bce) and [here](http://www.ufcstats.com/fighter-details/1338e2c7480bdf9e).
- **fight_spider**: Scraped the details of each ufc event. This was necessary because the page that the fighter_spider scrape had no information on the date of the bout, which was necessary to calculate the age of each fighter at the time of the fight. An example of the pages scraped can be found [here](http://www.ufcstats.com/event-details/53278852bcd91e11).

Information was obtained on a total of 5535 fights. The distribution of these fights over the years can be seen below:

<h5 align="center">Number of Fights in the UFC Each Year</h5>
<p align="center">
  <img src="https://github.com/ravimalde/ufc_fight_predictor/blob/master/images/number_of_fights.png" width=850>
</p>

<a name="cleaning"></a>
### Data Cleaning and Feature Engineering

A standard data cleaning procedure was followed where the data was formatted into the correct data types and then missing values were handled appropriately, usually filling in the value with a related column (e.g. height and reach) or filling in with the median of the weightclass group.

A major effort in the project was to prevent data leakage. It was important to only use data of a fighter that was available prior to the fight occuring. For example, a fighter's current career statistics should not be used to train the model on the outcome of a fight that occured 5 fights ago, because those statistics would have been different at the time of the fight. Therefore, for each past fight, the fighters' pre-fight statistics were calculated using the detailed breakdown of their previous bouts. Additionally to this, only the fighters' 5 previous fights were used to caluclate their pre-fight statistics. This decision was made to do this because over a fighter's career, their performance and style of fighting can change so only their recent performances should be considered.

The features fed into the model were as follows:

- Height (m)
- Weight (lbs)
- Reach (m)
- Age (years)
- Fighting Stance
- Strikes Landed (per minute)
- Striking Accuracy (%)
- Strikes Absorbed (per minute)
- Striking Defence (%)
- Takedown Average (per 15 minutes)
- Takedown Accuracy (%)
- Takedown Defence (%)
- Submissions Average (per 15 mins)

<a name="modelling"></a>
### Modelling

The data was split into three sub datasets; training (2828 instances), validation (708 instances) and test (400 instances) datasets. Unfortunately the test dataset is smaller than what I would have liked, but due to the small size of the total dataset, the decision was taken to dedicate more of the data to training in hopes of increasing real world performance. Many classification models were created and their hyperparameters were tuned using GridSearchCV to cross validate their performance. The best performers from each model type are presented below (sorted by performance on the validation dataset):

<h5 align="center">Model Performances</h5>
<p align="center">
  <img src="https://github.com/ravimalde/ufc_fight_predictor/blob/master/images/model_evalutaion.png" width=500>
</p>

The stacking model comprised of a support vector machine, random forest and voting classifier (random forest + svm) model was the best performer (see diagram below for model architecture). This came as a surprise because stacking models often perform best when configured with very different models, so that a model's weak performance in one area can be picked up by another model's better performance in that region of the dataset. Nevertheless, the numbers don't lie, and the best performing model was formed of models that had similarities. 

<h5 align="center">Stacking Model Architecture</h5>
<p align="center">
  <img src="https://github.com/ravimalde/ufc_fight_predictor/blob/master/images/stacking_architecture.png" width=700 align=middle>
</p>

An interesting insight from the model is the relative feature importances. These were produced using the random forest model as neither the overall stacking model nor the SVM and voting classifier models have the ability to produce feature importances. It is assumed that the feature importance of the stacking model is of a similar distribution to the random forest model. It appears age, win percentage, strikes landed per minute, takedown average and strikes absorbed per minute are the five best predictors of a fight.

<h5 align="center">Relative Feature Importance</h5>
<p align="center">
  <img src="https://github.com/ravimalde/ufc_fight_predictor/blob/master/images/feature_importance.png" width=850 align=middle>
</p>

**The model achieved an accuracy of 0.64 on the test dataset**. The nature of combat sports, particularly MMA, is that they are very unpredictable and upsets are frequent (in fact this is in part why I believe the sport is gaining huge popularity); so although this acuracy isn't impressive on paper, I'm happy with the outcome and confident that the performance could be improved upon in the future as more data becomes available.

<a name="developing_application"></a>
### Developing Application

To create the application I used a realtively new framework called [Streamlit](www.streamlit.com). This tool allows for the creation of interactive machine learning applications in an extremely pythonic way. I designed the application to allow the user to choose from any two fighters in the organisation and predict who the winner will be while displaying what the confidence is in that outcome occuring. It also presents all of the fighter statistics in matplotlib visualisations so that the user can easily see a visual representation of how the fighter's compare.

<a name="deploying_application"></a>
### Deploying Application

Lastly, the application was deployed to the web using [Heroku](www.heroku.com). This is a cloud platform as a service that allows developers to deploy and scale applicatons. The process was fairly straightforward; simply create an account with them, make a Procfile which is automatically detected by Heroku and serves to outline the commands that need to be run in order to launch the application, and then create a requirements.txt file to tell Heroku which python dependencies are necessary for the application to function. The final step is then to git push the entire github repository to the Heroku branch. Voila, the application is now live and can be accessed by anyone around the world.

<a name="limitations"></a>
### Limitations

1. The current model is not 'symmetric'. By this I mean that the weights for fighter_x and fighter_y features are not equal. This can be seen in the feature importance plot above; for example, age_y is more important than age_x. This mean that the model has some bias. In testing, the model actually favours fighter y winning. This is not due to fighter_y winning more often in the dataset as this was checked to be a 50/50 split between fighter x and fighter y winning. The origin of this bias is still unknown, however a workaround was implemented. The application predicts with the model twice, swapping the order of the fighters input into the model in the second iteration. The model then presents to the user the most confident prediction of the two, thus ensuring that the same result is given each time, no matter what order the fighter's are input.

2. The dataset is fairly small at only 3936 instances in the cleaned dataset. This impedes the ability for the models to learn from the training set. It also increases the role that chance pays in any results because the validation and test dataset are also smaller than desired. Unfortunatetly at present there is no workaround for this issue.

<a name="future_work"></a>
### Future Work

1. The primary bulk of future work will be aimed at fixing the model symmetry. One possible way to do this would be to duplicate the training dataset, switch the positions of the x and y fighters, and then concatenate it with the original dataset. This may result in the difference of weights for the x and y features being much smaller and thus reduce the bias. If not, then further investigation into this issue is necessary.

2. The model will be periodically retrained as more data is available. Over the last few years the UFC has held approximately 500 fights per year and this figure is likely to rise as MMA is growing in popularity. Therefore in a few years time the dataset could be significantly larger, likely resulting in better model performance.
