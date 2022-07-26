# NBA Playground 

## Data Source
The dataset used for this project can be found here: https://www.kaggle.com/datasets/nathanlauga/nba-games

## KAT_REGR.py
The purpose of this project was to take a hands-on learning approach to understanding sci-kit learn and its application in building Machine Learning models. To do this, I looked at NBA player data and tried to create an accurate prediction of points scored given a set of features. KAT_REGR.py takes in player data (In this case, I focused on Karl-Anthony Towns, but small changes will make it easy to select any player in the league) and uses various predictions models progressing from simple and multi-linear regression models to more advanced methods such as binary search and random forest regressors. This file is derived from a notebook where I performed all the exploratory data analysis and initial model testing. The purpose of rewriting KAT_ML.ipynb to be used on an IDE was to 1. Practice creating models as functions 2. Have the ability to step into my code and understand how the data is moving, as well as debug code that was breaking my model. 

## Feature Engineering
Once I understood how to use the SKLearn packages needed for my project, I started looking into how to improve these models. Despite a variety of algorithms for predicitng player points, none of them yielded accurate results based on performance statistics such as RSME, MAE, and r2. The first step in the Feature Engineering notebook was to understand the mutual information shared between the target (points) and the features I used in my baseline model. Results of the mutual information test showed that some of the features used in the baseline model were mutually indepenent from the target variable, and several features contained MI scores of less than 0.2. The first step in improving the model was to remove most of these features, but that meant I had to find new features with higher MI scores to replace them. One method I have been exploring is using more advanced statistics, such as effective field goal percentage and true shooting percentage, to more accurately capture a player's performance. Other ideas for further feature engineering include exploring opponent and team statistics to measure impact on an indiviudal player's scoring. 

## helpers.py
These are very basic functions, mostly used for manipulating datatypes, that package a few lines of code I found myself repetitively implementing in various places. 

## requirements
List of dependencies installed in a virtual environment.

