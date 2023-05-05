![image](https://user-images.githubusercontent.com/112208238/236426436-ada473b5-b104-43ab-bcde-8469e6fad59f.png)

# IMDB Movie Ratings Sentiment Analysis

This is a Kaggle project that involves sentiment analysis on a large dataset of [movie reviews from IMDB](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis) to predict whether a given review is positive or negative.

## Overview

  * The IMDB Movie Rating Sentiment Analysis challenge on Kaggle involves building a machine learning model to predict the sentiment (positive or negative) of movie reviews from the IMDB dataset. The task is to use the provided training data to develop a model that can accurately classify the sentiment of movie reviews in the testing data.
  * My approach here was to compare three different models to run on the dataset so that I can evaluate the accuracy of the models and then find the best one from the set models.
  * My best model here was the Logistic Regression Classifier that gave me a 88% accuracy and it can be best used to analyze the sentiment of the given dataset.

### Data

  * Type: CSV file.
  * Size: The dataset contained 50,000 data points.
  * Instances (Train, Test, Validation Split): There are 50,000 data points in this dataset which were divided into 80-20 split for training and testing with none for validation.

#### Preprocessing / Clean up

* The data cleaning process incorporates multiple techniques, including lowercasing, URL removal, punctuation removal, number removal, tokenization, stop word removal, and stemming. These techniques aim to convert the text to lowercase, eliminate URLs, punctuation, and numbers, split text into individual words, remove frequently used stop words, and reduce words to their root form, respectively.

#### Data Visualization

![Screenshot (5)](https://user-images.githubusercontent.com/112208238/236421519-7fe4a946-dcf7-4d99-bb2d-40366f70fd87.png)

As we can see in this above barplot the number of positive and negative reviews are equally distributed which tells me that the dataset is balanced.

![Screenshot (6)](https://user-images.githubusercontent.com/112208238/236422645-f2c69906-8080-4f77-b8de-ce5d70fee8d7.png)
![Screenshot (7)](https://user-images.githubusercontent.com/112208238/236422651-6e3630f9-3b69-49d5-a9da-68073674a807.png)

These images above show us the wordcloud created with the positive and negative words most frequently used in the dataset with positive and negative sentiments respectivey.

### Problem Formulation

* I used 3 different models to measure the accuracy and various different aspects. The models I used were:
    * Logistic Regression.
    * RandomForestClassifier.
    * XGBoost.

### Training

* The training was quick and easy because there weren't many instances to test and train upon.

### Performance Comparison

![Screenshot (8)](https://user-images.githubusercontent.com/112208238/236424463-6073acb3-5751-4081-9bb1-963e81f81614.png)

The table above shows us the comparison of the models and the best model to use for this particular sentiment analysis.

### Conclusions

*  It can be concluded that sentiment analysis can be effectively performed on movie reviews using machine learning algorithms. In this project, a dataset of movie reviews was used to train and test several models including logistic regression, xgboost and random forests. The performance of these models was evaluated using metrics such as accuracy, precision, recall, and F1 score. The results showed that the logistic regression model performed the best with an accuracy of 88%. It was also observed that the use of feature engineering techniques such as n-grams and TF-IDF can significantly improve the performance of the models. Overall, the project demonstrates the usefulness of sentiment analysis in analyzing large volumes of text data such as movie reviews and provides insights into the effectiveness of different machine learning algorithms for this task.

### Future Work

* Apart from these models that I used I would like to test the dataset upon various other model sand build one myself to test it on real time data.

### Overview of files in repository

* There are only 2 files in this repository, one being the README file and the other the IMDB Review project code.ipynb which contains the code for the challenge.

### Data

* The dataset can be downloaded from this [Kaggle Challenge](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis).
