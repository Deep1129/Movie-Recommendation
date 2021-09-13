# Movie-Recommendation

![Image1](https://github.com/Deep1129/Movie-Recommendation/blob/ab2db9ab44a1f64df1e0dff1dcf01d09ce98b2ac/netflix.jpeg)

## Problem Statement
Given the user-movie rating data, we have to recommend new movies to the users.

We can see the above problem as various problems:
* Finding ratings a user give to new movie.
* Finding similar users and similar movies.

Therefore, to solve the problem in hand we will try to first find similar users and similar movies and then will try to predict the ratings a user will give to the movies he/she hasn't rated yet.

## Data

Source : https://www.kaggle.com/netflix-inc/netflix-prize-data/data

Data files :

* combined_data_1.txt
* combined_data_2.txt
* combined_data_3.txt
* combined_data_4.txt
* movie_titles.csv
  
The first line of each file contains the movie id followed by a colon. Each subsequent line in the file corresponds to a rating from a customer and its date in the following format:

CustomerID,Rating,Date

NOTE: The dataset given is too large (480189 users and 17770 movies), processing this data and applying ML model on this will require high computational resources and time. Therefore, I have used subset of this data.

## Some EDA results from the notebook

![image2](https://github.com/Deep1129/Movie-Recommendation/blob/ab2db9ab44a1f64df1e0dff1dcf01d09ce98b2ac/rating_distribution.png)  ![image3](https://github.com/Deep1129/Movie-Recommendation/blob/ab2db9ab44a1f64df1e0dff1dcf01d09ce98b2ac/ratings_per_movie.png)  

## ML models used

### * xgboost
### * Basline model using surprise library
### * kNN model using surprise library
### * SVD matrix factorization 

For using xgboost as first model, top 5 similar movies ratings, top 5 similar movies ratings, global average, movie average rating, user average ratings were used as initial 13 feature to predict the required rating.

Then, each model is used separately and later output of other models were used as features along with the previously had features and again xgboost was used.

## Results

Following shows the rmse values obtained using various models:

* knn_bsl_u :      1.0726493739667242  // kNN baseline using user user similarity

* svd       :      1.0726600049083366  // SVD

* knn_bsl_m  :      1.072758832653683
// kNN baseline using user user similarity

* bsl_algo    :    1.0730330260516174 
// Baseline 

* all_features :    1.074563836885384
// Using all features ( initial 13 features + all model output)

* all_models    :  1.0753876364207704
// Using models output ( initial 13 features not included)

* xgb_bsl        :  1.077191541044971
// 13 features + baseline

* first_algo     : 1.0894551558842125
// Initial 13 features

* xgb_knn_bsl     :1.0936628441242844 
// Initial 13 features + kNN baseline 
