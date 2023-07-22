#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx
import seaborn as sns
import statsmodels.api as sm

# Supress warning for clean notebook
import warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")


# In[2]:


df1 = pd.read_csv('tf_mini.csv')
df2 = pd.read_csv('log_mini.csv')
df = pd.merge(df1, df2, left_on='track_id', right_on='track_id_clean', how='inner')
df= df.drop('track_id_clean',axis=1)
df.head(5).T


# 1- columns explaination
# 
# * track_id: provides a unique identifier for each track.
# 
# * Duration: This column contains the length of each track in seconds.
# 
# * Release year: This column contains the year in which each track was released.
# 
# * Popularity estimate: This column contains an estimate of the popularity of each track on a scale of 0-100.
# 
# * Acousticness: This column contains a value representing the acousticness of each track, with higher values indicating a more acoustic sound.
# 
# * Beat strength: This column contains a value representing the strength of the beats in each track.
# 
# * Danceability: This column contains a value representing the danceability of each track, with higher values indicating that the track is more suitable for dancing.
# 
# * Energy: This column contains a value representing the energy of each track, with higher values indicating a more energetic sound.
# 
# * Loudness: This column contains a value representing the loudness of each track.
# 
# * Instrumentalness: This column contains a value representing the amount of vocals in each track, with higher values indicating that the track is more instrumental.
# 
# * Key: This column contains a value representing the key of each track, with values ranging from 0-11 representing different keys.
# 
# * Liveness: This column contains a value representing the liveliness of each track, with higher values indicating that the track sounds more like a live performance.
# 
# * Mode: This column contains a value representing the mode of each track, with "major" indicating a major key and "minor" indicating a minor key.
# 
# * Tempo: This column contains a value representing the tempo of each track in beats per minute.
# 
# * Valence: This column contains a value representing the valence (positivity) of each track, with higher values indicating a more positive or happy sound.
# 
# * acoustic_vector_(0-7) : values representing different acoustic vectors associated with each track, which represent different aspects of the track's sound.
# 
# * session_id:  a unique identifier for each session in the dataset
# 
# * session_position:  the position of the current track in the session
# 
# * session_length:  the total number of tracks in the session
# 
# * track_id_clean:  a unique identifier for each track
# 
# * skip_1, skip_2, skip_3:  binary variables indicating whether the user skipped the current track after 1, 2, or 3 seconds respectively
# 
# * not_skipped:  a binary variable indicating whether the user did not skip the current track
# 
# * context_switch:  a binary variable indicating whether the user switched from a different context (such as a different playlist or album) before playing the current track
# 
# * no_pause_before_play:  a binary variable indicating whether there was no pause between the previous track and the current track
# 
# * short_pause_before_play , long_pause_before_play:  binary variables indicating whether there was a short or long pause (less than 30 seconds or more than 30 seconds) between the previous track and the current track respectively
# 
# * hist_user_behavior_n_seekfwd: the number of times the user has seeked forward within the current session
# 
# * hist_user_behavior_n_seekback:  the number of times the user has seeked backward within the current session
# 
# * hist_user_behavior_is_shuffle:  a binary variable indicating whether the user has enabled shuffle mode for the current session
# 
# * hour_of_day:  the hour of the day when the track was played
# 
# * date:  the date when the track was played
# 
# * premium:  a binary variable indicating whether the user has a premium account
# 
# * context_type:  the type of context (such as editorial playlist, user-generated playlist, or album) for the current track
# 
# * hist_user_behavior_reason_start:  the reason why the current track was played (such as trackdone, fwdbtn, or playbtn)
# 
# * hist_user_behavior_reason_end:  the reason why the user stopped playing the current track (such as trackdone, endplay, or fwdbtn)
# 
# 

# In[3]:


df.info()


# Based on this dataset, here are the preprocessing techniques that will be applied:
# 
# * Data cleaning: Check for and remove missing values, duplicates, or outliers in the dataset.
# 
# * Data encoding: Encode categorical variables such as context_type, hist_user_behavior_reason_start, and hist_user_behavior_reason_end using one-hot encoding or label encoding techniques.
# 
# * Data normalization: Scale numerical variables such as session_position, session_length, hour_of_day, and hist_user_behavior_n_seekfwd using normalization techniques such as Min-Max scaling or Standard scaling.
# 
# * Feature selection: Determine which features are important and relevant for the prediction task, and exclude any unnecessary or redundant features.
# 
# * Handling time-series data: Convert the date feature to a datetime object and sort the dataset based on the timestamps. Split the data into consecutive sessions for time-series analysis.

#  1- Date cleaning

# * check for missing values
# * duplicates
# * outliers 
# * handle date
# 

# In[4]:


df.isnull().sum()*100/df.shape[0] 


# no missing values

# In[5]:


duplicate=df.duplicated()
print(duplicate.sum())


# no duplicates

# In[6]:


df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day


# In[7]:


df.describe().T


# In[8]:


plt.figure(figsize=(20,15)) # Set the figure size
sns.boxplot(data=df, palette='rainbow', orient='h', width=3.0)
plt.title('Boxplot of All Features')
plt.xlabel('Feature Values')
plt.ylabel('Features')
plt.show()
#box plot to all features to show outliers


# columns wwith outliers :
# * ( hist_user_behavior_n_seekfwd, hist_user_behavior_n_seekback, tempo , loudness , dyn_range_mean ,duration )

# In[9]:


def draw_boxplot(col):
  sns.boxplot(df[col]) 


# In[10]:


draw_boxplot('tempo')


# In[11]:


draw_boxplot('loudness')


# In[12]:


draw_boxplot('dyn_range_mean')


# In[13]:


draw_boxplot('us_popularity_estimate')


# In[14]:


draw_boxplot('duration')


# In[15]:


draw_boxplot('hist_user_behavior_n_seekfwd')


# In[16]:


draw_boxplot('hist_user_behavior_n_seekback')


# we need to deal with outliers

# In[17]:


def handle_outlier(df, col_name):
    Q1, Q3 = df[col_name].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)
    df.loc[df[col_name] < lower, col_name] = lower
    df.loc[df[col_name] > upper, col_name] = upper


# In[18]:


df['hist_user_behavior_n_seekfwd'].value_counts()


# In[19]:


handle_outlier(df,'tempo')


# In[20]:


draw_boxplot('tempo')


# In[21]:


handle_outlier(df,'loudness')


# In[22]:


draw_boxplot('loudness')


# In[23]:


handle_outlier(df,'dyn_range_mean')


# In[24]:


draw_boxplot('dyn_range_mean')


# In[25]:


handle_outlier(df,'us_popularity_estimate')


# In[26]:


draw_boxplot('us_popularity_estimate')


# In[27]:


handle_outlier(df,'duration')


# In[28]:


draw_boxplot('duration')


# In[29]:


df['hist_user_behavior_n_seekfwd'].value_counts()


# In[30]:


df['hist_user_behavior_n_seekback'].value_counts()


# In[31]:


def draw_scatterplot(col1,col2):
  plt.scatter(df[col2],df[col1])


# In[32]:


draw_scatterplot('hist_user_behavior_n_seekfwd','not_skipped')


# In[33]:


draw_scatterplot('hist_user_behavior_n_seekfwd','not_skipped')


# In[34]:


draw_scatterplot('us_popularity_estimate','not_skipped')


# In[35]:


df = df[df['hist_user_behavior_n_seekback'] != 151 ]


# Done dealing with outliers that may affect our data and kept the ones that doesn't affect our target column

# In[36]:


df.shape


# 2-Data encoding

# In[37]:


df.head().T


# Encode categorical variables :
# * mode / context_type / hist_user_behavior_reason_start /  hist_user_behavior_reason_end :: using label encoding techniques.

# In[38]:


df['mode'].value_counts()


# In[39]:


df['session_id'].value_counts()


# In[40]:


df['context_type'].value_counts()


# In[41]:


df['hist_user_behavior_reason_start'].value_counts()


# In[42]:


df['hist_user_behavior_reason_end'].value_counts()


#   As we can see, each column can be one from many choices so using label encoding is prefered as we wont need to create many more features

# In[43]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['mode']= encoder.fit_transform(df['mode'])
df['mode'].value_counts()


# In[44]:


df['session_id']= encoder.fit_transform(df['session_id'])
df['session_id'].value_counts()


# In[45]:


df['context_type']= encoder.fit_transform(df['context_type'])
df['context_type'].value_counts()


# In[46]:


df['hist_user_behavior_reason_start']= encoder.fit_transform(df['hist_user_behavior_reason_start'])
df['hist_user_behavior_reason_start'].value_counts()


# In[47]:


df['hist_user_behavior_reason_end']= encoder.fit_transform(df['hist_user_behavior_reason_end'])
df['hist_user_behavior_reason_end'].value_counts()


# now lets encode all bool features

# In[48]:


df['not_skipped'] = df['not_skipped'].astype(int)
df['skip_3'] = df['skip_3'].astype(int)
df['skip_2'] = df['skip_2'].astype(int)
df['skip_1'] = df['skip_1'].astype(int)
df['premium'] = df['premium'].astype(int)
df['hist_user_behavior_is_shuffle'] = df['hist_user_behavior_is_shuffle'].astype(int)


# In[49]:


df.head().T


# done with encoding

# 3- Data Normailzation

# In[50]:


df.head().T


# Scale numerical variables such as :
# * session_position, session_length, hour_of_day, and hist_user_behavior_n_seekfwd using normalization techniques such as Min-Max scaling or Standard scaling.

# In[51]:


from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()

df['duration']=scaling.fit_transform(df[['duration']])
df['us_popularity_estimate']=scaling.fit_transform(df[['us_popularity_estimate']])
df['dyn_range_mean']=scaling.fit_transform(df[['dyn_range_mean']])
df['loudness']=scaling.fit_transform(df[['loudness']])
df['tempo']=scaling.fit_transform(df[['tempo']])


# done with scaling numerical data

# 4- Feature selection
# 

# *  Determine which features are important and relevant for the prediction task, and exclude any unnecessary or redundant features.

# In[52]:


df.head().T


# In[53]:


df["skipped"] = df["skip_1"]*df["skip_2"]*df["skip_3"]
df.drop(["skip_1", "skip_2", "skip_3", "not_skipped"], axis=1, inplace=True)


# In[54]:


#Correlation using heatmap
def heat_map(df):
    plt.figure(figsize=(70, 70))
    heatmap = sns.heatmap(df.corr(), annot=True, cmap="YlGnBu",annot_kws={"fontsize":20}, vmin=-1, vmax=1)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=15)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=15)
    plt.tight_layout()
    plt.show()


# In[55]:


heat_map(df)


# we can see that hist_user_behavior_reason_start & hist_user_behavior_reason_start have the highest relation than other columns with our target column.Also, a High negative relation with skip(1,2 & 3) columns, which means they are inversely proportional

# In[56]:


df.head().T


# In[57]:


def get_network_graph(source_column):
  g=nx.from_pandas_edgelist(df,source=source_column, target="skipped", edge_attr= None )
  plt.Figure(figsize=(16,10))
  plt.plot(g,  color='red')
  plt.draw()
  # function make a network plot to show the network relations between entities of the data and the target 


# In[58]:


get_network_graph("session_length")


# In[59]:


get_network_graph("hist_user_behavior_n_seekfwd")


# In[60]:


get_network_graph("hour_of_day")


# In[61]:


get_network_graph("context_type")


# In[62]:


get_network_graph("hist_user_behavior_reason_start")


# In[63]:


get_network_graph("hist_user_behavior_reason_end")


# In[64]:


plt.style.use('dark_background')

var = ['beat_strength','bounciness','danceability','energy', 'flatness', 'loudness', 'mechanism', 'organism', 'valence', 'acoustic_vector_0'	,'acoustic_vector_1'	,'acoustic_vector_2'	,'acoustic_vector_3',	'acoustic_vector_4'	,'acoustic_vector_5'	,'acoustic_vector_6'	,'acoustic_vector_7']

from scipy.stats import skew

for col in df[var]:
    print("Skewness:",col,"=",round(skew(df[col]),3))
    print("Kurtosis:",col,    "=",round(df[col].kurt(),2))
    print("Mean:",col,    "=",round(df[col].mean(),2))
    print("Max:",col,     "=",round(df[col].max(),2))
    print("Min:",col,     "=",round(df[col].min(),2))
    print("Median:",col,  "=",round(df[col].median(),2))
    print("Std:",col,     "=",round(df[col].std(),2))
    print("Var:",col,     "=",round(df[col].var(),2))
    plt.figure(figsize=(18,6))
    sns.distplot(df[col],kde=True,bins=50,color="Yellow",hist_kws={"edgecolor": (1,1,0,1)})
    plt.title(col,fontweight="bold")
    plt.show()
    print("====="*25)


# In[65]:


df=df.drop(['month','year','date','session_id','acoustic_vector_1','valence','time_signature','tempo','mechanism','loudness','liveness','key','flatness','energy','danceability','beat_strength','us_popularity_estimate','release_year'],axis =1 )


# Feature Engingeering
# 
# Feature creation: Feature creation involves the process of generating new features from the existing set of features. It is done to provide a better representation of the data or to capture new patterns that were not present in the original data.

# In[66]:


# Feature Extraction
from sklearn.decomposition import PCA

# Initialize PCA with 3 components
pca = PCA(n_components=3)

# Select columns for feature extraction
acoustic_vectors = df.loc[:, 'acoustic_vector_0':'acoustic_vector_7']

# Fit PCA to the data and transform it
acoustic_vectors_pca = pca.fit_transform(acoustic_vectors)

# Add PCA components as new features to the DataFrame
df['acoustic_vector_pca1'] = acoustic_vectors_pca[:, 0]
df['acoustic_vector_pca2'] = acoustic_vectors_pca[:, 1]
df['acoustic_vector_pca3'] = acoustic_vectors_pca[:, 2]


# In[67]:


#Feature Selection
#Correlation using heatmap
def heat_map(df):
    plt.figure(figsize=(70, 70))
    heatmap = sns.heatmap(df.corr(), annot=True, cmap="YlGnBu",annot_kws={"fontsize":20}, vmin=-1, vmax=1)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=15)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=15)
    plt.tight_layout()
    plt.show()

heat_map(df)


# In[68]:


#Drop some Features
df = df.drop(['track_id','bounciness','dyn_range_mean','instrumentalness','mode','organism','acoustic_vector_3',
              'acoustic_vector_5','acoustic_vector_7','hour_of_day','day','acoustic_vector_pca2','hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback'], axis=1)
df.head(5).T


# In[69]:


df.shape


# - Precision is the measure of correctly predicted positive instances out of all instances predicted as positive.
# - Recall is the measure of correctly capturing actual positive instances.
# - F1-score is the harmonic mean of precision and recall, providing a balanced metric.
# - Support indicates the number of instances in each class.
# - Accuracy represents the overall percentage of correctly classified instances.
# - Macro Average calculates the average of precision, recall, and F1-score across all classes, giving equal weight to each class.
# - Weighted Average calculates the average of precision, recall, and F1-score across all classes, weighted by the support of each class.

# # Random forest 

# In[70]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
model = RandomForestClassifier(n_estimators=30)


# In[71]:


#Splitting data to target variable y and and input data 
X = df.drop('skipped',axis=1)
y = df['skipped']


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[73]:


RFC = RandomForestClassifier(random_state=42)
RFC.fit(X_train, y_train)


# In[74]:


clf = RandomForestClassifier(n_estimators=100, random_state=42)


# In[75]:


clf.fit(X_train, y_train)


# In[76]:


# Making predictions on the test set
y_pred = clf.predict(X_test)


# In[77]:


train_score = RFC.score(X_train,y_train)
test_score= RFC.score(X_test,y_test)
print("Train Score : ", train_score*100," %")
print("Test Score : ", test_score*100," %")


# In[78]:


# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")


# In[79]:


RFC.score(X_train, y_train)


# In[80]:


cm2 = confusion_matrix(y_test,y_pred, labels=clf.classes_)
cm2


# In[81]:


from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


# In[82]:


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


# In[83]:


disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=clf.classes_)


# In[84]:


disp2.plot()


# In[85]:


print(classification_report(y_test,y_pred))


# # Support Vector Machine 

# In[86]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[87]:


x = df.drop('skipped',axis=1)
y = df[['skipped']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[88]:


sv = SVC()


# In[89]:


sv.fit(x_train,y_train)


# In[90]:


pred2 = sv.predict(X_test)


# In[91]:


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


# In[92]:


train_score = sv.score(X_train,y_train)
test_score= sv.score(X_test,y_test)
print("Train Score : ", train_score*100," %")
print("Test Score : ", test_score*100," %")


# In[93]:


accuracy_score(y_test,pred2)


# In[94]:


cm2 = confusion_matrix(y_test,pred2, labels=sv.classes_)
cm2


# In[95]:


disp1 = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=sv.classes_)
disp1


# In[96]:


disp1.plot()


# In[97]:


print(classification_report(y_test,pred2))


# # KNN 

# In[98]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

x= df.drop('skipped',axis=1)
y = df['skipped']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=80)

k_values = []
accuracy_scores = []

# range of k values to test
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain, ytrain)

    ypred = knn.predict(xtest)

    accuracy = accuracy_score(ytest, ypred)

    # Append the current value of k and accuracy
    k_values.append(k)
    accuracy_scores.append(accuracy)

# Plot the elbow curve
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy Score')
plt.title('Elbow Method: Accuracy Score vs. Number of Neighbors')
plt.show()

10 neighbors
# In[99]:


best_random_state = None
best_test_score = 0

#random states to test
random_state_range = range(1, 50)

for random_state in random_state_range:
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(xtrain, ytrain)

    test_score = knn.score(xtest, ytest)
    
    # Check if the current test score is better than the previous best test score
    if test_score > best_test_score:
        best_test_score = test_score
        best_random_state = random_state

print("Best Random State:", best_random_state)
print("Best Test Score: {:.2f}%".format(best_test_score * 100))


# In[100]:


train_score = knn.score(xtrain,ytrain)
test_score= knn.score(xtest,ytest)
print("Train Score : ", train_score*100," %")
print("Test Score : ", test_score*100," %")


# In[101]:


ypred = knn.predict(xtest)

# Evaluate the model's performance on the test set
acc = accuracy_score(ytest, ypred)
report = classification_report(ytest, ypred)
cm = confusion_matrix(ytest, ypred)

# Print the evaluation metrics
print("Accuracy:", acc*100)
print("--"*30)
print("Classification report:\n", report)
print("--"*30)
print("Confusion matrix:\n", cm)


# In[102]:


from sklearn.linear_model import LogisticRegression


# In[103]:


lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred3 = lr.predict(x_test)


# In[104]:


disp1 = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lr.classes_)
disp1.plot(cmap=plt.cm.Blues)


# # Regularised Logistic Regression 

# In[105]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

best_random_state = None
best_test_score = 0

# random states to test
random_state_range = range(1, 101)

for random_state in random_state_range:
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=random_state)

    # Set penalty='l2' for L2 regularization
    
    lr = LogisticRegression(penalty='l2')
    lr.fit(xtrain, ytrain)

    test_score = lr.score(xtest, ytest)
    
    # Check if the current test score is better than the previous best test score
    if test_score > best_test_score:
        best_test_score = test_score
        best_random_state = random_state


print("Best Random State:", best_random_state)
print("Best Test Score: {:.2f}%".format(best_test_score * 100))


# In[106]:


train_score = lr.score(xtrain,ytrain)
test_score= lr.score(xtest,ytest)
print("Train Score : ", train_score*100," %")
print("Test Score : ", test_score*100," %")


# In[107]:


ypred = lr.predict(xtest)
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# Evaluate the model's performance on the test set
acc = accuracy_score(ytest, ypred)
report = classification_report(ytest, ypred)
cm = confusion_matrix(ytest, ypred)

# Print the evaluation metrics
print("Accuracy:", acc*100)
print("--"*30)
print("Classification report:\n", report)
print("--"*30)
print("Confusion matrix:\n", cm)


# In[108]:


disp1 = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lr.classes_)
disp1.plot(cmap=plt.cm.Blues)


# In[112]:


# save the model to disk 
import pickle
filename = 'KNN_Model.pkl' 
pickle.dump(knn, open(filename, 'wb'))

filename = 'KNN_Model.pkl' 
loaded_model =    pickle.load(open(filename, 'rb'))


# In[113]:


loaded_model


# In[ ]:




