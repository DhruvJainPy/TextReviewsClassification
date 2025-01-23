# Report on Customer Sentiments Analysis for Marketing of Baby Products

Introduction: 
This Report focuses on the analysis of the dataset which contains features regarding reviews of Baby Products. Our goal is to develop a classification model using Natural Language Processing (NLP) for the prediction of customer sentiments based on the text reviews. 

Data Overview: 
Entries & Features: The dataset comprises of 60L instances and 10 features. 
Key Features: Rating, Title, Text, Images, ASIN, Parent ASIN, Helpful Votes, Verifies Purchases, etc. 
Target Variables: Our Main focus is to classify Rating in 3 categories namely (0-Negative, 1-Neutral, 2 – Positive).

Data Cleaning & Preprocessing: 
Rating ranges from 1-5 which have been categorised as follows:
1 – 2  -> Negative
3  -> Neutral
4 – 5  -> Positive
From the total of 60L instances a Sample of 1.2L is generated using Stratified Sampling. Missing values have been removed in data cleaning.

Exploratory Data Analysis (EDA): 
Distribution of Ratings haven been visualized using Pie Chart and Bar Chart. Diagrammatic Representation of Verified Purchases, Parent ASIN, Helpful Votes have also been shown.

Model Development: 
Models Used: Logistic Regression, Naïve Bayes Classifier, Random Forest Classifier and Voting Classifier. Data Split: The data was split into an 80-10-10 training, test and validation set.

Model Evaluation: 
Evaluation Metrics: Used Confusion Matrix, ROC Curve, AUC, Classification Report which contains Accuracy, Precision, Recall, Support and F1-score to evaluate model performance. Performance Comparison: Compared the performance of each model.

Model Interpretation: 
Logistic Regression & Naïve Bayes Classifier tends to overfit the data. On the other hand, Random Forest Classifier gave consistent results across Train and Test data. Voting Classifier also overfits the data but less compared to Logistic & Naïve Bayes.

Conclusions: 
Key Findings: Random Forest Classifier is the best performing model among the 3. 
Limitations: We have only considered text review for our analysis due to lack of knowledge to utilize Image Data as well using CNN. 
Future Work: Using Image data as well for more efficient model building by utilizing CNN.

