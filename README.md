# google_landmark_retrieval
Google Landmark Retrieval Challenge
Kaggle challenge to find similar landmark in a dataset.

Notes:
 - Training data are not labelled. It could be interesting to use "Google Landmark Recognition Challenge" to train a model with the categories and use it in inference over the test set. (There are probably image in common between the two challenges but they don't have the same ids and urls)

Step 1:
Unsupervised learning directly on the test dataset to find similar landmark using sift features and KNN.
Step 2:
Supervised learning using labels from the Google Landmark Recognition Challenge.
