# Google Landmark Retrieval Challenge
Kaggle challenge to find similar landmark in a dataset.
  
## Initial Approach
My first approach was to use basic computer vision features using SIFT to group similar images using unsupervised learning. The problem is that the features are too simple. The grouped images are similar in a visual way (same objects, same colors ...) but they are not grouped by landmark.

## Similarity Method
Training data in the original challenge are not labelled. Thus I have used the "Google Landmark Recognition Challenge" which consists in classifing landmark with around 15000 categories. This way I have built a data base of similar data (two random images of the same label are considered as similar).

The idea is to build a metric of similarity to distinguish when two images represent the same landmark or not. Thus an embedding of each image is made and they are compared to find their proximity.
This way, I have build a Siamese Neural Network which takes two images as input and output a probability to contain the same landmark. During inference, the similarity between each image of the test dataset is computed. Similar images with a high probability are finaly selected to build the submission file.

## Computational problems
My training/testing are made on a 2011 Macbook Pro with 16GB RAM and an Intel Core i5.
It is not enough neither to train a deep CNN nor to manipulate the train/test datasets (million of images). Per example, the test dataset is composed of 100 000 images, building the submission file needs to compute similarity of each image with all the others (complexity = o(n^2)).
