# Quora Question Pairs Classifier 

A classifier to determine if two questions are duplicates of one another i.e. if they have the same intent. Uses a neural network model trained on a dataset of Quora question pairs. 

### Feature Engineering

The most basic features were how many characters/words are in each question, and the differences in these lengths. I also used a bag of words model to vectorize each question and added those to the feature vector. The [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) library was useful for adding edit-distance related features. 

### Neural Network

The neural net is currently implemented with a few relu hidden layers and a softmax output layer. It uses the crossentropy loss function and a gradient descent optimizer.

### Resources
* [Kaggle Challenge](https://www.kaggle.com/c/quora-question-pairs)
* [Dataset](https://www.kaggle.com/c/quora-question-pairs/download/test.csv.zip)