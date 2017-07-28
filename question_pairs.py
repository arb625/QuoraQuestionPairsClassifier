import csv
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

def load_question_pairs_data():
  print("loading data")
  file_path = "question-pairs-dataset/questions.csv"
  with open(file_path, 'r', encoding="utf8") as csvfile:

    qids = []
    question_pairs = []
    labels = []

    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
      qid1 = row["qid1"]
      qid2 = row["qid2"]
      qids.append((qid1, qid2))

      question1 = row["question1"]
      question2 = row["question2"]
      question_pairs.append((question1, question2))

      label = int(row["is_duplicate"])
      one_hot_encoded_label = [0,0]
      one_hot_encoded_label[label] = 1
      labels.append(one_hot_encoded_label)

  featurized_questions = featurize_questions(question_pairs[:50000])

  training_percent, test_percent = .8, .2
  training_amount, test_amount = math.floor(len(featurized_questions)*training_percent), math.floor(len(featurized_questions)*test_percent) 

  training_set = np.asarray(featurized_questions[:training_amount])
  training_labels =  np.asarray(labels[:training_amount])
  test_set =  np.asarray(featurized_questions[training_amount: training_amount + test_amount])
  test_labels =  np.asarray(labels[training_amount: training_amount + test_amount])

  print(np.shape(training_set))
  print(np.shape(training_labels))
  print(np.shape(test_set))
  print(np.shape(test_labels))

  print("done loading data")
  return training_set, training_labels, test_set, test_labels


def featurize_questions(question_pairs):
  print("featurizing data")
  featurized_questions = []

  vectorizer = TfidfVectorizer()

  all_questions = [question for question_pair in question_pairs for question in question_pair]
  vectorizer.fit(all_questions)
  
  for i in range(len(question_pairs)):
    if i % 1000 == 0:
      print(i, flush=True)
    question1, question2 = question_pairs[i]
    feature_vector = []

    # basic features
    feature_vector.append(len(question1))
    feature_vector.append(len(question2))
    feature_vector.append(len(question1) - len(question2))
    feature_vector.append(len(question1.split(' ')))
    feature_vector.append(len(question2.split(' ')))
    feature_vector.append(len(set(question1.split(' ')) & set(question2.split(' '))))

    # bag of words (weighted)
    bag_of_words = vectorizer.transform([question1, question2]).toarray()
    feature_vector.extend(bag_of_words[0])
    feature_vector.extend(bag_of_words[1])

    # edit distance based features
    feature_vector.append(fuzz.ratio(question1, question2))
    feature_vector.append(fuzz.partial_ratio(question1, question2))
    feature_vector.append(fuzz.token_sort_ratio(question1, question2))
    feature_vector.append(fuzz.token_set_ratio(question1, question2))

    featurized_questions.append(feature_vector)

  print("done featurizing data")

  return featurized_questions

def train_neural_net(training_set, training_labels, test_set, test_labels):
  print("training neural net")

  n,d = np.shape(training_set)
  sess = tf.Session()
  K.set_session(sess)

  X = tf.placeholder(tf.float32, shape=(None, d))
  h1 = Dense(128, activation='relu')(X)
  h2 = Dense(128, activation='relu')(h1)
  preds = Dense(2, activation='softmax')(h2)

  labels = tf.placeholder(tf.float32, shape=(None, 2))
  loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Initialize all variables
  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  with sess.as_default():
    num_epochs = 5
    batch_size = 50
    for epoch in range(num_epochs):
      print("epoch: " + str(epoch), flush=True)
      for i in range(n, batch_size):
        print(i, flush=True)
        batch = training_set[i:i+batch_size]
        batch_labels = training_labels[i:i+batch_size]
        train_step.run(feed_dict={X: batch, labels: batch_labels})

  print("done training neural net")

  print("predicting and evaluating on test set")
  
  acc_value = accuracy(labels, preds)
  print(np.shape(test_labels))
  print(test_labels)
  with sess.as_default():
    print(np.shape(preds.eval(feed_dict={X: test_set, labels: test_labels})))
    print(preds.eval(feed_dict={X: test_set, labels: test_labels}))
    print(np.shape(acc_value.eval(feed_dict={X: test_set, labels: test_labels})))
    print(np.mean(acc_value.eval(feed_dict={X: test_set, labels: test_labels})))

  print("done predicting and evaluating on test set")

def question_pairs():
  training_set, training_labels, test_set, test_labels = load_question_pairs_data()
  train_neural_net(training_set, training_labels, test_set, test_labels)
  # predict_and_eval_on_test(test_set, test_labels)

if __name__ == "__main__":
  question_pairs()

