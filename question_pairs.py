# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import csv, math, sys
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from fuzzywuzzy import fuzz
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.layers.recurrent import LSTM
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from keras.models import load_model
import pickle

SAVE_FEATURES_FILE = "features.pickle" 
SAVE_MODEL_FILE = "model.h5"

def load_and_featurize_question_pairs_data():
  print("loading data from source")
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
      labels.append(label)


  featurized_questions = featurize_questions(question_pairs[:10000])
  featurized_questions = normalize(featurized_questions)
  labels = keras.utils.to_categorical(labels, num_classes=2)

  shuffled_indices = list(range(len(featurized_questions)))
  np.random.shuffle(shuffled_indices)

  shuffled_questions = []
  shuffled_labels = []
  for index in shuffled_indices:
    shuffled_questions.append(featurized_questions[index])
    shuffled_labels.append(labels[index])

  training_percent, test_percent = .8, .2
  training_amount, test_amount = math.floor(len(featurized_questions)*training_percent), math.floor(len(featurized_questions)*test_percent) 

  training_set = np.asarray(shuffled_questions[:training_amount])
  training_labels =  np.asarray(shuffled_labels[:training_amount])
  test_set =  np.asarray(shuffled_questions[training_amount: training_amount + test_amount])
  test_labels =  np.asarray(shuffled_labels[training_amount: training_amount + test_amount])

  print(np.shape(training_set))
  print(np.shape(training_labels))
  print(np.shape(test_set))
  print(np.shape(test_labels))

  print("done loading data from source")
  save_featurized_data(training_set, training_labels, test_set, test_labels)
  return training_set, training_labels, test_set, test_labels

def save_featurized_data(training_set, training_labels, test_set, test_labels):
  print("saving featurized data")
  data = [training_set, training_labels, test_set, test_labels]
  features_out = open(SAVE_FEATURES_FILE, "wb")
  pickle.dump(data, features_out)
  features_out.close()

def load_featurized_data():
  print("loading featurized data")
  features_in = open(SAVE_FEATURES_FILE, "rb")
  training_set, training_labels, test_set, test_labels = pickle.load(features_in)
  return training_set, training_labels, test_set, test_labels

def featurize_questions(question_pairs):
  print("featurizing data")
  featurized_questions = []

  vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)

  all_questions = [question for question_pair in question_pairs for question in question_pair]
  vectorizer.fit(all_questions)
  
  for i in range(len(question_pairs)):
    if i % 100 == 0:
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
  return np.array(featurized_questions)

def train_neural_net(training_set, training_labels, test_set, test_labels):
  print("training neural net")

  n,d = np.shape(training_set)
  sess = tf.Session()
  K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))

  model = Sequential()
  num_hidden_layers = 5
  num_hidden_layer_nodes = 256
  
  # model.add(Conv1D(input_shape=(1024, d), filters=5, kernel_size=10))
  model.add(Dense(num_hidden_layer_nodes, input_dim=d, activation='relu'))
  for i in range(num_hidden_layers):
    model.add(Dense(num_hidden_layer_nodes, activation='relu'))
    # if i % 2 == 1:
    # #   model.add(Conv1D(input_shape=(num_hidden_layer_nodes, 1), filters=5, kernel_size=10))
    #   model.add(LSTM(num_hidden_layer_nodes))
  model.add(Dense(2, activation='softmax'))

  # For a binary classification problem
  model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

  # training_set = pad_training_set(training_set)
  model.fit(training_set, training_labels, epochs=10, batch_size=16)
  print("done training neural net")

  save_trained_model(model)
  return model

# def pad_training_set(training_set):

#   def expand_x(x):
#     x_expanded = []
#     for i in range(len(x)):
#       if x[i] > 0:
#         x_expanded.extend([i]*x[i])
#     return x_expanded

#   n,d = np.shape(training_set)
#   X = []
#   for x in training_set:
#     x_expanded = expand_x(x)
#     new_x = np.eye(d)[x_expanded] 
#     X.append(new_x)
#   return pad_sequences(X, maxlen=1024)

def save_trained_model(model):
  print("saving model")
  model.save(SAVE_MODEL_FILE)

def load_trained_model():
  print("loading model")
  model = load_model(SAVE_MODEL_FILE)
  return model

def predict_and_eval_on_test(model, test_set, test_labels):
  score = model.evaluate(test_set, test_labels, batch_size=128)
  print("\naccuracy", score[1])

def question_pairs(load_features_mode, load_model_mode):
  if load_features_mode:
    training_set, training_labels, test_set, test_labels = load_featurized_data()
  else:
    training_set, training_labels, test_set, test_labels = load_and_featurize_question_pairs_data()
  if load_model_mode:
    model = load_trained_model()
  else:
    model = train_neural_net(training_set, training_labels, test_set, test_labels)
  predict_and_eval_on_test(model, test_set, test_labels)

def get_load_modes(args):
  features_flag = "--lf"
  model_flag = "--lm"

  load_features_mode = False
  load_model_mode = False
  if features_flag in args:
    load_features_mode = True
  if model_flag in args:
    load_model_mode = True

  return load_features_mode, load_model_mode

load_features_mode, load_model_mode = get_load_modes(sys.argv)
question_pairs(load_features_mode, load_model_mode)

