CUDA_VISIBLE_DEVICES=0

train_dataset_dir = "psp.npy.gz"
best_model_file = './saint_pssp.h5'
cb513_dataset_dir = "CB513.npy.gz"

load_saved_model = True
lr=0.0005
conv_layer_dropout_rate = 0.2
dense_layer_dropout_rate = 0.5

evaluate_on_cb513 = True
show_F1score_cb513 = True

from re import X
import tensorflow as tf
import gzip
from tensorflow.python.client import device_lib
import os
import subprocess
import numpy as np
from keras.initializers import Ones, Zeros
from keras.layers import Layer
from copy import deepcopy
import h5py
from keras import backend as K
from keras.layers import Layer
from keras.layers import add
from keras import backend
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras import optimizers, callbacks
from keras.layers import BatchNormalization, Dropout
from keras.models import Model
from keras.layers import RepeatVector, Multiply, Flatten, Dot, Softmax, Lambda, Add, BatchNormalization, Dropout, concatenate
from keras.layers import Input, Dense, Lambda, TimeDistributed, Reshape, Permute, Masking
from keras.regularizers import l2
import keras
from keras.callbacks import LambdaCallback
from keras import callbacks, backend
from keras.optimizers import Adam
from pprint import pprint
import gc
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
  


def check_gpu():
  print(device_lib.list_local_devices())
  if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
  else:
    print("Please install GPU version of TF")


def load_gz(path):
  f = gzip.open(path, 'rb')
  return np.load(f)

def decode(x): # used to take make the tokens
  return np.argmax(x)

def to_int_seq(y1, y2):
    seqs=[]
    for i in range(len(y1)):

        for j in range(len(y1[i])):
            if np.sum(y1[i, j, :]) != 0:
                seq_i = np.argmax(y2[i][j])
                seqs.append(seq_i)

    return seqs
  
def probabilities_to_onehot(y):

  one_hot = np.zeros(y.shape)
  for i in range(y.shape[0]):
      for j in range(y.shape[1]):
          if np.sum(y[i, j, :]) > 0:
              one_hot[i, j, np.argmax(y[i, j, :])] = 1
  return one_hot
  
  
def make_confusion_matrix(yt, yp):
  fig = plt.figure()
  #matrix = confusion_matrix(yt, yp, normalize='true')
  matrix = confusion_matrix(yt, yp)
  df_cm = pd.DataFrame(matrix, ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'], ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'])
  # plt.figure(figsize=(10,7))
  sn.set(font_scale=1.0)  # for label size
  #sn.heatmap(df_cm, annot=True, fmt ='.2f', annot_kws={"size": 10})  # font size
  sn.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 10})  # font size
  recall = np.diag(matrix)/np.sum(matrix, axis=1)
  precision = np.diag(matrix)/np.sum(matrix, axis=0)

  print("recall: "+str(recall))
  print("precision: "+str(precision))

  plt.show()
  fig.savefig('Saint confusion_matrix.png')

  fig = plt.figure()
  matrix_recall = (matrix.astype('float') / np.sum(matrix, axis=1)[:,np.newaxis]).round(2)
  df_cm = pd.DataFrame(matrix_recall, ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'], ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'])
  sn.set(font_scale=1.0)  # for label size
  sn.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 10})  # font size
  plt.show()
    
  fig.savefig('Saint confusion_matrix_recall.png')
  fig = plt.figure()
  matrix_precision = (matrix / matrix.astype(np.float).sum(axis=0)).round(2)
  df_cm = pd.DataFrame(matrix_precision, ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'], ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'])
  sn.set(font_scale=1.0)  # for label size
  sn.heatmap(df_cm, annot=True, cmap="PuRd", fmt='g', annot_kws={"size": 10})  # font size
  plt.show()
  fig.savefig('Saint confusion_matrix_precision.png')

  return matrix

  

class LayerNormalization(Layer):
  def __init__(self, eps: float = 1e-5, **kwargs) -> None:
    self.eps = eps
    super().__init__(**kwargs)

  def build(self, input_shape):
    self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
    self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
    super().build(input_shape)

  def call(self, x, **kwargs):
    u = K.mean(x, axis=-1, keepdims=True)
    s = K.mean(K.square(x - u), axis=-1, keepdims=True)
    z = (x - u) / K.sqrt(s + self.eps)
    return self.gamma * z + self.beta

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
      'eps': self.eps,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
    
    

class StepDecay():
  def __init__(self, initAlpha=0.0005, factor=0.9, dropEvery=60):
    self.initAlpha = initAlpha
    self.factor = factor
    self.dropEvery = dropEvery

  def __call__(self, epoch):
    exp = np.floor((epoch + 1) / self.dropEvery)
    alpha = self.initAlpha * (self.factor ** exp)
    return float(alpha)

# Attention #

class WeightedSumLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightedSumLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self._x = K.variable(0.2)
        self._x._trainable = True
        self._trainable_weights = [self._x]

        super(WeightedSumLayer, self).build(input_shape)

    def call(self, x):
        A, B = x
        result = add([self._x*A ,(1-self._x)*B])
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def shape_list(x):
    if backend.backend() != 'theano':
        tmp = backend.int_shape(x)
    else:
        tmp = x.shape
    tmp = list(tmp)
    tmp[0] = -1
    return tmp

def tp_fp_fn_counter(predicted_pss, true_pss, lengths):
  pred_dict = {}
  secondary_labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
  for i in range(8):
    pred_dict[i] = {'label':secondary_labels[i], 'TP':0, 'FP':0, 'FN':0}

  for i in range(len(true_pss)):
    for j in range(lengths[i]):
      pred_ = np.argmax(predicted_pss[i][j])
      true_ = np.argmax(true_pss[i][j])
      if pred_ == true_ :
        pred_dict[pred_]['TP'] += 1
      else:
        pred_dict[pred_]['FP'] += 1
        pred_dict[true_]['FN'] += 1
  return pred_dict


def amino_acid_wise_precision_recall_F1(predicted_pss, true_pss, lengths):
  tp_fp_fn_dict = tp_fp_fn_counter(predicted_pss, true_pss, lengths)
  prec_recall_dict = {}
  secondary_labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
  for i in range(8):
    prec_recall_dict[i] = {'label': secondary_labels[i], 'precision': 0, 'recall': 0, 'F1': 0}

  for i in range(8):
    try:
      prec_recall_dict[i]['precision'] = tp_fp_fn_dict[i]['TP'] / (tp_fp_fn_dict[i]['TP'] + tp_fp_fn_dict[i]['FP'])
      prec_recall_dict[i]['recall'] = tp_fp_fn_dict[i]['TP'] / (tp_fp_fn_dict[i]['TP'] + tp_fp_fn_dict[i]['FN'])
      prec_recall_dict[i]['F1'] = 2 * (prec_recall_dict[i]['precision'] * prec_recall_dict[i]['recall']) / (
      prec_recall_dict[i]['precision'] + prec_recall_dict[i]['recall'])
    except:
      # print('All zero values. skipped.')
      pass

  return prec_recall_dict, tp_fp_fn_dict


def truncated_accuracy(y_true, y_predict):
  mask = K.sum(y_true, axis=2)
  y_pred_labels = K.cast(K.argmax(y_predict, axis=2), 'float32')
  y_true_labels = K.cast(K.argmax(y_true, axis=2), 'float32')
  is_same = K.cast(K.equal(
    y_true_labels, y_pred_labels), 'float32')
  num_same = K.sum(is_same * mask, axis=1)
  lengths = K.sum(mask, axis=1)
  return K.mean(num_same / lengths, axis=0)

def Q8_accuracy(y_true, y_pred):
  y = tf.argmax(y_true, axis=- 1)
  y_ = tf.argmax(y_pred, axis=- 1)
  mask = tf.greater(y, 0)
  print (K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx()))
  return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())


###................ load the model...........###
if load_saved_model:
  best_model_file = best_model_file
  model = keras.models.load_model(best_model_file, custom_objects={'LayerNormalization':LayerNormalization,'shape_list':shape_list,'backend':backend,'WeightedSumLayer':WeightedSumLayer, 'truncated_accuracy':truncated_accuracy, 'Q8_accuracy':Q8_accuracy})

adam = Adam(lr=lr)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              sample_weight_mode='temporal',
              metrics=[Q8_accuracy, 'mae', truncated_accuracy])

if evaluate_on_cb513 or show_F1score_cb513:
  print('Testing on CB513...\n')
  ###............... load and process CB513 to validate ..............###

  dataindex = list(range(35, 56))
  labelindex = range(22, 30)
  print('Loading Test data [ CB513 ]...')
  cb513 = load_gz(cb513_dataset_dir)
  cb513 = np.reshape(cb513, (514, 700, 57))
  # print(cb513.shape)
  x_test = cb513[:, :, dataindex]
  y_test = cb513[:, :, labelindex]

  cb513_protein_one_hot = cb513[:, :, : 21]
  cb513_protein_one_hot_with_noseq = cb513[:, :, : 22]
  lengths_cb = np.sum(np.sum(cb513_protein_one_hot, axis=2), axis=1).astype(int)
  # print(cb513_protein_one_hot_with_noseq.shape)
  del cb513_protein_one_hot
  gc.collect()

  cb513_seq = np.zeros((cb513_protein_one_hot_with_noseq.shape[0], cb513_protein_one_hot_with_noseq.shape[1]))
  for j in range(cb513_protein_one_hot_with_noseq.shape[0]):
    for i in range(cb513_protein_one_hot_with_noseq.shape[1]):
      datum = cb513_protein_one_hot_with_noseq[j][i]
      cb513_seq[j][i] = int(decode(datum))

  cb513_pos = np.array(range(700))
  cb513_pos = np.repeat([cb513_pos], 514, axis=0)

  cb_scores = model.evaluate([x_test, cb513_protein_one_hot_with_noseq, cb513_pos], y_test)
  # print(cb_scores)
  print("Accuracy: " + str(round(cb_scores[3]*100, 2)) + "%, Loss: " + str(cb_scores[0])+'\n\n')
  
  pred = model.predict([x_test, cb513_protein_one_hot_with_noseq, cb513_pos])
  yt = to_int_seq(y_test, y_test)
  one_hot_predictions = probabilities_to_onehot(pred)
  print("y_test.shape: "+str(y_test.shape))
  yp = to_int_seq(y_test, one_hot_predictions)
  yt = list(map(str, yt))
  yp = list(map(str, yp))
  print(make_confusion_matrix(yt, yp))
  acc = Q8_accuracy(y_test, pred)
  m = tf.keras.metrics.Mean()
  m.update_state(acc)
  acc2 =m.result().numpy()
  print('accuracy on cb513:')
  tf.print(acc2)

  ### for amino_acid_wise_precision_recall_F1
  if show_F1score_cb513:
    print('Calcualting Precision, Recall and F1 score...\n')
    y_pred_casp = model.predict([x_test, cb513_protein_one_hot_with_noseq, cb513_pos], verbose=1)
    precision_recall_F1_dict = amino_acid_wise_precision_recall_F1(y_pred_casp, y_test, lengths=lengths_cb)
    print('Precision, Recall and F1-score (per amino-acid)')
    pprint(precision_recall_F1_dict[0])
    print('\n\n')
    print('False Negative(FN), False Positive(FP), True Positive counts(TP) counts (per amino-acid)')
    pprint(precision_recall_F1_dict[1])
    print('\n\n')

