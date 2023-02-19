import sys
import gzip
import pickle
from keras_tuner import HyperParameter
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import HistogramProto
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from keras.models import Model
from keras.models import load_model
from keras.layers import Embedding, Dense, TimeDistributed, Concatenate, BatchNormalization, Masking, LSTM
from keras.layers import Bidirectional, Activation, Dropout, GRU, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from keras.metrics import categorical_accuracy
from keras import backend as K
from sklearn.metrics import confusion_matrix
from keras.regularizers import l1, l2
from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras.layers import add
from keras.layers import RepeatVector, Multiply, Flatten, Dot, Softmax, Lambda, Add, BatchNormalization, Dropout
from keras import backend
import keras
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from keras.layers import LSTM, GRU
from keras.layers import Input,SpatialDropout1D, Embedding, LSTM, Dense, Convolution2D, Lambda, GRU, TimeDistributed, Reshape, Permute, Convolution1D, Masking, Bidirectional
from keras.optimizers import Adam
from keras.layers import concatenate
from keras import optimizers, callbacks
from keras.layers import BatchNormalization, Dropout
import kerastuner as kt


def load_gz(path):
  f = gzip.open(path, 'rb')
  return np.load(f)


cb513 = load_gz('CB513.npy.gz')
print(cb513.shape)
cb6133filtered = load_gz('psp.npy.gz')

del cb513
del cb6133filtered

import gc
gc.collect()

columns = ["id", "len", "input", "profiles", "expected"]
maxlen_seq = r = 700  # protein residues padded to 700
f = 57  # number of features for each residue


###Self Attention Module###

from keras.initializers import Ones, Zeros
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

def attention_scaled_dot(activations):
#https://arxiv.org/pdf/1706.03762.pdf

    units = int(activations.shape[2])
    words = int(activations.shape[1])
    Query = TimeDistributed(Dense(units, activation=None, use_bias=False))(activations)
    Query = Dropout(.2)(Query)
    Key = TimeDistributed(Dense(units, activation=None, use_bias=False))(activations)
    Key = Dropout(.2)(Keys)
    Values = TimeDistributed(Dense(units, activation=None, use_bias=False))(activations)
    Values = Dropout(.2)(Values)
    QK_T = Dot(axes=-1, normalize=False)([Query,Key]) # list of two tensors
    QK_T = Lambda( lambda inp: inp[0]/ backend.sqrt(backend.cast(shape_list(inp[1])[-1], backend.floatx())))([QK_T, Values]) 
    QK_T = Softmax(axis=-1)(QK_T) # do softmax
    QK_T = Dropout(.2)(QK_T)
    Values = Permute([2, 1])(Values)
    V_prime = Dot(axes=-1, normalize=False)([QK_T,Values]) # list of two tensors
    return V_prime


def _get_pos_encoding_matrix(protein_len: int, d_emb: int) -> np.array: #calc position encoding matric
  pos_enc = np.array(
    [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
     range(protein_len)], dtype=np.float32)
  pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
  pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
  return pos_enc

def position_embedding(pos_ids, output_dim=50):
  # gene_vocab_len = 22
  protein_len = 700
  output_dim = int(output_dim)

  pos_emb = Dropout(.1)(Embedding(protein_len, output_dim, trainable=False, input_length=protein_len,
                                  # name='PositionEmbedding',
                                  weights=[_get_pos_encoding_matrix(protein_len, output_dim)])(pos_ids))

  pos_emb = LayerNormalization(1e-5)(pos_emb)
  return pos_emb

def attention_module(x, pos_ids=None, drop_rate=.1):
  original_dim = int(x.shape[-1])
  if pos_ids is not None:
    pos_embedding = position_embedding(pos_ids=pos_ids, output_dim=original_dim)
    x = Add()([x, pos_embedding])
  att_layer = attention_scaled_dot(x)
  att_layer = Dropout(drop_rate)(att_layer)
  x = WeightedSumLayer()([att_layer, x])
  x = Dropout(drop_rate)(x)
  x = BatchNormalization()(x)
  return x




def plot_data(fit):

    #  "Accuracy"
    fig = plt.figure()
    plt.plot(fit.history['Q8_accuracy'])
    plt.plot(fit.history['val_Q8_accuracy'])
    plt.title('model accuracy')
    np.save('history_DCRNN_200_epochs_attention.npy',fit.history)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()
    fig.savefig('accuracy.png')
    # "Loss"
    fig = plt.figure()
    plt.plot(fit.history['loss'])
    plt.plot(fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()
    fig.savefig('loss.png')

def get_data(arr, bounds=None):
    if bounds is None: bounds = range(len(arr))

    data = [None for i in bounds]

    for i in bounds:
        seq, q8, profiles = '', '', []
        for j in range(r):
            jf = j * f

            # Residue convert from one-hot to decoded
            residue_onehot = arr[i, jf + 0:jf + 22]
            residue = residue_list[np.argmax(residue_onehot)]

            # Q8 one-hot encoded to decoded structure symbol
            residue_q8_onehot = arr[i, jf + 22:jf + 31]
            residue_q8 = q8_list[np.argmax(residue_q8_onehot)]

            if residue == 'NoSeq': break  # terminating sequence symbol

            nc_terminals = arr[i, jf + 31:jf + 33]  # nc_terminals = [0. 0.]
            sa = arr[i, jf + 33:jf + 35]  # sa = [0. 0.]
            profile = arr[i, jf + 35:jf + 57]  # profile features

            seq += residue  # concat residues into amino acid sequence
            q8 += residue_q8  # concat secondary structure into secondary structure sequence
            profiles.append(profile)


        data[i] = [str(i + 1), len(seq), seq, np.array(profiles), q8]

    return pd.DataFrame(data, columns=columns)


def show_secondary(array):

    to_img = np.copy(array)
    for i in range(to_img.shape[0]):
        for j in range(to_img.shape[1]):
            for k in range(to_img.shape[2]):
                if to_img[i, j, k] == 1:
                    to_img[i, j, k] = 255
    #img = Image.fromarray(to_img)
    plt.figure(figsize=(28, 10))
    plt.imshow(np.transpose(to_img[7, :, :]))
    plt.show()

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
    fig.savefig('confusion_matrix.png')

    fig = plt.figure()
    matrix_recall = (matrix.astype('float') / np.sum(matrix, axis=1)[:,np.newaxis]).round(2)
    df_cm = pd.DataFrame(matrix_recall, ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'], ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'])
    sn.set(font_scale=1.0)  # for label size
    sn.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 10})  # font size
    plt.show()
    
    fig.savefig('confusion_matrix_recall.png')
    fig = plt.figure()
    matrix_precision = (matrix / matrix.astype(np.float).sum(axis=0)).round(2)
    df_cm = pd.DataFrame(matrix_precision, ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'], ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'])
    sn.set(font_scale=1.0)  # for label size
    sn.heatmap(df_cm, annot=True, cmap="PuRd", fmt='g', annot_kws={"size": 10})  # font size
    plt.show()
    fig.savefig('confusion_matrix_precision.png')

    return matrix

# Convert probabilities or one_hot to secondary structure string sequences
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

def categorical_crossentropy(y_true, y_pred):

    return K.mean(K.categorical_crossentropy(y_pred, y_true))

def conv_block(x, activation=True, batch_norm=True, drop_out=True, res=True):
    cnn = Conv1D(64, 11, padding="same")(x)
    if activation: cnn = TimeDistributed(Activation("relu"))(cnn)
    if batch_norm: cnn = TimeDistributed(BatchNormalization())(cnn)
    if drop_out:   cnn = TimeDistributed(Dropout(0.5))(cnn)
    if res:        cnn = Concatenate(axis=-1)([x, cnn])

    return cnn


def super_conv_block(x):
    c3 = Conv1D(32, 1, padding="same")(x)
    c3 = TimeDistributed(Activation("relu"))(c3)
    #c3 = TimeDistributed(Dropout(0.1))(c3)
    c3 = TimeDistributed(BatchNormalization())(c3)

    c7 = Conv1D(64, 3, padding="same")(x)
    c7 = TimeDistributed(Activation("relu"))(c7)
    #c7 = TimeDistributed(Dropout(0.1))(c7)
    c7 = TimeDistributed(BatchNormalization())(c7)

    c11 = Conv1D(128, 5, padding="same")(x)
    c11 = TimeDistributed(Activation("relu"))(c11)
    #c11 = TimeDistributed(Dropout(0.1))(c11)
    c11 = TimeDistributed(BatchNormalization())(c11)

    x = Concatenate(axis=-1)([x, c3, c7, c11])
    x = TimeDistributed(Dropout(0.5))(x)
    return x

def hyper_super_conv_block(x, hp):
    c3 = Conv1D(32, 1, padding="same")(x)
    c3 = TimeDistributed(Activation("relu"))(c3)
    c3 = TimeDistributed(Dropout(hp.Float(name='drops_4',
            min_value=0.0, max_value=0.3, step=0.1)))(c3)
    c3 = TimeDistributed(BatchNormalization())(c3)
    

    c7 = Conv1D(64, 3, padding="same")(x)
    c7 = TimeDistributed(Activation("relu"))(c7)
    c7 = TimeDistributed(Dropout(hp.Float(name='drops_5',
            min_value=0.0, max_value=0.3, step=0.1)))(c7)
    c7 = TimeDistributed(BatchNormalization())(c7)

    c11 = Conv1D(128, 5, padding="same")(x)
    c11 = TimeDistributed(Activation("relu"))(c11)
    c11 = TimeDistributed(Dropout(hp.Float(name='drops_6',
            min_value=0.0, max_value=0.3, step=0.1)))(c11)
    c11 = TimeDistributed(BatchNormalization())(c11)
 
    x = Concatenate(axis=-1)([x, c3, c7, c11])
    x = TimeDistributed(Dropout(0.5))(x)
    return x

# The custom accuracy metric used for this task

def Q8_accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis=- 1)
    y_ = tf.argmax(y_pred, axis=- 1)
    mask = tf.greater(y, 0)
    print (K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx()))
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

def accuracy_Q8(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if np.sum(real[i, j, :]) == 0:
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    print(total)
    return correct / total
def count_noSeq(X):
    total = X.shape[0] * X.shape[1]
    count = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j, 30] == 1:
                count = count + 1
    return count

def accuracy_Q8variant(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if np.sum(real[i, j, :]) == 0:
                if np.sum(pred[i, j, :]) == 0:
                    correct = correct + 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    print(total)
    return correct / total
################################################################
""" train data processing """
################################################################

dataset_raw_input = load_gz('psp.npy.gz')
dataset = np.reshape(dataset_raw_input, (5365, 700, 57))
del dataset_raw_input
gc.collect()

features_data = dataset[:, :, 0:21]

profile_data = dataset[:, :, 35:56]
s_structure_label = dataset[:, :, 22:30]

s_accessibility_label = dataset[:, :, 33:35]

residue_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
                'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', 'NoSeq']
q8_list = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

aminoacids_lenght = 700 # protein residues padded to 700
features = 57 # number of features for each residue
feature_index = np.hstack((np.arange(0, 21), np.arange(35, 56)))  # 42-d features
X = dataset[:, :, feature_index]
#print(count_noSeq(dataset))
print(X.shape)
print(X[200][699])
vals = np.arange(0, 8)
num_seqs, seqlen, feature_dim = np.shape(X)

num_classes = len(q8_list) - 1

""" build secondary structure label"""
labels_new = np.zeros((num_seqs, seqlen))
for i in range(np.size(s_structure_label, axis=0)):
    labels_new[i, :] = np.dot(s_structure_label[i, :, :], vals)
labels_new = labels_new.astype('int32')
s_structure_label2 = labels_new


""" build solvent accessibility label"""
labels_new_2 = np.zeros((num_seqs, seqlen))
vals2 = np.array([2, 1])
for i in range(np.size(s_accessibility_label, axis=0)):
    labels_new_2[i, :] = np.dot(s_accessibility_label[i, :, :], vals2)
labels_new_2 = labels_new_2.astype('int32')
s_accessibility_label = labels_new_2
print(s_accessibility_label[200])
""" shuffle data"""
seq_shuffle = np.arange(0, num_seqs)
np.random.shuffle(seq_shuffle)
X_train = X[seq_shuffle[0:4869]]
s_struct_train = s_structure_label[seq_shuffle[0:4869]]
s_acc_train = s_accessibility_label[seq_shuffle[0:4869]]
features_data = features_data[seq_shuffle[0:4869]]
profile_data = profile_data[seq_shuffle[0:4869]]


print(profile_data[12][12])


################################################################
"""validation data processing"""
################################################################

X_valid = X[seq_shuffle[4869:5534]]
s_struct_valid = s_structure_label[seq_shuffle[4869:5365]]
s_acc_valid = s_accessibility_label[seq_shuffle[4869:5365]]

################################################################
"""test data processing"""
################################################################

testset_raw_input = load_gz('CB513.npy.gz')
testset = np.reshape(testset_raw_input, (514, 700, 57))

del testset_raw_input
gc.collect()

s_structure_label = testset[:, :, 22:30]

s_accessibility_label = testset[:, :, 33:35]
feature_index = np.hstack((np.arange(0, 21), np.arange(35, 56)))  # 42-d features
X = testset[:, :, feature_index]

# getting meta
num_seqs, seqlen, feature_dim = np.shape(X)
num_classes = 8

vals = np.arange(0, 8)

""" build secondary structure label"""
labels_new = np.zeros((num_seqs, seqlen))
for i in range(np.size(s_structure_label, axis=0)):
    labels_new[i, :] = np.dot(s_structure_label[i, :, :], vals)
labels_new = labels_new.astype('int32')
s_structure_label2 = labels_new

""" build solvent accessibility label"""
labels_new_2 = np.zeros((num_seqs,seqlen))
vals2 = np.array([2, 1])
for i in range(np.size(s_accessibility_label, axis=0)):
    labels_new_2[i, :] = np.dot(s_accessibility_label[i, :, :], vals2)
labels_new_2 = labels_new_2.astype('int32')
s_accessibility_label = labels_new_2

X_test = X
s_struct_test = s_structure_label
s_acc_test = s_accessibility_label

import gc
del dataset
del features_data
del profile_data
gc.collect()


################################################################
"""model"""
################################################################

def generate_model():
    input_1 = Input(shape=(700, 21))
    input_2 = Input(shape=(700, 21))

    input_1_embedded = Dense(50, activation='relu')(input_1)
    pos_ids = Input(batch_shape=(None, 700), name='position_input', dtype='int32')

    input_concat = Concatenate(axis=-1)([input_1_embedded, input_2])

    x = super_conv_block(input_concat)

    x = conv_block(x)
    #x = attention_module(x, pos_ids, drop_rate=0.2)
    #x = Dropout(0.1)(x)
    x = super_conv_block(x)
    x = conv_block(x)
    tf.print("X pre-attention:", x.shape, output_stream=sys.stderr)
    #x = attention_module(x, pos_ids, drop_rate=0.2)
    #x = Dropout(0.1)(x)
    tf.print("X post-attention:", x.shape, output_stream=sys.stderr)
    x = super_conv_block(x)
    x = conv_block(x)
    #x = attention_module(x, pos_ids, drop_rate=0.2)
    # x = Dropout(0.1)(x)

    gru1 = Bidirectional(GRU(128,
                             return_sequences='True',
                             activation='tanh',
                             recurrent_activation='sigmoid',
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             dropout=0.3,
                             recurrent_dropout=0,
                             implementation=1))(x)
    '''
    local_and_global1= Concatenate(axis=-1)([gru1, x])
    c = Conv1D(500, 1, padding="same")(local_and_global1)
    c = TimeDistributed(Activation("relu"))(c)
    c = TimeDistributed(Dropout(0.2))(c)
    c = TimeDistributed(BatchNormalization())(c)
    '''
    gru2 = Bidirectional(GRU(128,
                             return_sequences='True',
                             activation='tanh',
                             recurrent_activation='sigmoid',
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             dropout=0.3,
                             recurrent_dropout=0,
                             implementation=1))(gru1)
    '''
    local_and_global2= Concatenate(axis=-1)([gru2, gru1])
    c2 = Conv1D(500, 1, padding="same")(local_and_global2)
    c2 = TimeDistributed(Activation("relu"))(c2)
    c2 = TimeDistributed(Dropout(0.3))(c2)
    c2 = TimeDistributed(BatchNormalization())(c2)
    att = attention_module(c2, pos_ids)
    '''
    gru3 = Bidirectional(GRU(128,
                             return_sequences='True',
                             activation='tanh',
                             recurrent_activation='sigmoid',
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             dropout=0.3,
                             recurrent_dropout=0,
                             implementation=1))(gru2)
    # x = attention_module(x, pos_ids)
    local_and_global = Concatenate(axis=-1)([gru3, x])
    x = TimeDistributed(Dense(256,
                              activation='relu',
                              use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))(gru3)
    x = Dropout(0.5)(x)
    x = attention_module(x, pos_ids)
    x = TimeDistributed(Dense(128,
                              activation='relu',
                              use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))(x)
    x = Dropout(0.5)(x)

    x = TimeDistributed(Dense(128,
                              activation='relu',
                              use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))(x)

    x = Dropout(0.5)(x)
    # x = attention_module(x,pos_ids)
    out_1 = TimeDistributed(Dense(8, activation="softmax"))(x)
    model = Model((input_1, input_2, pos_ids), out_1)
    return model

model = generate_model()
model.summary()
for i, l in enumerate(model.layers):
    print(f'layer {i}: {l}')
    print(f'has input mask: {l.input_mask}')
    print(f'has output mask: {l.output_mask}')
#mask_layer=Masking(mask_value=0., input_shape=(timesteps, features))(inputs)
#print(mask_layer._keras_mask)

model.compile(optimizer="nadam",
  loss="categorical_crossentropy",
  sample_weight_mode = "temporal", 
  metrics=[Q8_accuracy])

import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose= 1)
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

### Self Attention Module position encoding: Testing Set ###

cb513_pos = np.array(range(700))
cb513_pos = np.repeat([cb513_pos], 514, axis=0)

### Testing del modello ###

saved_model = load_model('best_model.h5', custom_objects={"Q8_accuracy" : Q8_accuracy, "LayerNormalization": LayerNormalization, "WeightedSumLayer":WeightedSumLayer, "backend": backend, "shape_list": shape_list})
pred = saved_model.predict([X_test[:, :, 0:21], X_test[:, :, 21:42], cb513_pos])
one_hot_predictions = probabilities_to_onehot(pred)
yt = to_int_seq(s_struct_test, s_struct_test)
print("s_struct_test.shape: "+str(s_struct_test.shape))
print("one_hot_predictions.shape: "+str(one_hot_predictions.shape))
yp = to_int_seq(s_struct_test, one_hot_predictions)
yt = list(map(str, yt))
yp = list(map(str, yp))
print(make_confusion_matrix(yt, yp))
tf.compat.v1.disable_eager_execution()
acc = Q8_accuracy(s_struct_test, pred)
print ('accuracy on cb513:', tf.compat.v1.Session().run(acc).mean())
print(accuracy_Q8(s_struct_test, pred))

'''
### Self Attention Module position encoding: Train Set and Validation Set###

positions_seq = np.array(range(700))
positions_seq = np.repeat([positions_seq],4869, axis=0)

valid_pos = np.array(range(700))
valid_pos = np.repeat([valid_pos],496, axis=0)

cb513_pos = np.array(range(700))
cb513_pos = np.repeat([cb513_pos], 514, axis=0)
### Aggiunto by Bernardo per l'Attention Module: Fine ###

lengths_valid = np.sum(np.sum(X_valid[ :, :, :21], axis=2), axis=1).astype(int)
weight_mask_valid = np.zeros((X_valid.shape[0], 700))
for i in range(len(lengths_valid)):
  weight_mask_valid[i, : lengths_valid[i]] = 1.0

lengths_train = np.sum(np.sum(X_train[ :, :, :21], axis=2), axis=1).astype(int)

weight_mask_train = np.zeros((X_train.shape[0], 700))
for i in range(len(lengths_train)):
  weight_mask_train[i, : lengths_train[i]] = 1.0

### Fit del modello ###

fit = model.fit(x=[X_train[:, :, 0:21], X_train[:, :, 21:42], positions_seq], y=s_struct_train,
                validation_data=([X_valid[:, :, 0:21], X_valid[:, :, 21:42], valid_pos], s_struct_valid, weight_mask_valid),
                callbacks=[mc, tensorboard_callback],
                batch_size=64,
                epochs=200,
                sample_weight=weight_mask_train,
                verbose=1)
y_pre = model.predict([X_test[:, :, 0:21], X_test[:, :, 21:42], cb513_pos])

one_hot_predictions = probabilities_to_onehot(y_pre)
#print(Q8_accuracy(s_struct_test, one_hot_predictions).mean())

yt = to_int_seq(s_struct_test, s_struct_test)
print("s_struct_test.shape: "+str(s_struct_test.shape))
print("one_hot_predictions.shape: "+str(one_hot_predictions.shape))
yp = to_int_seq(s_struct_test, one_hot_predictions)
yt = list(map(str, yt))
yp = list(map(str, yp))
plot_data(fit)
print(make_confusion_matrix(yt, yp))
#show_secondary(s_struct_test)
#plt.savefig('true_secondary.png')
#show_secondary(y_pre)
#plt.savefig('predicted_secondary.png')
#plot_data(fit)
#print(K.get_value(Q8_accuracy(s_struct_test, y_pre)))

#history=np.load('my_history.npy',allow_pickle='TRUE').item()
#print(history['Q8_accuracy'])
'''

'''
### HyperParameter Tuning ###

def hyper_model(hp):
    input_1 = Input(shape=(700, 21))
    input_2 = Input(shape=(700, 21))

    #input_1_reshape = Reshape(-1, 21)(input_1)
    #input_2_reshape = Reshape(-1, 21)(input_2)

    #input1_masked = Masking(mask_value=0., input_shape=(700, 21))(input_1)
    #input2_masked = Masking(mask_value=0., input_shape=(700, 21))(input_2)


    input_1_embedded = Dense(50, activation='relu')(input_1)
    pos_ids = Input(batch_shape=(None, 700), name='position_input', dtype='int32')

    input_concat = Concatenate(axis=-1)([input_1_embedded, input_2])

    x = hyper_super_conv_block(input_concat, hp)

    x = conv_block(x)
    x = attention_module(x, pos_ids)
    x = Dropout(hp.Float(name='drops_1',
            min_value=0.0, max_value=0.5, step=0.1))(x)
    x = hyper_super_conv_block(x, hp)
    x = conv_block(x)
    tf.print("X pre-attention:", x.shape, output_stream=sys.stderr)
    x = attention_module(x, pos_ids)
    x = Dropout(hp.Float(name='drops_2',
            min_value=0.0, max_value=0.5, step=0.1))(x)
    tf.print("X post-attention:", x.shape, output_stream=sys.stderr)
    x = hyper_super_conv_block(x, hp)
    x = conv_block(x)
    x = attention_module(x, pos_ids)
    x = Dropout(hp.Float(name='drops_3',
            min_value=0.0, max_value=0.5, step=0.1))(x)

    #input = Reshape(( 700, 21+50, 1))(input_concat)

    #input = Reshape(( 700, 21+50))(input_concat)

    gru1 = Bidirectional(GRU(128,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.3,
                         recurrent_dropout=0,
                         implementation=1))(x)



    gru2 = Bidirectional(GRU(128,
                             return_sequences='True',
                             activation='tanh',
                             recurrent_activation='sigmoid',
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             dropout=0.3,
                             recurrent_dropout=0,
                             implementation=1))(gru1)

    gru3 = Bidirectional(GRU(128,
                             return_sequences='True',
                             activation='tanh',
                             recurrent_activation='sigmoid',
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             dropout=0.3,
                             recurrent_dropout=0,
                             implementation=1))(gru2)

    local_and_global = Concatenate(axis=-1)([gru3, x])
    x = TimeDistributed(Dense(256,
                              activation='relu',
                              use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))(local_and_global)
    x = Dropout(0.5)(x)
    
    x = TimeDistributed(Dense(128,
                              activation='relu',
                              use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))(x)
    x = Dropout(0.5)(x)

    x = TimeDistributed(Dense(128,
                              activation='relu',
                              use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))(x)

    x = Dropout(0.5)(x)
    #print(gru3)
    # x = attention_module(x,pos_ids)
    out_1 = TimeDistributed(Dense(8, activation="softmax"))(x)
    model = Model((input_1, input_2, pos_ids), out_1)
    model.summary()
    model.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=[Q8_accuracy])
    return model

### Aggiunto by Bernardo per l'Attention Module ###

positions_seq = np.array(range(700))
positions_seq = np.repeat([positions_seq],5022, axis=0)

valid_pos = np.array(range(700))
valid_pos = np.repeat([valid_pos],343, axis=0)

cb513_pos = np.array(range(700))
cb513_pos = np.repeat([cb513_pos], 514, axis=0)
### Aggiunto by Bernardo per l'Attention Module: Fine ###


tuner = kt.RandomSearch(hypermodel=hyper_model,
                        objective='val_loss',
                        max_trials=25,
                        directory='test_dir',
                        project_name='a')

tuner = kt.BayesianOptimization(hypermodel=hyper_model,
                                objective='val_loss',
                                max_trials=25,
                                num_initial_points=2,
                                directory='test_dir',
                                project_name='a')

es = tf.keras.callbacks.EarlyStopping(patience=15)
tuner.search([X_train[:, :, 0:21], X_train[:, :, 21:42], positions_seq], s_struct_train, epochs=85,
             batch_size=64,
             validation_data=([X_valid[:, :, 0:21], X_valid[:, :, 21:42], valid_pos], s_struct_valid),
             verbose=1,
            callbacks=[es])
'''


