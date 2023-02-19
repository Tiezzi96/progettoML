# config_train

## cb6133 dataset link: http://www.princeton.edu/~jzthree/datasets/ICML2014/
## cb513 dataset link: http://www.princeton.edu/~jzthree/datasets/ICML2014/
'''
# CUDA_VISIBLE_DEVICES=0
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
'''

train_dataset_dir = "psp.npy.gz"
best_model_file = './saint_pssp.h5'
cb513_dataset_dir = "CB513.npy.gz"

to_train = True #False
lr=0.0005
conv_layer_dropout_rate = 0.2
dense_layer_dropout_rate = 0.5

# end config_train

# utility.py

import tensorflow as tf
import gzip
from tensorflow.python.client import device_lib
import numpy as np
import os
import subprocess
from keras.initializers import Ones, Zeros
from keras.layers import Layer
from copy import deepcopy
import h5py
import keras
from keras.callbacks import LambdaCallback
from keras import callbacks, backend
from keras.optimizers import Adam
import gc
from keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import add
from tensorflow.keras import backend
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Input,SpatialDropout1D, Embedding, LSTM, Dense, Convolution2D, Lambda, GRU, TimeDistributed, Reshape, Permute, Convolution1D, Masking, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.layers import RepeatVector, Multiply, Flatten, Dot, Softmax, Lambda, Add, Dropout, concatenate, BatchNormalization
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix



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

'''
class StepDecayStepDecay():
  def __init__(self, initAlpha=0.0005, factor=0.9, dropEvery=60):
    self.initAlpha = initAlpha
    self.factor = factor
    self.dropEvery = dropEvery

  def __call__(self, epoch):
    exp = np.floor((epoch + 1) / self.dropEvery)
    alpha = self.initAlpha * (self.factor ** exp)
    return float(alpha)
'''
# end utility.py


# attention module tools.py


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
    Key = Dropout(.2)(Key)
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

# end attention module tools.py


# metric.py
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

    print(recall)
    print(precision)

    plt.show()
    fig.savefig('confusion_matrix.png')

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
    
def plot_graph(history):
    print(history.history.keys())
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.plot(history.history['Q8_accuracy'])
    plt.plot(history.history['val_Q8_accuracy'])
    plt.title('model Q8 accuracy')
    plt.ylabel('Q8 accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Q8 accuracy.png")
    plt.show()
    plt.plot(history.history['truncated_accuracy'])
    plt.plot(history.history['val_truncated_accuracy'])
    plt.title('model truncated accuracy')
    plt.ylabel('truncated accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("truncated accuracy.png")
    plt.show()
  
    with open('history_saint_100_epochs_3su3_convolutions_weight_mask.npy', 'wb') as f:
        np.save(f, history.history)
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")
    #plt.show()

# end metric.py

# model.py

def inceptionBlock(x):
  x = BatchNormalization()(x)
  conv1_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  print("Conv1_1 pre dropout "+str(conv1_1))
  conv1_1 = Dropout(conv_layer_dropout_rate)(conv1_1)  # https://www.quora.com/Can-l-combine-dropout-and-l2-regularization
  print("Conv1_1 post dropout "+str(conv1_1))
  conv1_1 = BatchNormalization()(conv1_1)

  conv2_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  conv2_1 = Dropout(conv_layer_dropout_rate)(conv2_1)
  conv2_1 = BatchNormalization()(conv2_1)
  conv2_2 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2_1)
  conv2_2 = Dropout(conv_layer_dropout_rate)(conv2_2)
  conv2_2 = BatchNormalization()(conv2_2)

  conv3_1 = Convolution1D(100, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
  conv3_1 = Dropout(conv_layer_dropout_rate)(conv3_1)
  conv3_1 = BatchNormalization()(conv3_1)
  conv3_2 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_1)
  conv3_2 = Dropout(conv_layer_dropout_rate)(conv3_2)
  conv3_2 = BatchNormalization()(conv3_2)
  conv3_3 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_2)
  conv3_3 = Dropout(conv_layer_dropout_rate)(conv3_3)
  conv3_3 = BatchNormalization()(conv3_3)
  conv3_4 = Convolution1D(100, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_3)
  conv3_4 = Dropout(conv_layer_dropout_rate)(conv3_4)
  conv3_4 = BatchNormalization()(conv3_4)

  concat = concatenate([conv1_1, conv2_2, conv3_4])
  #concat = concatenate([conv1_1, conv2_2])
  #concat = conv1_1
  concat = BatchNormalization()(concat)

  return concat

def deep3iBLock_with_attention(x, pos_ids=None):
  block1_1 = inceptionBlock(x)
  block1_1 = attention_module(block1_1, pos_ids)

  block2_1 = inceptionBlock(x)
  block2_2 = inceptionBlock(block2_1)
  block2_2 = attention_module(block2_2, pos_ids)

  block3_1 = inceptionBlock(x)
  block3_2 = inceptionBlock(block3_1)
  block3_3 = inceptionBlock(block3_2)
  block3_4 = inceptionBlock(block3_3)
  block3_4 = attention_module(block3_4, pos_ids)


  concat = concatenate([block1_1, block2_2, block3_4])
  # concat = concatenate([block1_1, block2_2])
  # concat = block1_1
  concat = BatchNormalization()(concat)

  return concat


def generate_model():
  pssm_input = Input(shape=(700, 21,), name='pssm_input')
  seq_input = Input(shape=(700, 22,), name='seq_input')
  pos_ids = Input(batch_shape=(None, 700), name='position_input', dtype='int32')

  main_input = concatenate([seq_input, pssm_input])

  block1 = deep3iBLock_with_attention(main_input, pos_ids)
  block2 = deep3iBLock_with_attention(block1, pos_ids)
  block2 = attention_module(block2, pos_ids)

  conv11 = Convolution1D(100, 11, activation='relu', padding='same', kernel_regularizer=l2(0.001))(block2)
  conv11 = attention_module(conv11, pos_ids)

  dense1 = TimeDistributed(Dense(units=256, activation='relu'))(conv11)
  dense1 = Dropout(dense_layer_dropout_rate)(dense1)
  dense1 = attention_module(dense1, pos_ids)

  main_output = TimeDistributed(Dense(units=8, activation='softmax', name='main_output'))(dense1)

  model = Model([pssm_input, seq_input, pos_ids], main_output)
  return model

# end model.py




###................. load and process CB5916 dataset for training .............###

cb6133 = load_gz(train_dataset_dir)

dataindex = list(range(35, 56))
labelindex = range(22, 30)
'''
total_proteins = cb6133.shape[0]
cb6133 = np.reshape(cb6133, (total_proteins, 700, 57))

cb6133_protein_one_hot = cb6133[:, :, : 21]
cb6133_protein_one_hot_with_noseq = cb6133[:, :, : 22]
print('cb6133_protein_one_hot_with_noseq.shape: '+str(cb6133_protein_one_hot_with_noseq.shape))
lengths = np.sum(np.sum(cb6133_protein_one_hot, axis=2), axis=1).astype(int)

traindata = cb6133[:, :, dataindex]
trainlabel = cb6133[:, :, labelindex]
lengths_train = lengths[:]

cb6133_seq = np.zeros((cb6133_protein_one_hot_with_noseq.shape[0],cb6133_protein_one_hot_with_noseq.shape[1]))
for j in range(cb6133_protein_one_hot_with_noseq.shape[0]):
    for i in range(cb6133_protein_one_hot_with_noseq.shape[1]):
        datum = cb6133_protein_one_hot_with_noseq[j][i]
        #print('index: %d' % i)
        #print('encoded datum: %s' % datum)
        cb6133_seq[j][i] = int(decode(datum))
        #print('decoded datum: %s' % cb6133_seq[j][i])
        #print()
    #break

positions_seq = np.array(range(700))
positions_seq = np.repeat([positions_seq],total_proteins, axis=0)

## freeup some memory of RAM
del cb6133
del cb6133_protein_one_hot
del lengths
gc.collect()
'''

total_proteins = cb6133.shape[0]
cb6133 = np.reshape(cb6133, (total_proteins, 700, 57))
feature_index = np.hstack((np.arange(0, 22), np.arange(35, 56)))
print('len(np.arange(0, 22)): '+str(len(np.arange(0, 22))))
train_data = cb6133[:, :, feature_index]
trainlabel = cb6133[:, :, labelindex]

seq_shuffle = np.arange(0, total_proteins)
np.random.shuffle(seq_shuffle)
train_data = train_data[seq_shuffle[0:4869]]
trainlabel = trainlabel[seq_shuffle[0:4869]]
print('train_data.shape: '+str(train_data.shape))
print('trainlabel.shape: '+str(trainlabel.shape))

################################################################
"""validation data processing"""
################################################################
valid_data = cb6133[:, :, feature_index]
valid_label = cb6133[:, :, labelindex]

valid_data = valid_data[seq_shuffle[4869:5365]]
valid_label = valid_label[seq_shuffle[4869:5365]]

print('valid_data.shape: '+str(valid_data.shape))
print('valid_label.shape: '+str(valid_label.shape))

positions_seq = np.array(range(700))
positions_seq = np.repeat([positions_seq],4869, axis=0)

valid_pos = np.array(range(700))
valid_pos = np.repeat([valid_pos],496, axis=0)

lengths_valid = np.sum(np.sum(valid_data[ :, :, 0:21], axis=2), axis=1).astype(int)
weight_mask_valid = np.zeros((valid_data.shape[0], 700))
for i in range(len(lengths_valid)):
  weight_mask_valid[i, : lengths_valid[i]] = 1.0

lengths_train = np.sum(np.sum(train_data[ :, :, 0:21], axis=2), axis=1).astype(int)

weight_mask_train = np.zeros((train_data.shape[0], 700))
for i in range(len(lengths_train)):
  weight_mask_train[i, : lengths_train[i]] = 1.0


'''
###............... load and process CB513 to validate ..............###

cb513 = load_gz("CB513.npy.gz")
cb513 = np.reshape(cb513, (514, 700, 57))
#print(cb513.shape)
x_test = cb513[:,:,dataindex]
y_test = cb513[:,:,labelindex]

print("y_test[1][56]: "+str(y_test[1][56]))

cb513_protein_one_hot = cb513[:, :, : 21]
cb513_protein_one_hot_with_noseq = cb513[:, :, : 22]
lengths_cb = np.sum(np.sum(cb513_protein_one_hot, axis=2), axis=1).astype(int)
print("lenght_cb: "+str(lengths_cb))
#print(cb513_protein_one_hot_with_noseq.shape)
del cb513_protein_one_hot
gc.collect()

cb513_seq = np.zeros((cb513_protein_one_hot_with_noseq.shape[0],cb513_protein_one_hot_with_noseq.shape[1]))
for j in range(cb513_protein_one_hot_with_noseq.shape[0]):
    for i in range(cb513_protein_one_hot_with_noseq.shape[1]):
        datum = cb513_protein_one_hot_with_noseq[j][i]
        cb513_seq[j][i] = int(decode(datum))

cb513_pos = np.array(range(700))
cb513_pos = np.repeat([cb513_pos],514, axis=0)

###............. generate weight masks ..........###
weight_mask_cb513 = np.zeros((x_test.shape[0], 700))
for i in range(len(lengths_cb)):
  weight_mask_cb513[i, : lengths_cb[i]] = 1.0

weight_mask_train = np.zeros((traindata.shape[0], 700))
for i in range(len(lengths_train)):
  weight_mask_train[i, : lengths_train[i]] = 1.0

print("weight_mask_cb513 shape : "+str(weight_mask_cb513.shape))
print("weight_mask_cb513: "+str(weight_mask_cb513[0]))
'''

###................ generate the model...........###
model = generate_model()

###............. generate necessary callbacks ..........###

class CollectOutputAndTarget(tf.keras.callbacks.Callback):

    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
    
    def on_epoch_end(self, epoch, logs={}):
        x,y = self.test_data
        y_pred_epoch = self.model.predict(x)
        acc_epoch = Q8_accuracy(y, y_pred_epoch)
        m = tf.keras.metrics.Mean()
        m.update_state(acc_epoch)
        acc_epoch2 =m.result().numpy()
        print ('accuracy:', acc_epoch2)
        print(epoch)
        if np.less_equal(acc_epoch2, 0.3) and epoch > 0:
          print('accuracy epoch:', acc_epoch2)
          one_hot_predictions = probabilities_to_onehot(y_pred_epoch)
          #print(Q8_accuracy(s_struct_test, one_hot_predictions).mean())
          yt = to_int_seq(y, y)
          print("s_struct_test.shape: "+str(y.shape))
          print("one_hot_predictions.shape: "+str(one_hot_predictions.shape))
          yp = to_int_seq(y, one_hot_predictions)
          yt = list(map(str, yt))
          yp = list(map(str, yp))
          print(make_confusion_matrix(yt, yp))


best_model_file = best_model_file
checkpoint = callbacks.ModelCheckpoint(best_model_file, monitor='val_truncated_accuracy', verbose=1, save_best_only=True, mode='max')
#lr_decay = callbacks.LearningRateScheduler(StepDecay(initAlpha=0.0005, factor=0.9, dropEvery=60))
cbk = CollectOutputAndTarget(model, ([valid_data[:,:,22:43], valid_data[:,:,0:22], valid_pos], valid_label))

adam = Adam(lr=lr)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              sample_weight_mode='temporal',
              metrics=[Q8_accuracy, 'mae', truncated_accuracy])

import matplotlib.pyplot as plt
### plot model structure ###
from keras.utils.vis_utils import plot_model
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        
###........... train ...............###

'''
history = model.fit(x=[traindata, cb6133_protein_one_hot_with_noseq, positions_seq], y=trainlabel,
        batch_size=32,
        epochs=100, 
        validation_data=([x_test, cb513_protein_one_hot_with_noseq, cb513_pos], y_test, weight_mask_cb513),
        shuffle=True,
        sample_weight=weight_mask_train,
        callbacks=[checkpoint], verbose=1)
'''
history = model.fit(x=[train_data[:,:,22:43], train_data[:,:,0:22], positions_seq], y=trainlabel,
        batch_size=16,
        epochs=200, 
        validation_data=([valid_data[:,:,22:43], valid_data[:,:,0:22], valid_pos], valid_label, weight_mask_valid),
        shuffle=True,
        sample_weight=weight_mask_train,
        callbacks=[checkpoint, cbk], verbose=1)
  
plot_graph(history)
