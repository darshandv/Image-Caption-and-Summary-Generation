from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Embedding
from tensorflow.python.keras.layers.recurrent import GRU
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras import backend as K

# from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import tensorflow as tf
import math

from utils import *


dataDir='.'
dataType='train2017'

batch_size = 128
embedding_size = 256
cell_state_size = 512
epoch_start = 0
epoch_end = 20
num_words = 15000

capsAnnFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)

activation_file = h5py.File('vgg_activations.h5','r')
vgg_activations = activation_file['last_layer_activations'][:]

start = 'ssstrt '
end = ' eenddd'

captions = getCaptions(file = capsAnnFile)
captions_marked = [[str(start)+str(info)+str(end) for info in cap] for cap in captions]
all_caps_train = get_train_captions(captions_marked,cap_all=True)

tokenizer = TokenizerExt(all_caps_train,num_words=num_words,oov_token=num_words+1)
train_tokens = tokenizer.captions_to_tokens(captions_marked)

num_images = vgg_activations.shape[0]
id_map = np.array(range(num_images))

steps = int(len(all_caps_train)/batch_size)
activation_vector_length = vgg_activations.shape[1]
del captions
del captions_marked
del coco_inst
del coco_caps

image_activation_input = Input(shape=(activation_vector_length,),name='img_act_input')

model_map_layer = Dense(cell_state_size,activation='tanh',name='fc_map')(image_activation_input)

lang_model_input = Input(shape=(None,),name="lang_input")
lang_embed = Embedding(input_dim=num_words,output_dim=embedding_size,name='lang_embed')(lang_model_input)

lang_gru1 = GRU(cell_state_size, name='lang_gru1',return_sequences=True)(lang_embed,initial_state=model_map_layer)
lang_gru2 = GRU(cell_state_size, name='lang_gru2',return_sequences=True)(lang_gru1,initial_state=model_map_layer)
lang_gru3 = GRU(cell_state_size, name='lang_gru3',return_sequences=True)(lang_gru2,initial_state=model_map_layer)

lang_out = Dense(num_words,activation='linear',name='lang_out')(lang_gru3)
language_model = Model(inputs=[image_activation_input,lang_model_input],outputs=[lang_out])
	

def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

optimizer = optimizers.RMSprop()
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
language_model.compile(optimizer=optimizer,
                      loss=sparse_cross_entropy
                      ,target_tensors=[decoder_target]
                      )


path_checkpoint = 'model_weights.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      verbose=1,
                                      save_weights_only=True)
callback_tensorboard = TensorBoard(log_dir='./train_logs/',histogram_freq=0,write_graph=False)
callbacks = [callback_checkpoint, callback_tensorboard]

for i in range(epoch_start,epoch_end,1):
    generator = data_generator(train_tokens,id_map,vgg_activations,batch_size=batch_size,single_caption=False)
    steps = int(len(all_caps_train)/batch_size)
    language_model.fit_generator(generator,
                                steps_per_epoch=steps,
                                epochs=i+1,
                                callbacks=callbacks,
                                 initial_epoch = i)
