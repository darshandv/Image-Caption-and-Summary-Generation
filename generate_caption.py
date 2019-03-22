from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Embedding
from tensorflow.python.keras.layers.recurrent import GRU

# from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import *

dataDir='.'
dataType='train2017'

embedding_size = 256
cell_state_size = 512
num_words = 15000
activation_vector_length = 4096

def load_image(path,size=(224,224,)):
    img = Image.open(path)
    img = img.resize(size=size,resample=Image.LANCZOS)
    img = np.array(img)/255.0
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    return img


def generate_captions(vgg_model,language_model,tokenizer,path='train2017/000000000025.jpg',max_tokens=30):
    img = load_image(path)
    image_batch = np.expand_dims(img, axis=0)
    activations = vgg_model.predict(image_batch)
    lang_input = np.zeros(shape=(1,max_tokens),dtype=np.int)
    
    token_index = tokenizer.word_index[start.strip()]
    output_text = ''
    count = 0
    
    while token_index != end and count < max_tokens :
        lang_input[0,count] = token_index
        X = [np.array(activations),np.array(lang_input)]
        lang_out = language_model.predict(X)
        one_hot = lang_out[0,count,:]
        token_index = np.argmax(one_hot)
        word = tokenizer.token_to_word(token_index)
        output_text+=" "+str(word)
        count+=1
    
    print('The output is :',output_text,'.')
    # plt.imshow(img)
    # plt.show()

capsAnnFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)

start = 'ssstrt '
end = ' eenddd'

captions = getCaptions(file = capsAnnFile)
captions_marked = [[str(start)+str(info)+str(end) for info in cap] for cap in captions]
all_caps_train = get_train_captions(captions_marked,cap_all=True)

tokenizer = TokenizerExt(all_caps_train,num_words=num_words,oov_token=num_words+1)
train_tokens = tokenizer.captions_to_tokens(captions_marked)
# print('Total words:',len(tokenizer.word_counts))

image_model = VGG16(include_top=True, weights='imagenet')
VGG_last_layer = image_model.get_layer('fc2')
vgg_model = Model(inputs = image_model.input, outputs = VGG_last_layer.output)

image_activation_input = Input(shape=(activation_vector_length,),name='img_act_input')

model_map_layer = Dense(cell_state_size,activation='tanh',name='fc_map')(image_activation_input)

lang_model_input = Input(shape=(None,),name="lang_input")
lang_embed = Embedding(input_dim=num_words,output_dim=embedding_size,name='lang_embed')(lang_model_input)

lang_gru1 = GRU(cell_state_size, name='lang_gru1',return_sequences=True)(lang_embed,initial_state=model_map_layer)
lang_gru2 = GRU(cell_state_size, name='lang_gru2',return_sequences=True)(lang_gru1,initial_state=model_map_layer)
lang_gru3 = GRU(cell_state_size, name='lang_gru3',return_sequences=True)(lang_gru2,initial_state=model_map_layer)

lang_out = Dense(num_words,activation='linear',name='lang_out')(lang_gru3)
language_model = Model(inputs=[image_activation_input,lang_model_input],outputs=[lang_out])

path_checkpoint = 'model_weights.keras'
language_model.load_weights(path_checkpoint)

generate_captions(vgg_model,language_model,tokenizer,path='000000000025.jpg',max_tokens=30)
