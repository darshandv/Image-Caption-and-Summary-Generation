from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
import json

# Initialize required data in first few cells
dataDir='.'
dataType='train2017'
capsAnnFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)


# If using pycocotools this funcition is necessary
def getAllIds(coco_inst) :
    cats = coco_inst.loadCats(coco_inst.getCatIds())
    nms = [cat['name'] for cat in cats]
    
    catIds = coco_inst.getCatIds(catNms=[nms])
    imgIds = coco_inst.getImgIds(catIds=catIds)

    return sorted(imgIds)


# def get_captions(imgId,coco_caps):
#     annIds = coco_caps.getAnnIds(imgIds=imgId);
#     anns = coco_caps.loadAnns(annIds)
    
#     return anns

# If using pycocotools this funcition is necessary
def get_captions(id_list,coco_caps):
    captions = []
    for ids in id_list:
        annIds = coco_caps.getAnnIds(imgIds=ids)
        anns = coco_caps.loadAnns(annIds)
        data = [obj['caption'] for obj in anns]
        captions.append(data) 
    
    return captions

def getCaptions(file = capsAnnFile):
    dataset = json.load(open(file, 'r'))
    annotations = dataset['annotations']
    captions_dict = {}
    for node in annotations:
        caption = node['caption']
        id = node['image_id']
        if id not in captions_dict:
            captions_dict[id] = []
        captions_dict[id].append(caption)
    sorted_captions = sorted(captions_dict.items(),key = lambda x:x[0])
    final_captions = []
    for i in sorted_captions:
        final_captions.append(i[1])
    return final_captions

def get_train_captions(captions_marked,i=0,cap_all=False):
    if cap_all:
        return [sent for cap in captions_marked for sent in cap]
    else:
        return [cap[i] for cap in captions_marked]


class TokenizerExt(Tokenizer):
    
    def __init__(self, texts, num_words=None,oov_token=0):
        Tokenizer.__init__(self, num_words=num_words,oov_token=oov_token)
        self.fit_on_texts(texts)
        self.index_to_word = dict(zip(self.word_index.values(),self.word_index.keys()))

    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token] for token in tokens if token != 0]
        text = " ".join(words)

        return text
    
    def captions_to_tokens(self, captions_list):
        if len(captions_list[0])<6:
            tokens = [self.texts_to_sequences(captions) for captions in captions_list]
        else :
            tokens = self.texts_to_sequences(captions_list)
        return tokens


def get_tokens(train_tokens,ids,single_caption=False):
    results = []
    if single_caption:
        for idx in ids:
            rand = np.random.choice(ids)
            cap_tokens = train_tokens[idx]
            results.append(cap_tokens)
    else :
        for idx in ids:
            num_captions = len(train_tokens[idx])
            rand = np.random.choice(num_captions)
            cap_tokens = train_tokens[idx][rand]
            results.append(cap_tokens)

    return results


def data_generator(train_tokens,id_map,vgg_activations,batch_size = 32, single_caption = False):
    np.random.shuffle(id_map)
    
    index = 0
    num_images = vgg_activations.shape[0]
    while True:
        if index>=num_images or num_images-index<batch_size:
            index = 0
            continue
        ids = id_map[index:index+batch_size]
        index = index+batch_size
        
        image_model_activations = vgg_activations[ids]
        
        # if not single_caption:
        #     cap_tokens = get_tokens(train_tokens,ids)
        
        # else :
        #     cap_tokens = get_tokens(train_tokens,ids,single_caption=True)
        
        cap_tokens = get_tokens(train_tokens,ids,single_caption=single_caption)

        num_tokens = [len(t) for t in cap_tokens]
        max_tokens = np.max(num_tokens)
        
        tokens_padded = pad_sequences(cap_tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')

        x_data = [np.array(image_model_activations),np.array(tokens_padded[:,:-1])]
#         y_data = np.expand_dims(tokens_padded[:,1:],axis=-1)
        y_data = tokens_padded[:,1:]
        
        yield (x_data,y_data)



def load_image(path,size=(224,224,)):
    img = Image.open(path)
    img = img.resize(size=size,resample=Image.LANCZOS)
    img = np.array(img)/255.0
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    return img



def generate_captions(path='train2017/000000000009.jpg',max_tokens=30):
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
    
    print(lang_input)
    plt.imshow(img)
    plt.show()
    print(output_text)

