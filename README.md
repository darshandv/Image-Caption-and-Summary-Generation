# Image Caption and Summary Generation

The project aims at generating summary or caption for a given image.

There are 2 important parts of the project. The first part is the image model where the model architecture outputs the image features as non-linear activations of pixel values of the image.
<br/>
These values are fed into the second part i.e. the language model which generates the summary sentence based on the output of the image model.

The dataset used for the project is MSCOCO dataset.
You can download the trianing dataset by clicking [here](http://images.cocodataset.org/zips/train2017.zip) (Please note that the dataset is around 19.3 GB).
<br/>
You also need the captions for them and they can be downloaded by clicking [here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).
<br/>
Both of them are downloaded from the official cocodataset website.

You need to extract them and keep them ready before starting.


We have also used Python API for using coco dataset. Please install by following the procedures mentioned for PythonAPI from [here](https://github.com/cocodataset/cocoapi)
### Progress
To save the training time the activations from image model is generated pre hand because we are using trained model as an image model and not meddling with the weights of the model.

So after cloning the project run -
``` python
python image_model_activations.py
```
This generates a file containing all the activations.

Then to train the model run -
``` python
python train.py
```
To generate captions run -
``` python
python generate_caption.py
```

Some of the results from this model :
![Image 1](/images/1.png)
---
![Image 2](/images/2.png)
---
![Image 3](/images/3.png)
---
![Image 5](/images/5.png)
---
![Image 6](/images/6.png)
---
![Image 7](/images/7.png)
---
![Image 8](/images/8.png)
---
![Image 9](/images/9.png)
