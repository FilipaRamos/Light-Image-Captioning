# Light-Image-Captioning

Welcome to Light-Image-Captioning! Here you can find my research as I dive into the world of Image Captioning.

Available models:

1. Baseline Model
    - Image feature extraction with VGG19 plus a simple LSTM cell block for language generation (Inception V3 also available)
2. Transformer Model
    - Image feature extraction with ResNet101 plus a Transformer architecture that receives image features as input to the encoder and generates the caption through the decoder

## Requirements

Tested on a machine with:

- Intel i7-8700
- 16GB RAM
- 1x Nvidia GTX 1080 Ti

With the following software configuration:

- Ubuntu 20.04
- CUDA 11.5
- cuDNN 8.2

#### Versions

In order to consult the versions of all packages that were available in my environment, see `requirements.txt`. In case you don't want to go through the entire requirements, the following list represents the most important packages:

- tensorflow (-gpu) (v2.7.0)
- keras (v2.7.0)
- nltk (v3.6.5)
- pytorch (v1.10.1)
- torchvision (v0.11.2)

## Directory

We developed both pytorch (transformers) and tensorflow (transformers_tf) versions for the transformer. The LSTM architecture was fully developed with Keras and Tensorflow and can be found ontransfermers_tf/model.py. All reported results are from the PyTorch version.

The directory structure is as follows:

```
light-image-captioning
│   README.md
|   .gitignore
|   requirements.txt
|   utils.py
|   data_preparation.py
|   feature_extraction.py
|   prepare_input.py
|   
└───config
|   │   flickr_resnet_transformer.cfg
|   │   ...
|   │   
|  
└───data
│   |   features.pkl
|   |   ...
|   |
│   
└───data-coco
|   │   caption_dataset
|   |   |   dataset_coco.json
|   |   |
|   |
|   │   gen_data
|   |   └─── ...
|   |   train2014
|   |   └─── ...
|   |   val2014
|   |   └─── ...
|   │   
|  
└───transformers
|   │   metrics
|   |   |   
|   |   └─── cider
|   |   |   |
|   |   |   |   __init__.py
|   |   |   |   cider_scorer.py
|   |   |   |   cider.py
|   |   └─── tmp
|   |   |   |   ...
|   |   |   |
|   |   dataset.py
|   |   train.py
|   |   eval.py
|   |   utils.py
|   |   simple_transformer.py
|   |   transformer.py
|   |   README.md
|  
└───transformers_tf
|   │   data_generator.py
|   |   demo.py
|   |   eval.py
|   |   layers.py
|   |   model.py
|   |   train.py
|   |   README.md 
```

## Data Preparation

In order to prepare the data for testing the models, the Flickr 8k dataset must be downloaded into `data`. Download [Flick8k_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) and [Flickr8k_Text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip). Extract the zips in the `data` folder.

For COCO, we keep a separate folder named `data-coco`. Download and extract [COCO_Train](http://images.cocodataset.org/zips/train2014.zip), [COCO_Val](http://images.cocodataset.org/zips/val2014.zip) to the aforementioned folder. For the karpathy splits, download [Karpathy Splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and extract into `data-coco/caption_dataset`.

And we are done! Refer to each folder's README for further instructions.