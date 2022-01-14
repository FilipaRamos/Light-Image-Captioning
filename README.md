# Light-Image-Captioning

Welcome to Light-Image-Captioning! Here you can find my research as I dive into the world of Image Captioning.

Available models:

1. Baseline Model
    - Image feature extraction with VGG19 plus a simple LSTM cell block for language generation
2. Transformer Model
    - Image feature extraction with Inception V3 plus a Transformer architecture that receives image features as input to the encoder and generates the caption through the decoder
3. Variations
    - TODO

## Requirements

This code has been built with Keras using a Tensorflow backend. The models are very light in both CPU and RAM. In order to save RAM during training, image features are pre-computed.

Tested on a machine with:

- Intel i7-8700
- 16GB RAM
- Nvidia GTX 1080 Ti

With the following software:

- Ubuntu 20.04
- CUDA 11.5
- cuDNN 8.2

#### Versions

In order to consult the versions of all packages that were available in my environment, see `requirements.txt`. In case you don't want to go through the entire requirements, the following list represents the most important packages:

- tensorflow (-gpu) (v2.7.0)
- keras (v2.7.0)
- nltk (v3.6.5)

## Data Preparation

In order to prepare the data for testing the models, the Flickr 8k dataset must be downloaded into `data`. Download [Flick8k_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) and [Flickr8k_Text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip). Extract the zips in the `data` folder.

In order to save RAM when training, we pre-compute image features using the backbone object detector and save them to a pkl file. Run the following command to do this:

```
python3 feature_extraction.py [config] # If no config is specified, defaults to flickr_inception_transformer
python3 feature_extraction.py flickr_inception_transformer # Notice that the .cfg is not needed
```

And we are done!

## Runing Training and Validation

The training loop does not run evaluation by default, so these must be carried out individually. Keep in mind that the evaluation uses the checkpoint folder that is passed as argument.

To train:

```
python3 train.py [config]
```