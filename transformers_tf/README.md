## Transformer tf

This code has been built with Keras using a Tensorflow backend. The models are very light in both CPU and RAM. 
In order to save RAM when training, we pre-compute image features using the backbone object detector and save them to a pkl file. Run the following command to compute features:

```
python3 feature_extraction.py [config] # If no config is specified, defaults to flickr_inception_transformer
python3 feature_extraction.py flickr_inception_transformer # Notice that the .cfg is not needed
```

# Execution

```
cd transformers_tf
python3 train.py
```

NOTE: this transformer code currently contains a GPU silent error which I have been unable to resolve.