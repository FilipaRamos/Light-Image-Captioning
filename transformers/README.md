## Transformer

This code has been built with PyTorch and Torchvision. Metrics includes the CIDER score from Github user tylin.

# Data Preparation

Since COCO is a much larger dataset, another preparation step must be performed so that, during training, the bottleneck does not become read/write accesses.

As such, the following command must be run on the main directory:

```
python3 prepare_input.py
```

# Execution

```
cd transformers
python3 train.py [flickr/coco]
```

Evaluation is performed after each epoch and results are reported for all respective metrics.