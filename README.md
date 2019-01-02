# tensorflow-resnet-image-clustering 
Image clustering(classification) using resnet written in Tensorflow.


## Project file structure
```
tensorflow-renset-image-clustering
│
└───src
│   │   model.py                        # ResNet model
│   │   helper.py                       # ResNet model helper functions
│   │   config.py                       # Global constants
│   │   main.py                         # Training or make predicted labels file on Training/Test dataset
│   │   make_labels_true.py             # Make (true labels file, image filenames file, mapped label json file) on Training/Test dataset
│   │   evaluation.py                   # Evaluate two of predicted labels file
│   
└───models                              # Trained models
│   └───[YourDatasetName]
│
└───data                                
│   └───[YourDatasetName]
│       └───training
│       └───test
└───img
│   └───[YourDatasetName]                
│       └───training                    # Images for training
│       └───test                        # Images for test

```

## Training
* in <b>src/config.py</b>, modify these to fit your dataset:<pre><code>
DATASET
_IMAGE_WIDTH
_IMAGE_HEIGHT
_IMAGE_CHANNELS
_NUM_CLASSES
_NUM_IMAGES</pre></code>
* put your training image(<b>filename should contain "[label number]_" as a prefix. e.g) 10047545284_10226223039_0.jpg</b>) dataset in <b>img/[YourDatasetName]/training/</b>
* run <pre><code>python3 src/make_labels_true.py --training</code></pre>
* run <pre><code>python3 src/main.py</code></pre>
possible arguments: <br><br>
`--max_training_epochs` : maximum training epoch<br> 
`--batch_size` : batch size<br>
`--checkpoint_save_epoch` : checkpoint saving frequency. 1 means save checkpoint every epoch.<br>
`--checkpoint_max_to_keep` : how many checkpoint files to maintain.<br> 
`--resnet_size` : resnet size. you can choose one among [50, 101, 152]. The larger resnet size, the more residual block layers.<br><br>
you can modify training hyper paramters in <b>src/config.py</b><pre><code>
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5</pre></code>
 
## Inference(predict) on training dataset
* run <pre><code>python3 src/main.py --training_predict</code></pre>
<b> *`--resnet_size` should be same as resnet size you gave when training.<br>   
predicted labels text file will be located "TRAINING_DATA_DIR" in src/config.py. You can set this output location by giving `--training_predict_output_path`.<br>
predicted labels text filename will include timestamp(e.g. labels_pred_1546239433.txt).<br>

## Inference(predict) on test dataset
* put your test image(.jpg files) dataset in <b>img/test/</b>
* run <pre><code>python3 src/make_labels_true.py</pre></code>
* run <pre><code>python3 src/main.py --test_predict</pre></code>
<b> *`--resnet_size` should be same as resnet size you gave when training.<br>
predicted labels text file will be located "TEST_DATA_DIR" in src/config.py. You can set this output location by giving `--test_predict_output_path`.<br>
predicted labels text filename will include timestamp.(e.g. labels_pred_1546239433.txt).

## Evaluation on training dataset

* run <pre><code>python3 src/evaluation.py --labels_pred1 [labels_pred_file] --labels_pred2 [labels_pred_file]</pre></code>
It will print evaluation score of each predicted labels file.

## Code reference

* Original code: https://github.com/tensorflow/models/tree/master/official/resnet <br>
* License: the Apache License, Version 2.0 (the "License")


## License

[MIT](https://github.com/doublejtoh/tensorflow-resnet-image-clustering/blob/master/LICENSE)
