# Human Pose Estimation with Stacked Hourglass Network

## Introduction

----
This repository is a Tensorflow implementation of the [Stack-Hourglass Network](https://arxiv.org/abs/1603.06937). COCO dataset is used for training and evalutation.

## Stacked Hourglass Architecture

----
![Stacked Hourglass module from the paper](/figures/shn_paper.png)
![Stacked Hourglass module](/figures/hourglass_module.png)

## Guide on train/evaluation/demo

----
argparse has not been supported yet. It is recommended to run everything in Google Colab. Notebooks which were used for testing and development process can be found in ` dev` folder. Settings can be adjusted in `configs` folder.

# Train

1. Preparing dataset
    * COCO dataset can be downloaded using `get_data.sh`
    * Dataset is converted to TFRecordDataset.
    * For more details do look at 'gen_tfrecords.ipynb' notebook
2. train
    * Traing takes a very long time if you want the best resources.
    * It is better if you use Colab's GPUs and save the result onto GDrive to be loaded later.
    * Construct `trainer` object with your preferred setting.
    * Available losses : Mean SQUARE ERROR (MSE), WEIGHTED MEAN SQUARE ERROR (weighted_MSE) and Intersection over Union (IoU).
    * For the first training session call `trainer.train()` and for resuming training `trainer.resume_train()`.
    * For more details: 'Train.ipynb'.
    * When finishn training, you can save the model in Tensorflow [SavedModel format](https://www.tensorflow.org/guide/saved_model) in `saved_model` folder for convenience.

## Evaluation

----
* Available metrics: OKS and PCK.
![OKS evaluation](/figures/OKS.png)
![PCK evaluation](/figures/PCK.png)
* For more details: `evaluation.ipynb`.


## Demo

----

* For person detector I use YOLOv5.
* For webcam, I obtain input through browser, therefore there is boilerplate for JavaScript. Prop to [The AI Guy](https://www.youtube.com/watch?v=YjWh7QvVH60).
![Demo image](/demo_images/ballet_detected.png)
![Demo image](/demo_images/walking_detected.png)
* For more details: `demo.ipynb.`

Speical thanks to M. for helping me during the process.


