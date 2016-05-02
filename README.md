# cs295-gender-determinator-using-fingerprints
A simple python application which predicts the gender of the fingerprint owner from fingerprint image using trained Convolutional Neural Networks (CNN). A Mini Project requirement for CS 295 course at University of the Philippines Diliman AY 2015-2016 under Sir Prospero Naval.

## Types of CNN Architectures Trained
* LeNet
* AlexNet
* Maxout Network 
* Exponential Linear Unit Network

## Requirements
* [NIST Special Database 4] (http://www.nist.gov/srd/nistsd4.cfm) used as dataset
* [CAFFE Deep Learning Framework] (http://caffe.berkeleyvision.org/) for training and classification
* [OpenCV](http://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html#gsc.tab=0) Computer Vision Library
* Python 2.7.11


## Instructions
1. Clone and download the repository.
2. Train using the Nist Special Database 4 database with the model files provided.
3. Update the script with trained caffe models
3. Run the application

  ```  
  python gender-determinator.py
  ```
