# BlundersPride
### Idea
---------------------------------------------
#### American Sign Language (ASL) to Text
ASL Speaking Community has a huge population of about 250,000-500,000. Of all the people who are deaf and mute, only 10% of their family and friends can understand sign language. This application aims at improving this percentage considerably.
It's as simple as that. The decription and future scope section below provides a good idea of how this project will expand and hopefully it will.

Right now we're able to take in live video feed and predict the number from its sign in ASL. With the advance TPUs/GPUs provided by google would considerably decrease our training time and provide us a better working model.

### Tools and Technologies Used
------------------------------------------------------------------------------
* Python3
    * Keras (Tensorflow Backend)
    * Open-CV (For preprocessing images, Accessing the webcam and Writing out the sequence)
* Used CNN - Convolution Neural Network for training the data-set.
    * 3-groups: (each) 1 Conv2D Layer, MaxPooling Layer and Dropout 
    * Flatten Layer
    * 2 groups: 1 Fully Connected Layer, Dropout
* Dataset
    * Prime Source: Kaggle
    
### How does it work ?
------------------------------------------------------------------------------
* Setting up the environment and downloading all dependencies:
```
$ conda create --name <envname> --file requirements.txt
```
* After this simply run LiveVideoFeed.py using python.
* Place your hand in the blue box to get the predictions.
* Currently the sequence of prediction is shown in another window. 
* The video frame just shows the current prediction.
* Deployment soon (in a few weeks hopefully).

### Description and Future Scope
------------------------------------------------------------------------------
* This application can be further extended to predict actions in ASL as currently it is working only for numbers.
* We can also add this as a plugin to softwares like Skype so that mute people can easily use it to communicate and express their thoughts to other non-ASL speaking people.
* Add this to any video streaming service as a (in crude terms)subtitle for ASL Speakers would be ideal.
------------------------------------------------------------------------------
### ScreenShot
-------------------------------------------------------------------------------
![Three](/Screenshot.png)
### Copyright Information
-------------------------------------------------------------------------------
[License](https://github.com/hackabit18/BlundersPride/blob/master/LICENSE)
