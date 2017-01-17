Udacity Self-Driving Car Engineer: Project 3 - Behavioral Cloning
===================

Approach
---
In order to establish some ground-truth, I started out with only the Udacity data, wanting to get the car drive around the track at least once with the published baseline data. Once I achieved that, I added my own data and gradually introduced more preprocessing steps as outlined below in order to improve two things:

- Smoothness of the drive
- Prevent overfitting, defined by good performance on Track 2

I used the established nVidia and commaAI models on scaled down versions of the training set (64x64x3).

Data Preprocessing:
---
I added a few steps to augment the recorded data (in order):

- **Steering angles close to zero (straight steering) are ignored** according to a given probability
***Reason:*** Recording of a car going around Track 1 will be skewed towards going straight since much of the track doesn’t require steering input.
***Implementation:*** Steering angles close to zero are ignored according to a given probability. I arrived at a factor of 0.5 at the end.
***Result:*** Less erratic driving
- **Flipped data**
***Reason:*** Recording of a car going around Track 1 will be skewed towards making left turns.
Implementation: All images in the dataset are flipped and the corresponding steering angle is inverted
***Result:*** Less erratic driving
- **Use of stereo images**
***Reason:*** Just an experiment, but it greatly improved the performance of my model
Implementation: Tried offsets between 0.05 and 0.5. Greater values improved the driving performance of the car. Values below 0.2 didn’t have as much of an impact, while getting closer to 0.5 caused the car’s driving to become very jittery with lots of over-correcting. I settled on 0.25, which was a surprisingly high value.
***Result:*** Much less erratic driving
- **Cropping**
I cropped a little more than the top fifth of the image as well as the bottom to remove the car’s hood from the image. This yielded a generally smoother ride around track 1. I originally cropped one third off the top, but that led to worse performance on Track 2 due to the slopes.
***Result:*** Better generalization
- **Scaling**
Images were scaled to 64x64 pixels. This seemed safe to do since the game creates clean, synthetic images with little complexity to begin with.
***Result:*** Much faster training times
- **Randomize brightness and add artificial shadows**
***Reason:*** My model would tend to swerve around shaded areas, clearly identifying them as hard road boundaries. It would also start behaving unpredictably when entering and exiting entirely dark areas of Track 2.
Implementation: Brightness of the entire image is scaled randomly between 0.25 and 1.25.
Additionally, artificial “shadows” are added simply as a rectangle of darkness that is blended into the original image. Shadows in the game world are fairly simple geometrically, and sticking to an even simpler rectangle shape helped the model enough to be less sensitive to shadows.
***Result:*** Car drives more predictably through shady areas. After this augmentation, my model started being able to navigate Track 2 in it’s entirety.

(sample images)

Data normalization
---
Carried out in the first layer of the Keras model and simply scales the RGB values to be between -1.0 and 1.0.

Training Data
---
I created a lot of my own data, including smooth laps around the track as well as recording individual recoveries from bad states to the right and left of the track - all at reduced speed in order to make sure the entire track is evenly sampled.
To make sure my PC writes all images to disk without hiccups, I recorded at the lowest resolution the simulator allowed at “good” graphics settings.
After I achieved reasonable driving results with the Udacity data, I added my own datasets to the model to polish the driving behavior.

Model Architecture and Training Strategy
---
I tried out two existing architectures to tackle this project:
Implemented **nVidia** model based on:
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
Used **commaAI** model from:
https://github.com/commaai/research/blob/master/train_steering_model.py

For both, I reduced kernel sizes to be 3x3 at most due to my much smaller input images.
From the beginning, I got more reliable results from commaAi. The nVidia model would sometimes turn “stale” at higher Epoch counts. Returning only one steering value for any given input.

I tested two approaches for my **training data generator**.

1) Per epoch, pass exactly one sample of each element in the training set to the model then reset and move on to next epoch
2) Pass a randomly chosen element of the training set to the model continuously and play with the number of samples per epoch

I got better, more reliable training results from approach 2. It made even more sense to stick to the random generator after adding preprocessing steps that are based on randomness (brightness and shadows) - the images are now somewhat unique regardless of whether the generator sends on duplicates or not.

I am training on 20.000 samples per epoch with a batch size of 64.

Drive.py changes
---
I am calling my model’s preprocessing function. In it’s default configuration, it only resizes the images to the model’s resolution.
I also increased the default speed to 0.3 in order to make it through Track 2.

Future Improvement
---
- Higher sampling rate for autonomous mode. This would probably smooth out the drive at top speed quite a bit.
- Smooth out steering angle in drive.py over one or two frames to reduce remaining input peaks.
- Large steering inputs should cause a reduction in throttle to prevent overly aggressive turns.