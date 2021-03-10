# **Traffic Sign Recognition** 

## Writeup Report

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/dataset_exploration.png "Dataset Exploration"
[image4]: ./German_Traffic_Sign/original/image1.png "Traffic Sign 1"
[image5]: ./German_Traffic_Sign/original/image2.png "Traffic Sign 2"
[image6]: ./German_Traffic_Sign/original/image3.png "Traffic Sign 3"
[image7]: ./German_Traffic_Sign/original/image4.png "Traffic Sign 4"
[image8]: ./German_Traffic_Sign/original/image5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Python to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. It is a bar chart showing the frequency of each classID (traffic sign lebel).

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to normalized the image data because I'd like to benefit from **numerical stability**


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		    |     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x3 RGB image   							| 
|                		    |                      							| 
| Convolution (Layer1) 5x5  | 1x1 stride, 'VALID' padding, outputs 28x28x6 	|
| RELU					    |												|
| Max pooling	      	    | 2x2 stride,  outputs 14x14x6  				|
|           	      	    |                                 				|
| Convolution (Layer2) 5x5  | 1x1 stride, 'VALID' padding, outputs 10x10x6	|
| RELU					    |												|
| Max pooling	      	    | 2x2 stride,  outputs 5x5x16  				    |
| Flatten       	        | Flatten the inputs(5x5x16) and outputs 400    |
|						    |												|
| Fully connected (Layer3)  | input 400 and outputs 120        				|
| RELU					    |												|
| Dropout					| Keep probability : 0.5 (50%)					|
|						    |												|
| Fully connected (Layer4)  | input 120 and outputs 84        				|
| RELU					    |												|
| Dropout					| Keep probability : 0.5 (50%)					|
|						    |												|
| Fully connected (Layer5)  | input 84 and outputs 43        				|
|						    |												|
|						    |												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used: 
- learning rate being 0.001
- optimizer type being AdamOptimizer (Adam algorithm is similar to stochastic gradient descent)
- cross_entropy to calculate the loss
- batch size 128
- 50 epochs

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of **0.965**
* test set accuracy of **0.944**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?  
I used the architecture introduced in the Lenet video because it's the default one that I could start with finetuning.

* What were some problems with the initial architecture?  
The final accuracy is not good enough even I increse the number of epochs

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.  
I added the **dropout** technique in fully connected layer 3 and layer 4 to adjust the architecture and increase the accuracy.I think the reason is that I avoid the over-fitting problem via taking the regularization approach.

* Which parameters were tuned? How were they adjusted and why?  
I used **0.5** as the parameter in the dropout function. It's just the default value I used and luckly it works.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  
The convolution layer helps to filter the image input and output tthe feature that we need, so that even the same target  apperas in the different part of the image, the model can still identify them.  
The dropout function helps to create a successful model by avoiding over-fitting porblem.

If a well known architecture was chosen:
* What architecture was chosen?  
Lenet-5

* Why did you believe it would be relevant to the traffic sign application?  
It's suggested by the course, and it could take a 32x32 image as an input

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
Validation accuracy : **0.965**  
Test accuracy : **0.944**
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                |     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Roundabout mandatory          | Roundabout mandatory							| 
| Priority road 		        | Priority road									|
| No entry				        | No entry										|
| Stop  	      		        | Stop      					 				|
| Dangerous curve to the right	| Dangerous curve to the right					|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100.0%. This compares favorably to the accuracy on the test set of 94.4%, however, it might change if I choose different images to test.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a **Roundabout mandatory** (probability of 1.0), and the image does contain a **Roundabout mandatory** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| 100.0 		        | Roundabout mandatory   						    | 
| 2.52225866061e-11		| Right-of-way at the next intersection 		    |
| 1.1031169603e-11		| Vehicles over 3.5 metric tons prohibited		    |
| 4.176017442e-12		| End of no passing by vehicles over 3.5 metric tons|
| 4.04604303146e-13	    | Speed limit (100km/h)							    |


For the second image, the model is relatively sure that this is a **Priority road** (probability of 1.0), and the image does contain a **Priority road** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| 100.0 		        | Priority road           						    | 
| 0.0		            | Speed limit (20km/h)                   		    |
| 0.0		            | Speed limit (30km/h)                  		    |
| 0.0		            | Speed limit (50km/h)                              |
| 0.0           	    | Speed limit (60km/h)  						    |


For the third image, the model is relatively sure that this is a **No entry** (probability of 1.0), and the image does contain a **No entry** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| 100.0 		        | No entry                 						    | 
| 9.50551673926e-22		| Stop                                   		    |
| 1.01594273188e-33		| Traffic signals                       		    |
| 3.6715051121e-36		| Priority road                                     |
| 0.0           	    | Speed limit (20km/h)							    |


For the fourth image, the model is relatively sure that this is a **Stop** (probability of 1.0), and the image does contain a **Stop** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| 100.0 		        | Stop                  						    | 
| 1.20499676903e-07		| No entry                                		    |
| 2.35014136244e-12		| Speed limit (60km/h)                     		    |
| 1.97914657887e-1		| Priority road                                     |
| 2.62692632664e-17	    | Turn left ahead   							    |


For the fifth image, the model is relatively sure that this is a **Dangerous curve to the right** (probability of 1.0), and the image does contain a **Dangerous curve to the right** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| 100.0 		        | Dangerous curve to the right					    | 
| 9.48774004782e-12		| No passing                               		    |
| 5.54056532777e-14		| End of no passing                     		    |
| 4.15247923285e-14		| Double curve                                      |
| 2.80556138761e-15	    | Ahead only           							    |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


