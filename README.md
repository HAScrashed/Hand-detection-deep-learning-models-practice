# About
## Created by Bartosz Sorek & Robert Pintera
### General information
The project constinst of multiple Jupyter Notebooks with Python code with **results and code** of creating own deep learning models. More comments are in notebooks. 
- 6_finger_numbers_recognition has model created by Bartosz Sorek and trained on kaggle dataset [Finger Digits 0-5](https://www.kaggle.com/datasets/roshea6/finger-digits-05). It's capable of recognising 6 gestures basing on black and white images regardless of image rotation.
- 20_gestures_recognition has model created by Bartosz Sorek and trained on kaggle dataset [Hand Gesture Recognition Dataset](https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset). It's capable of recognising 20 gestures basing on black and white images regardless of image rotation.
- CombinedModels is combination of trained model 6_finger_numbers_recognition by Bartosz Sorek and hand mask creation model (segmentation) created by Robert Pintera. By adding additional code and own layer, models can connect and use advantage of both types.
### Media
- **Classification results**
  
- **Combined model schema**
  
- **Combined model output**
  
### Details (snippets from the raport)
#### Purpose of the project
The main goal of the project was the AI creation of hand image grayscale masks based on full-color hand images and using such data for further applications. Investment in the project has expanded our knowledge in the field of deep learning.
#### Problem definition
Modern technologies increase the range of possibilities for interaction with various devices without the need to use physical controllers. Using machine learning and deep learning algorithms, inter alia it is possible to detect and interpret hand and finger gestures.
These technologies can be helpful in different areas of life. They can be applied in virtual reality (VR) or augmented reality (AR) by tracking hand and detecting relevant movements that can be useful for performing an appropriate operation. Hand detection can also be beneficial for deaf people who use sing language to communicate with others. These technologies can be used to ease interaction with devices for invalids.
Main focus was on preprocessing and examining still images displaying a hand in various positions. Models made by us are able to create image masks allowing for hand separation from the rest of the image. Thanks to their universality, they are a proper entry for more complex deep learning models responsible for hand pose estimation and key-point tracking. To present effectiveness and usage of such, we have created and trained classification models capable of interpreting hand gestures and predicting hand rotation in 2D space. Such types of models combined produce the powerful all-in-one tool.
#### Models’ design
For detecting hand masks, the DeepLabV3+ and Unet models were chosen. Both are suitable for solving this problem.
DeepLabV3+ is an advanced encoder-decoder based segmentation model that is often used in tasks requiring high accuracy. Being an improved version of DeepLabV3, it is characterized by:
- Atrous Spatial Pyramid Pooling (ASPP) - the model using it allows to analyze features on different scales, which helps to segment hands of different sizes and shapes.
-	Decoder - introduces an additional decoder module that helps in better reconstruction of objects and image details.
-	ResNet - Using the advanced network as feature extractors makes the model better able to handle difficult cases.
Another model used for image segmentation is Unet. The network is based on a convolutional neural network, whose architecture consists of an encoder and a decoder. The symmetrical design and skip connection (Connecting convolution layers from an encoder with up-sampling from a decoder) of the model work well in reconstructing details, which improves the accuracy of segmentation mask detection.


Two databases - finger-digits-05 and hand-gesture-recognition-dataset were used for the classification task. They both share similar model schema – convolutional layers for detecting important features that are connecting to attention block before dense layers with dropouts in between. Attention block uses Multi-Head Attention mechanism and layer normalization improving general accuracy by providing additional support in finding the most important image features. 
Since all training samples were always centered and had the same rotation, models were given ImageDataGenerator for creating unique augmented samples each epoch with random rotation (up to 80 degrees), shift (vertical and horizontal up to 20%), shear (up to 10%) and even mirrored versions in both axes. This greatly improved generalization. 
Using attention block increased accuracies of both models by around 3%. The first model is capable of detecting 6 numbers from a hand gesture – 0, 1, 2, 3, 4, 5 with train set accuracy of 100%, validation set accuracy of 100%. The second model can predict 20 gestures with train set accuracy of 98%, 99% validation set, 99% test set. 
Models use categorical cross-entropy and y sets are one-hot encoded for correct classification. Outputs have Softmax activation function, and the best value can be chosen. The rest of the layers use ReLU activation function because of its speed and efficient calculations.
#### Model combination
Trained models - mask creation model with the gesture classification were combined into one big model. It was performed on Unet structured model and 6-gesture classification model. To combine models, there was a need for an additional layer, so it has been created. Self-made BoundingBoxCropLayer is capable of cropping full-size generated mask image to just mask pixels, converting pixels to binary format and rescaling results to input dimensions of the second model. Additionally, it saves information if the mask has any non-zero pixels.


