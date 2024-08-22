# Deepfake-Detector
## Group Members

1. OLIVE MULOMA
2. ABIGAIL MWENDWA
3. HAWKINS MURITHI
4. HARRY ATULAH

## Overview
With the advancement of AI technologies, distinguishing between real and AI-generated images has become increasingly challenging. This project aims to develop an advanced deepfake detection system by leveraging cutting-edge machine learning techniques. The system is designed to analyze image data, identify signs of manipulation, and provide a reliable tool for media verification. The development process involves several key stages, including data collection and preprocessing, model development, feature extraction, evaluation, and deployment.

## BUSINESS UNDERSTANDING

### Stakeholders
- Media and Content Verification Teams: These teams will benefit from improved tools for authenticating images, enhancing the credibility of visual content.
- General Public: Enhanced detection models will help maintain the integrity of visual media, contributing to a more trustworthy information landscape.
- Content Moderation Teams: These teams will implement the detection models to filter and manage visual content, ensuring the authenticity of images shared across platforms.
  
### Business Objectives
- Develop an Accurate Detection Model: Build a machine learning model using advanced deep learning techniques (CNNs, RNNs, and transfer learning) to distinguish between real and manipulated media with high precision.
- Ensure Robust Performance: Train the model on a diverse dataset to ensure consistent performance across various deepfake techniques, adapting to new manipulation methods over time.
- Enhance Usability: Create a user-friendly interface, such as a web application or API, allowing easy verification of media authenticity with clear, actionable feedback for a broad audience.

### Business Problem
Deepfake technology has rapidly advanced, enabling the creation of highly convincing manipulated media that can mislead audiences, spread misinformation, and pose significant threats to personal and organizational security. The challenge lies in developing a sophisticated system capable of detecting these deepfakes and ensuring the authenticity of digital content. This project seeks to address this challenge by creating a comprehensive solution that can accurately identify manipulated media and distinguish it from genuine content.

## Data Understanding

### Dataset 
The CIFAKE dataset, designed to evaluate the detection of AI-generated images, consists of 120,000 images split evenly between REAL and FAKE categories. REAL images are sourced from the CIFAR-10 dataset, while FAKE images are generated using Stable Diffusion. The dataset includes 100,000 images for training and 20,000 for testing. In our project, we leverage this dataset to train and assess computer vision models aimed at distinguishing between real and synthetic images. To efficiently manage and analyze the dataset, we use dataframes for organization and processing

### Data Analysis
The data analysis phase focused on understanding the CIFAKE dataset, which contains 60,000 real images from CIFAR-10 and 60,000 AI-generated images created with Stable Diffusion. Key steps included data exploration to identify the distribution of classes and ensure data quality. Visualizations were used to inspect image patterns and differences between real and synthetic images, guiding preprocessing and feature extraction strategies.

### Data Cleaning

In the data cleaning phase, we aimed to ensure the quality and integrity of our image dataset by addressing several key issues. We began by removing duplicate images using hashing techniques, which allowed us to identify and drop identical images effectively. Next, we checked for any missing labels and eliminated rows with incomplete data to maintain the dataset's consistency. We also verified the integrity of each image to detect and remove any corrupted files that could potentially disrupt the model's training process. This meticulous cleaning process helped us create a reliable and accurate dataset, ready for model development. The cleaned data is now stored in dataframes, ensuring a structured and organized format for further analysis and modeling.

## Visualizations
a) Distribution of image sizes

![image](https://github.com/olivemideva/Deepfake-Detector/blob/main/uploads/Distribution%20of%20image%20sizes.png)

b) Class distribution

![image](https://github.com/olivemideva/Deepfake-Detector/blob/main/uploads/Class%20distribution.png)

c) Average Pixel Intensity Distribution

![image](https://github.com/olivemideva/Deepfake-Detector/blob/main/uploads/pixel%20intensity%20distribution.png)

d) Aspect ratio Distribution

![image](https://github.com/olivemideva/Deepfake-Detector/blob/main/uploads/Aspect%20ratio%20distribution.png)

e) Sample Images

![image](https://github.com/olivemideva/Deepfake-Detector/blob/main/uploads/sample%20images.png)

## Modeliing 
- Convolutional Neural Networks (CNNs): Employed to automatically extract features and classify images based on learned patterns. Transfer learning was considered, utilizing pre-trained models like ResNet and VGG16 to enhance accuracy.
- Ensemble Methods: Combined multiple CNN models to improve robustness and overall classification performance.
- Custom Architectures: Designed a tailored CNN architecture to handle the unique properties of the CIFAKE dataset, such as its small image size and the subtle differences between real and synthetic classes.


## Evaluation
- Accuracy: The model achieved an overall accuracy of 94% in classifying real and AI-generated images.
- Precision and Recall: Both were high, leading to an F1-score of 94% for both classes.
- F1 Score: A slight bias was observed towards predicting AI-generated images with higher recall (97%) compared to real images (91%).
Robustness Testing: Extensive testing was conducted on a separate test set to ensure the models generalized well across different data subsets and potential variations in AI-generated images.

## Deployment
The front end of the application is built using Flask, a lightweight web framework for Python. 
This Flask app serves as the interface where users can upload images to determine whether they are real or deepfake. 
The app loads a pre-trained CNN model and processes incoming images by resizing and normalizing them before passing them through the model. The results, indicating whether the image is "Real" or "Fake," are returned to the user in JSON format. 
The app handles file uploads, ensures only valid image formats are accepted, and can serve the uploaded files back to the user. [Deepfake Detection Application](https://dolphin-app-3umwi.ondigitalocean.app/)

![image](https://github.com/olivemideva/Deepfake-Detector/blob/main/uploads/deployment1.jpg)

![image](https://github.com/olivemideva/Deepfake-Detector/blob/main/uploads/deployment2.jpg)


## Recommendations
1. Implement Verification Protocols: Establish protocols to verify media authenticity before publication. This ensures content accuracy and reduces the risk of spreading fake media.

2. Strengthen Collaboration: Foster collaboration between content creators and verification teams. This improves media verification accuracy and addresses fake content challenges.

3. Enhance Content Moderation: Integrate the deepfake detection model into content moderation systems. This automatically filters out fake images, ensuring only genuine content is published.






