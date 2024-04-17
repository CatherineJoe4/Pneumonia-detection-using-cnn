# Pneumonia-detection-using-cnn

## 1. Abstract: 

Pneumonia is a life-threatening infectious disease affecting one or both lungs in humans 
commonly caused by bacteria called Streptococcus pneumoniae. One in three deaths in 
India is caused due to pneumonia as reported by World Health Organization (WHO). 
Chest X-Rays which are used to diagnose pneumonia need expert radiotherapists for 
evaluation. Thus, developing an automatic system for detecting pneumonia would be 
beneficial for treating the disease without any delay, particularly in remote areas. Due to 
the success of deep learning algorithms in analyzing medical images, Convolutional 
Neural Networks (CNNs) have gained much attention for disease classification. We 
analytically determine the optimal CNN model for the purpose. Statistical results 
obtained demonstrates that pretrained CNN models employed along with supervised 
classifier algorithms can be very beneficial in analyzing chest X-ray images, specifically 
to detect Pneumonia. Our web application addresses the critical need for pneumonia 
detection, particularly in regions lacking access to expert radiologists. Utilizing deep 
learning techniques, specifically Convolutional Neural Networks (CNNs), we developed 
a custom CNN architecture for feature extraction and classification of chest X-rays as 
pneumonia or normal. 
## 2. Introduction: 

Pneumonia is a common and potentially serious infection of the lungs that can affect 
people of all ages. It is commonly caused by a bacteria called Streptococcus pneumoniae. 
Diagnosis of pneumonia often involves a combination of clinical evaluation, imaging 
tests such as chest X-rays or CT scans, and laboratory tests such as blood tests and 
sputum culture. Treatment typically includes antibiotics for bacterial pneumonia, antiviral 
medications for viral pneumonia, supportive care to relieve symptoms, and in severe 
cases, hospitalization may be necessary. To address this challenge, our project leverages 
the power of Convolutional Neural Networks (CNNs) in medical image analysis. CNNs 
have demonstrated remarkable success in various image classification tasks, including 
medical imaging. By harnessing this technology, we aim to develop an automated system 
capable of detecting pneumonia from chest X-ray images swiftly and accurately, without 
the need for extensive manual interpretation. 
## 3. Objective: 

Our objective is to develop a Convolutional Neural Network (CNN) model tailored for 
precise pneumonia detection from chest X-ray images. By training the model to 
accurately classify images into pneumonia-positive and pneumonia-negative categories, 
we aim to facilitate timely diagnosis and treatment of this life-threatening condition. The 
web application utilizing this model seeks to improve healthcare outcomes by reducing 
morbidity and mortality associated with pneumonia. By providing a user-friendly 
interface for uploading chest X-ray images, the web app enables individuals to 
conveniently assess their health status and receive prompt feedback on pneumonia 
presence, especially in regions with limited access to expert radiologists. This initiative 
aims to democratize access to medical diagnostics, enhancing early detection and 
intervention, ultimately saving lives. 
## 4. Dataset description: 

The dataset utilized in this study originates from one of Kaggle’s deep learning 
competitions and features lung X-ray images of infants aged one through five. These 
images were sourced from the Guangzhou Women and Children’s Medical Center and 
were meticulously verified by medical experts. All chest X-ray imaging procedures were 
conducted as part of routine clinical care for patients. The dataset comprises a total of 
5856 labeled images, among which 4273 exhibited signs of pneumonia, while the 
remaining 1583 were categorized as negative cases. 
## 5. Methodology: 

### 5.1. The preprocessing stage: 

The primary goal of using Convolutional Neural Network in most of the image 
classification tasks is to reduce the computational complexity of the model which is 
likely to increase if the inputs are images. To address the issue of class imbalance within 
our dataset, we implemented data augmentation techniques. Class imbalance, where one 
class (e.g., pneumonia-positive) is significantly more prevalent than the other (e.g., 
normal), can lead to biased model performance. Data augmentation involves generating 
synthetic data samples for the minority class to rebalance the dataset. This process helps 
prevent the model from being biased towards the majority class during training, thereby 
improving its ability to generalize to new, unseen data. Various augmentation techniques, 
such as rotation, flipping, and scaling, were applied to the pneumonia-negative class to 
create additional samples. By augmenting the data, we aimed to enhance the model's 
ability to accurately classify both pneumonia-positive and pneumonia-negative cases, 
ultimately improving the overall performance and robustness of our CNN model for 
pneumonia detection. In the code:
- `ImageDataGenerator` objects (`train_datagen` and `test_datagen`) are instantiated to 
perform data augmentation and normalization on the training and testing datasets, 
respectively.
- Augmentation parameters such as shear range, zoom range, and horizontal flip are 
specified in `train_datagen`, allowing the generator to create new augmented images by 
applying these transformations to the original images.
- Normalization is applied to both training and testing datasets by rescaling the pixel 
values to a range of [0, 1]. This ensures that the input data has consistent numerical 
values, which can improve model convergence during training.
Balancing the data:
- The number of images belonging to each class (normal and pneumonia) in the training 
directory (`train_dir`) is counted.
- If both classes have images present, class weights are computed inversely proportional 
to the class frequencies. This means that classes with fewer samples are assigned higher 
weights, making them more influential during training.
- The computed class weights (`class_weight`) are then passed to the model fitting 
process to adjust the loss function, accordingly, ensuring that the model gives equal 
importance to each class during training. 
Classes before and after balancing: 
## 5.2. Model Architecture: 

The convolutional neural network (CNN) architecture comprises sequential layers 
designed for pneumonia detection from chest X-ray images, with specific activation 
functions chosen for their benefits in learning complex patterns and improving model 
performance. 
### 5.2.1 Convolutional Layers:

 The initial layer consists of 32 filters with a kernel size of (3, 3) and ReLU (Rectified 
Linear Unit) activation function. ReLU introduces non-linearity, allowing the network to 
learn complex relationships between input features. It is chosen for its simplicity and 
effectiveness in preventing the vanishing gradient problem.
 Following each convolutional layer, max-pooling layers with a window size of (2, 2) 
are applied to down sample the feature maps, preserving essential information while 
reducing computational complexity. 
### 5.2.2 Subsequent Convolutional Layers:

 The network further deepens with 64 filters in the second convolutional layer, again 
with a (3, 3) kernel size and ReLU activation. ReLU is preferred over other activation 
functions like sigmoid or tanh due to its faster convergence during training and avoidance 
of the vanishing gradient problem.
 A max-pooling layer follows to extract more abstract features and reduce spatial 
dimensions. 
## 5.2.3 Deeper Layers: 

 The third convolutional layer increases the number of filters to 128, maintaining the (3, 
3) kernel size and ReLU activation. This enhances the network's capability to learn 
intricate patterns.
 Another max-pooling layer helps in feature extraction and dimensionality reduction. 
## 5.2.4 Flattening and Fully Connected Layers:

 The feature maps are then flattened into a one-dimensional vector to feed into the 
fully connected layers.
 A dense layer with 512 neurons and ReLU activation functions enables the network to 
learn complex relationships between features extracted from the convolutional layers. 
## 5.2.5 Output Layer:

 The final layer consists of a single neuron with a sigmoid activation function. Sigmoid 
activation is chosen for binary classification tasks as it squashes the network's output 
between 0 and 1, representing the probability of the input image belonging to the 
pneumonia-positive class. 
Overall, this CNN architecture leverages multiple convolutional and pooling layers to 
extract hierarchical features from chest X-ray images, followed by dense layers for 
classification. The use of ReLU and sigmoid activation functions contributes to faster 
convergence, better gradient flow, and improved model performance in pneumonia 
detection. 
## 5.3. Model Evaluation: 

The evaluation of the Convolutional Neural Network (CNN) model's performance for 
pneumonia detection entails a rigorous analysis, employing various key metrics and 
techniques to ascertain its efficacy in real-world scenarios. 
### 5.3.1 Prediction of test labels: 

Leveraging the trained CNN model, predictions are generated for the test dataset, 
providing a probability estimate for each chest X-ray image's classification as pneumonia 
positive. 
### 5.3.2 Calculation of test results: 

Test accuracy is computed to quantify the model's proficiency in correctly classifying 
samples within the test dataset. It serves as a fundamental measure of the model's overall 
classification performance. 
Test Accuracy: 0.9166666666666666
### 5.3.3 Generation of Confusion Matrix:

A comprehensive confusion matrix is constructed to delineate the model's predictions in 
detail, highlighting the counts of true positive (TP), true negative (TN), false positive 
(FP), and false negative (FN) classifications. This offers granular insights into the 
model's performance across different class categories. 
Confusion Matrix: TN: 190 FP: 44 FN: 8 TP: 382
### 5.3.4 Assessment of Sensitivity (Recall):

Sensitivity, or recall, is meticulously calculated to assess the model's capability in 
accurately identifying positive instances among all actual positive samples. This metric 
holds particular significance in medical contexts where sensitivity directly impacts 
diagnostic accuracy. 
Sensitivity (Recall): 0.9794871794871794
### 5.3.5 Determination of Specificity: 

Specificity is meticulously computed to gauge the model's effectiveness in correctly 
identifying negative instances among all actual negative samples. It serves as a 
complementary metric to sensitivity, offering insights into the model's discrimination 
ability. 
Specificity: 0.811965811965812 
### 5.3.6 Evaluation of Positive Predictive Value (Precision): 

Positive Predictive Value (PPV), commonly known as precision, is rigorously evaluated 
to ascertain the accuracy of positive predictions made by the model. It quantifies the 
proportion of true positive instances among all samples predicted as positive by the 
model. 
Positive Predictive Value (Precision): 0.8967136150234741 
### 5.3.7 Assessment of Negative Predictive Value: 

Negative Predictive Value (NPV) is meticulously assessed to gauge the accuracy of 
negative predictions made by the model. It provides a robust measure of the proportion of 
true negative instances among all samples predicted as negative by the model. 
Negative Predictive Value: 0.9595959595959596 
### 5.3.8 Estimation of Prevalence: 

Prevalence, a pivotal parameter, is estimated to contextualize the model's performance 
within the population. It facilitates a nuanced interpretation of sensitivity and specificity 
metrics, offering insights into the model's real-world implications. 
Prevalence: 0.625 
### 5.3.9 Visualization via ROC Curve: 

The Receiver Operating Characteristic (ROC) curve is skillfully plotted to visualize the 
interplay between sensitivity and specificity across varying classification thresholds. The 
area under the ROC curve (AUC) encapsulates the model's discriminatory prowess 
succinctly. 
### 5.3.10 Visualization via Confusion Matrix: 

The confusion matrix is a 2 dimensional array comparing predicted category labels to the 
true label. For binary classification, these are the True Positive, True Negative, False 
Positive and False Negative categories. 
### 5.3.11 Comprehensive Metric Reporting: 

A meticulous compilation of evaluation metrics, encompassing test accuracy, sensitivity, 
specificity, PPV, NPV, prevalence, and the confusion matrix, is presented. This 
assessment furnishes invaluable insights into the CNN model's performance, facilitating 
informed decisions regarding its practical utility in clinical settings. 



 
## 7.Conclusion: 

In conclusion, the application of Convolutional Neural Networks (CNNs) in 
pneumonia detection represents a significant advancement in AI for medical 
science. By leveraging CNNs, researchers and healthcare professionals can achieve 
high accuracy in diagnosing pneumonia from medical images, thereby facilitating 
early detection and timely treatment. This technology holds immense potential for 
improving patient outcomes, reducing healthcare costs, and advancing the field of 
medical imaging. However, further research and validation are necessary to ensure 
the reliability and generalizability of CNN-based pneumonia detection systems in 
real-world clinical settings. 
## 8.References: 

1. Jaiswal A.K.Tiwari, P.Kumar, S.Gupta, D.Khanna, A.Rodrigues.J.J 
Identifying pneumonia in chest x-rays: A deep learning approach. 
Measurement 145. 
2. Simonyan.K, Zisserman.A: A Very Deep Convolutional Neural Networks. 
Radiology 284. 
3. Saced.F.Paul, A.Karthigakumar ,P.Nayyar:A Convolutional neural network 
based on early detection p.p 1-17 

