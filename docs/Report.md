# Early COVID-19 Detection through Chest X-Ray Analysis

 
## Title and Author

<!-- - Analysis of Real Estate Sales Data -->
- Prepared for UMBC Data Science Master Degree Capstone by *Dr Chaojie (Jay) Wang*
- Author : *Saivasishta Koppaka*
- Link to the author's GitHub profile : https://github.com/DATA-606-2023-FALL-THURSDAY/koppaka_saivasishta
- Link to the author's LinkedIn profile : https://www.linkedin.com/in/sai-vasishta/
- Link to PowerPoint presentation file : https://docs.google.com/presentation/d/1E1d-Bb7VvuaRbrW_cmX7pzkKKYsM30Agt9P3rLljuEg/edit#slide=id.p
- Link to YouTube video :
  
## Objective 
In the wake of the Covid-19 pandemic, swift and accurate diagnosis has been paramount in combating the virus's spread and providing timely treatment. This research introduces a cutting-edge diagnostic platform employing a deep convolutional neural network (DCNN) to differentiate between Covid-19 and non-Covid-19 pneumonia from chest X-rays. The platform augments radiologists' capabilities, increasing diagnostic speed and accuracy, thus bolstering medical response to Covid-19. The DCNN integrates an explainable AI approach, enhancing model transparency by illustrating the decision-making process and improving prediction precision, with a demonstrated average accuracy above 94%, showcasing potential for widespread application in rapid screening and diagnosis of Covid-19. Inspired by Stanford University's CheXNet algorithm, which detects pneumonia and other pathological conditions from X-ray images, this study adapts a similar CNN algorithm. Utilizing a dataset of 227 chest X-ray images from Kaggle.

## Background
The ongoing COVID-19 pandemic has highlighted the need for rapid and accurate diagnostic tools. Chest X-rays are a valuable diagnostic tool in the assessment of lung diseases, including COVID-19. The presence of characteristic patterns in chest X-rays can aid in the early detection of COVID-19, allowing for prompt medical intervention and reducing the spread of the virus.

## Introduction
In the fight against the pandemic, the project employs Convolutional Neural Networks (CNNs) to bolster COVID-19 diagnostics through chest X-ray analysis. CNNs, a sophisticated deep learning architecture, are particularly suited for image recognition, capable of discerning intricate patterns indicative of various respiratory ailments, including the nuanced presentations of COVID-19-related pneumonia. These networks streamline feature extraction, learning from multiple layers of image data to identify distinguishing characteristics of the virus’s radiological footprint. This capability is harnessed to expedite the interpretation of X-rays, offering a swift, yet highly accurate assessment, thereby enhancing the efficacy of medical responses. The project's fusion of CNNs into the diagnostic workflow presents a significant stride towards alleviating the burden on radiology departments, ensuring quicker patient triage, and potentially improving outcomes by facilitating earlier treatment interventions.
This project harnesses the power of machine learning to enhance the diagnostic process, using a classified dataset of chest X-ray images from Kaggle, meticulously labeled into COVID-19 and normal categories, along with designated train and test subsets. Building upon the foundations laid by the CheXNet an algorithm that can detect pneumonia from chest X-rays, our approach employs a sophisticated variant of the DenseNet121 algorithm, optimized for the nuanced task of distinguishing COVID-19 from other similar respiratory conditions in X-ray imagery. The trained model, hosted on Google Drive, is seamlessly integrated with a Streamlit-based user interface. This interface enables users to upload X-ray images and swiftly receive a diagnosis, identifying if the person is affected by COVID-19 or if the result is normal, thereby streamlining the evaluation process for medical practitioners and patients alike.

### Data Sources

The dataset used for training and testing the model can be found at [this Kaggle dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset).

### Data Size

- Test Subset Size:
  - Number of images: 46
  - Total Size: 12.80 MB

- Train Subset Size:
  - Number of images: 181
  - Total Size: 126.18 MB

### Data Shape

#### Test Subset Image Shapes:
1. Image 1: Height=1146, Width=1262, Channels=3
2. Image 2: Height=557, Width=556, Channels=3
3. Image 3: Height=362, Width=439, Channels=3
4. Image 4: Height=448, Width=425, Channels=3
5. Image 5: Height=400, Width=523, Channels=3

#### Train Subset Image Shapes:
1. Image 1: Height=4032, Width=3024, Channels=3
2. Image 2: Height=557, Width=556, Channels=3
3. Image 3: Height=2336, Width=2836, Channels=3
4. Image 4: Height=1668, Width=1641, Channels=3
5. Image 5: Height=1703, Width=1690, Channels=3

### Usage

I will be creating a user-friendly interface that will allow users to upload their own chest X-ray images and test whether they have COVID-19 or not. 

## Research Questions

1. **Effectiveness of Deep Learning Models:**
   - How accurately can deep learning models classify COVID-19 cases from non-COVID-19 cases using chest X-ray images?
   
2. **Interpretability and Clinical Relevance:**
   - Can the model provide insights into the features contributing to its predictions, and how can this information be made clinically interpretable?
   
3. **Ethical Considerations:**
   - What ethical and privacy concerns should be addressed when deploying AI models for medical diagnosis, and how can patient data be safeguarded?

4. **Integration and Deployment:**
   - How can the deep learning model be seamlessly integrated into clinical workflows, and what infrastructure will be needed for widespread deployment?

This project aims to address these key research questions in the context of COVID-19 diagnosis through chest X-ray images.

## Exploratory Data Analysis
In our project for 'COVID-19 Detection Using Chest X-Rays', the Exploratory Data Analysis (EDA) section is pivotal. It details the systematic approach taken to prepare the image dataset for subsequent deep convolutional neural network (DCNN) training. Our dataset, sourced from Kaggle, comprises X-ray images pre-classified into 'COVID-19' and 'Normal' categories. The images are processed through normalization techniques, resizing them to a uniform dimension and standardizing their pixel intensity values for optimal neural network performance. Additionally, the dataset is split into training and testing subsets to evaluate the model's diagnostic efficacy accurately. The EDA also includes a visualization segment where sample X-ray images are plotted, providing an intuitive grasp of the data characteristics. This thorough preparation is foundational to ensuring the DCNN's robustness in identifying COVID-19 markers from chest X-rays.

![image](https://github.com/DATA-606-2023-FALL-THURSDAY/koppaka_saivasishta/blob/main/Normal_images.png)
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/koppaka_saivasishta/blob/main/covid_images.png)

In the preprocessing stage of our project, we implemented a normalization routine for the image dataset. Each image is loaded, converted from BGR to RGB color space, and resized to a consistent shape of 224x224 pixels. The pixel values are then normalized by scaling to a range between 0 and 1. To align with pre-trained models on ImageNet, we further normalize our data by the ImageNet dataset's mean and standard deviation, ensuring our model adapts well to the characteristics of images it was initially trained on. This step is crucial for maintaining consistency in input data and improving the model's performance during training.

## Model Architecture
The model architecture depicted in the diagram for the 'COVID-19 Detection Using Chest X-Rays'  delineates a sequential workflow starting with an X-ray image dataset. This dataset undergoes a critical preprocessing phase where images are not only standardized in terms of size and color but also potentially augmented to enhance the dataset's variability and robustness. Following preprocessing, the images are fed into a Deep Learning Convolutional Neural Network (CNN) which performs the classification task. The CNN, through its layered architecture, is able to extract and learn features from the X-ray images to discern between COVID-19 and normal cases. The final stage in this pipeline is the inference class where the processed data output from the CNN is interpreted to deliver a diagnostic result indicating the presence or absence of COVID-19 in the input chest X-ray image.
![image](https://github.com/DATA-606-2023-FALL-THURSDAY/koppaka_saivasishta/blob/main/Model_Architecture.png)

## Model Training
onehot encoding:
Initially, the labels are transformed into a binary matrix and then converted to a categorical format suitable for the classification task. Subsequently, the dataset is split, allocating 85% for training and 15% for testing, ensuring the model's performance is validated on unseen data. An ImageDataGenerator is then employed for augmenting the training data, enhancing the model's generalization capabilities by introducing variations like rotation, shifts, flips, shearing, and zooming. This approach is designed to simulate a broader range of real-world scenarios that the model might encounter.

ModelFactory:
The ModelFactory class in our project serves as a comprehensive repository for various pre-trained models, facilitating ease of access and standardized initialization of models such as VGG16, VGG19, DenseNet121, and others. Specifically, for our 'COVID-19 Detection Using Chest X-Rays' project, we have utilized DenseNet121, renowned for its efficiency in image classification tasks. The class structure allows for specifying model parameters, including input shape and the last convolutional layer name, ensuring a tailored setup for our requirements. This setup enables seamless integration and utilization of the pre-trained models, with the option to include ImageNet weights, thereby leveraging the vast repository of pre-existing knowledge to enhance our model's diagnostic capabilities.

ChexNet:
In our project, initially we used an innovative approach that utilizes a deep neural network model to accurately identify pneumonia from chest radiographs, outperforming the diagnostic accuracy of experienced radiologists. Our reference model is based on a deep learning framework trained on a substantial dataset of over 100,000 chest X-ray images, encompassing 14 different diseases. This model has been rigorously benchmarked against annotations from professional radiologists, demonstrating superior performance in terms of the F1 score. Further expanding its capabilities, our model adapts to recognize all 14 conditions within the dataset, setting a new standard for disease detection in chest radiographs.
Rather than utilizing the base ImageNet weights, the model is loaded with a specialized weight set from ChexNet.h5, tailored to our diagnostic requirements. This configuration underscores our commitment to precision and the utilization of domain-specific learning in our diagnostic tool.

Dense121:
 DenseNet121 is a variant of the Densely Connected Convolutional Networks, or DenseNets, which is a network architecture where each layer receives input from all preceding layers. DenseNet121 is a variant of the Densely Connected Convolutional Networks, or DenseNets, which is a network architecture where each layer receives input from all preceding layers.


Key Characteristics of DenseNet121:
1. Dense Connectivity: In DenseNets, each layer receives input from all preceding layers, which is contrasted with traditional architectures like ResNets where each layer receives input only from its immediate predecessor.
2. Concatenation of Features: Instead of adding outputs from previous layers, DenseNets concatenate them. This ensures that feature maps from earlier layers are used in subsequent layers.
3. Bottleneck Layers: To improve computational efficiency, DenseNets often employ bottleneck layers before each convolution layer, reducing the dimensionality of the input using a 1x1 convolution.
4. Deep Architecture: DenseNet121 implies that it has 121 layers, contributing to its ability to learn more complex patterns and hierarchies in the data.
5. Global Average Pooling: Typically, DenseNets utilize global average pooling after the final convolutional block, reducing the spatial dimensions before the final dense layers.
Advantages:
1. Parameter Efficiency: Due to the dense connectivity, each layer receives collective knowledge from all preceding layers, promoting feature reuse and 
2. Improved Gradient Flow: The dense connections also enhance gradient flow during backpropagation, mitigating issues related to vanishing or exploding gradients.
3. Robustness to Overfitting: Despite their depth, DenseNets tend to be more robust to overfitting, especially on smaller datasets, due to the direct connections between layers and parameter efficiency.

![image](https://github.com/DATA-606-2023-FALL-THURSDAY/koppaka_saivasishta/blob/main/Classification_report.png)
Here you can see that the accuracy is around 93%.

You can also see that
Both training and validation loss decrease over time, the model is learning effectively.
Increasing trends in both training and validation accuracy imply good learning.
The classification report indicates the model achieves a precision of 1.00 and a recall of 0.88 for COVID-19, demonstrating high accuracy and reliability in identifying positive cases. Normal cases have a precision of 0.85 and perfect recall. The training process, visualized in the graph, shows the loss and accuracy over 100 epochs, where the model's accuracy improves substantially over time, despite some fluctuations and peaks in validation loss, suggesting areas for future refinement.

Loss: 
If training loss decreases but validation loss increases or plateaus, this could be a sign of overfitting, meaning the model is learning the training data too well but failing to generalize to new data.
If both losses are high or not decreasing, this might indicate underfitting, meaning the model is not learning effectively from the training data.
Accuracy:
If training accuracy is high but validation accuracy is significantly lower, this could again be a sign of overfitting.
Consistently low accuracy in both might suggest underfitting or issues with the model architecture or data.

Training Loss: This represents how well the model is learning during the training phase. A downward trend in training loss over epochs indicates that the model is effectively learning patterns in the training data.
Validation Loss: This shows how well the model performs on data it hasn't seen during training. A validation loss that decreases in line with the training loss is a good sign, indicating that the model is generalizing well.
Training Accuracy: This measures the percentage of correctly predicted cases in the training set. An upward trend is desirable, showing the model is correctly learning to classify the training data.
Validation Accuracy: This indicates the percentage of correctly predicted cases in the validation set. Similar to training accuracy, an upward trend is positive.


## UI using streamlit
In my project, a user-friendly interface was developed using StreamLit, allowing for straightforward uploading and analysis of chest X-rays to determine COVID-19 infection status. The choice of StreamLit was motivated by its efficiency in creating sleek web applications for machine learning projects without the need for complex web development skills. Our Streamlit-based UI enhances user engagement by providing immediate feedback on whether the uploaded X-ray indicates a COVID-19 infection, thus serving as a practical tool for preliminary screening and aiding in swift medical response. This interface is particularly designed to be accessible for users with minimal technical expertise, ensuring broad usability.

![image](https://github.com/DATA-606-2023-FALL-THURSDAY/koppaka_saivasishta/blob/main/UI_1.png)

![image](https://github.com/DATA-606-2023-FALL-THURSDAY/koppaka_saivasishta/blob/main/UI_2.png)

## Conclusion
In conclusion, the results of this project show a potential role of a very accurate CNN algorithm to quickly identify patients, which could be useful and effective in combating the current outbreak of Covid-19. We are almost certain that it is possible for the proposed DCNN model, which shows the equivalent of the highest score for the accuracy of a specialized chest radiologist, represents a very effective examination tool for the rapid diagnosis of many infectious diseases such as the COVID-19 epidemic that do not require the introduction of a radiologist or physical examinations.

## Future work
In conclusion, the results of this project show a potential role of a very accurate CNN algorithm to quickly identify patients, which could be useful and effective in combating the current outbreak of Covid-19. We are almost certain that it is possible for the proposed DCNN model, which shows the equivalent of the highest score for the accuracy of a specialized chest radiologist, represents a very effective examination tool for the rapid diagnosis of many infectious diseases such as the COVID-19 epidemic that do not require the introduction of a radiologist or physical examinations.






