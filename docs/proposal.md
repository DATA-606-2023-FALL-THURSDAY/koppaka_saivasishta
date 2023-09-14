# COVID-19 Chest X-ray Image Classifier and Web Application

## Introduction

This repository will host a deep learning model for the detection of COVID-19 from chest X-ray images and a web application that will allow users to interact with the model. The model will use the entire image for prediction, classifying each image as "Covid" or "Normal."

### COVID-19 Overview

COVID-19, or "Coronavirus Disease 2019," is a highly contagious respiratory illness caused by the novel coronavirus, SARS-CoV-2. It was first identified in late 2019 in Wuhan, China, and has since led to a global pandemic. It primarily spreads through respiratory droplets, causing symptoms such as fever, cough, shortness of breath, and fatigue. Early diagnosis will be crucial for timely intervention and controlling its spread.

### Importance of Chest X-ray Classification

Accurate classification of chest X-ray images into "COVID-19" or "Normal" matters for several reasons:

1. **Early Intervention:** Swift identification of COVID-19 cases will enable early treatment, potentially improving outcomes.

2. **Resource Allocation:** Accurate classification will optimize resource allocation, ensuring proper care for patients.

3. **Disease Control:** Early detection will help isolate infected individuals, reducing transmission risks.

4. **Public Health:** Classification will aid in tracking disease spread, supporting informed decisions by health authorities.

5. **Medical Support:** Chest X-ray classification will complement diagnostic tools, especially in resource-limited areas.

6. **Research Insights:** Data from this model will contribute to research, enabling analysis of trends and treatment effectiveness.

In summary, classifying chest X-ray images will be vital for managing COVID-19's impact on individuals and communities.

### Background

The ongoing COVID-19 pandemic has highlighted the need for rapid and accurate diagnostic tools. Chest X-rays are a valuable diagnostic tool in the assessment of lung diseases, including COVID-19. The presence of characteristic patterns in chest X-rays can aid in the early detection of COVID-19, allowing for prompt medical intervention and reducing the spread of the virus.

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
