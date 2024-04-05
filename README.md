# AGE-GENDER-PREDICTION
This project uses deep learning to predict age and gender from facial images. It trains a model on a dataset of images labeled with age and gender. The model achieves 89.56% accuracy in gender prediction and an average error of 6.75 years in age prediction on the testing set.


### Dataset:
- The dataset used in this project is called UTKFace, which contains facial images of various individuals with their corresponding age and gender labels.
- Each image file in the dataset is named according to the convention: "age_gender_date&time.jpg".
- The age label is the age of the individual depicted in the image, and the gender label is binary (0 for male, 1 for female).

### Exploratory Data Analysis (EDA):
- EDA involves analyzing the dataset to understand its characteristics and distributions.
- The project visualizes the age distribution of individuals in the dataset using a histogram, which shows that the majority fall between the ages of 25 to 30 years old, with outliers at higher ages.
- Additionally, it displays a few sample images from the dataset along with their corresponding age and gender labels for visual inspection.

### Feature Extraction:
- Feature extraction involves converting raw data (images) into a format suitable for training a machine learning model.
- In this project, grayscale images are loaded, resized to 128x128 pixels, and converted into numpy arrays as features.
- The extracted features are then normalized to a range of [0, 1].

### Data Splitting:
- The dataset is split into training and testing sets using a ratio of 80:20.
- Both gender and age labels are split accordingly for training and testing.

### Model Architecture:
- The deep learning model architecture consists of convolutional layers followed by fully connected layers.
- Convolutional layers (Conv2D) are used for feature extraction from images due to their ability to capture spatial hierarchies in data.
- Max-pooling layers (MaxPooling2D) are employed to downsample feature maps, reducing computational complexity and overfitting.
- Dropout layers are added to prevent overfitting by randomly dropping a fraction of neurons during training.
- Two output layers are defined for predicting gender (binary classification) and age (regression).

### Model Compilation:
- The model is compiled with appropriate loss functions, optimizers, and evaluation metrics for both gender and age predictions.
- Binary cross-entropy loss is used for gender prediction, and mean absolute error (MAE) is used for age prediction.
- The Adam optimizer is chosen for optimization.

### Model Training:
- The model is trained on the training data with batch size 32 and for 30 epochs.
- During training, both gender and age labels are used as targets.
- Training progress and validation metrics are monitored to ensure model convergence and generalization.

### Model Evaluation:
- The project evaluates the trained model's performance using various metrics.
- Accuracy and loss graphs are plotted for both gender and age predictions to visualize training and validation performance.
- Additionally, gender prediction accuracy and age prediction mean absolute error (MAE) are calculated on the testing set to assess the model's overall performance.

### Evaluation Results:

Upon evaluating the trained model, the following results were obtained:

#### Gender Prediction Accuracy on Testing Set:
- The model achieved a gender prediction accuracy of approximately 89.56% on the testing set. 
- This accuracy represents the proportion of correctly predicted gender labels compared to the true labels in the testing set.
- The high accuracy indicates the effectiveness of the model in distinguishing between male and female individuals based on facial images.

#### Age Prediction MAE on Testing Set:
- The mean absolute error (MAE) for age prediction was approximately 6.75 years on the testing set.
- MAE quantifies the average absolute difference between the predicted and true age labels in the testing set.
- A lower MAE value signifies better accuracy in predicting the age of individuals.
- The relatively low MAE obtained demonstrates the model's capability to estimate the age of individuals with reasonable accuracy.

These evaluation metrics provide insights into the performance of the model in both gender and age predictions. The high accuracy for gender prediction and low MAE for age prediction validate the model's effectiveness in capturing relevant features from facial images and making accurate predictions. These results affirm the model's utility in real-world applications such as demographic analysis, personalized marketing, and age-specific product recommendations.

### Model Inference:
- Finally, the trained model is used to make predictions on new images.
- Sample images from the dataset are selected, and their original gender and age labels are compared with the model's predicted labels.
- The images along with their original and predicted labels are displayed for qualitative assessment of the model's performance.

### CONCLUSION:
This project demonstrates the end-to-end process of building a deep learning model for age and gender prediction from facial images, encompassing data preprocessing, model construction, training, evaluation, and inference.
