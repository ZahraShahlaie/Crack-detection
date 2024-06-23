### Automatic Crack Detection in Reinforced Concrete Structures Using Deep Learning

A paper published in 2023 has focused on the automated inspection of cracks and estimation of crack depths in reinforced concrete structures for damage assessment and determination of suitable repair methods. Most studies utilizing deep learning models for automated inspection have been limited to detecting and estimating crack width, length, area, and orientation. The innovation of this study lies in developing a comprehensive framework for automatic crack detection and depth estimation in concrete structures using images captured from portable devices. Initially, a two-class convolutional neural network (CNN) model was developed for automatic crack detection on concrete surfaces. The results demonstrate that these models are accurate and reliable for automated crack inspection, which can aid in assessing the condition of a concrete structure and selecting appropriate repair methods.

### Results and Findings:
1. **Two-Class CNN Model for Crack Detection**:
   - The model achieved a detection accuracy of 99.9% on a public dataset for detecting cracks in concrete images.
   - The trained model on the public dataset successfully detected cracks on a damaged concrete slab in the laboratory with an accuracy of 93.7%, demonstrating its capability to generalize to damaged structures.

### Limitations and Recommendations:
- The models were trained and evaluated for crack detection and depth prediction under uniform loading conditions, necessitating further studies to validate the model on cracks induced by other loading conditions such as cyclic loading.
- The models were trained and tested on a limited set of images under suitable lighting conditions and without background noise. Future studies should consider varying lighting conditions and background noise.

### Project Objectives:
This project involves developing a deep learning model for crack detection under low-light conditions, simulated nighttime scenarios, and using images captured from greater distances. Additionally, enhancing model accuracy by adding textures to crack-free images is targeted.

### Project Execution Steps:

1. **Image Preprocessing**:
   - **Adjusting Lighting**: Use image processing techniques such as brightness and contrast adjustment to simulate low-light and nighttime conditions.

2. **Adding Textures to Crack-Free Images**:
   - **Textures**: Add three different textures to crack-free images to enhance the model's accuracy in crack detection.
   - **Texture Variety**: Use various textures so that the model can identify cracks under different conditions.

3. **Modeling**:
   - **Training a Two-Class CNN Model**: Train a CNN model for crack detection. Use the collected images for training and validation.

### Introduction

This project aims to develop a deep learning model for detecting cracks in reinforced concrete structures under low-light conditions and simulated nighttime scenarios. Additionally, images taken from greater distances and with various textures added to enhance the model's accuracy in crack detection are utilized.

### Project Execution Steps

#### 1. Installing Required Libraries

Initially, necessary libraries were installed. For this project, various libraries including `kaggle` for dataset download, `gdown` for Google Drive download, and various image processing and deep learning libraries such as `opencv`, `tensorflow`, `keras`, etc., were used.

#### 2. Importing Libraries

In this stage, libraries required for image processing, deep learning modeling, and other necessary analyses were imported. This includes libraries for data processing, graphical representation, and machine learning operations.

#### 3. Downloading the Dataset

The desired dataset was downloaded from Kaggle. The `kaggle` API was utilized for this purpose, and the "Surface Crack Detection" dataset was downloaded. This dataset consists of images containing cracks and crack-free images of concrete structures.

#### 4. Extracting the Zip File

The downloaded zip file of the dataset was extracted to access the image files. The `zipfile` library was used for this purpose.

#### 5. Displaying Sample Images

At this stage, several sample images from both positive (containing cracks) and negative (crack-free) folders were loaded and displayed to examine the quality and characteristics of the images.

#### 6. Image Processing for Low-Light Simulation

The images in the dataset were processed to simulate low-light and nighttime conditions. This stage involved adjusting the brightness and contrast of the images using color space conversion to LAB and manipulating its different channels. The processed images were saved in new directories.

#### 7. Displaying Sample Processed Images

After processing the images for low-light simulation, some of these processed images were displayed to examine the processing results.

### Creating Zoomed-Out Images

#### Zoomed-Out Images from Positive Images

In this section, positive images (containing cracks) were selected from the dataset and combined with 8 negative (crack-free) images to create a 3x3 composite image. These images were saved in a new directory.

#### Zoomed-Out Images from Negative Images

Here, negative images (crack-free) were selected and combined with another set of 8 negative images to create a 3x3 composite image. These images were also saved in a new directory.

### Creating Larger Zoomed-Out Images

#### Larger Zoomed-Out Images from Positive Images

In this section, positive images (containing cracks) were selected from the dataset and combined with 24 negative (crack-free) images to create a 5x5 composite image. These images were saved in a new directory.

#### Larger Zoomed-Out Images from Negative Images

Here, negative images (crack-free) were selected and combined with another set of 24 negative images to create a 5x5 composite image. These images were also saved in a new directory.


## Image Texture Integration:

• Images in the directory "/content/Negative" are read one by one.

• For each image, textures prepared in the previous step are added using the `cv2.addWeighted` function to the original image. This operation creates new images that contain a blend of the original image and various textures.

The main objective of this code is to prepare high-quality data for training a deep learning model to detect cracks in images. By utilizing textures downloaded from Google Drive, this code transforms negative images in a way that enables the model to accurately detect cracks and perform better in identifying cracks in real-world images.

## Creating DataFrames for Training:

• All new images resulting from texture integration are categorized into "positive" and "negative".

• For each category, a DataFrame is created containing the file addresses of the images and their labels.

• These two categorized DataFrames are then combined, their rows shuffled randomly, and prepared for training model use.

The primary goal of this code is to implement three deep learning models using pretrained networks VGG16, InceptionV3, and MobileNet for crack detection in images. Below is a general overview of this code:

1. **Loading Pretrained Models:**
   - Initially, three convolutional neural networks VGG16, InceptionV3, and MobileNet are loaded using pretrained weights from the ImageNet dataset.

2. **Dataset and Model Preparation:**
   - For each model, the top layers of the pretrained networks are chosen and utilized for further model training.
   - After selecting the top layers, additional layers are added to the model, including Dense and Dropout layers to reduce overfitting and prevent fitting too closely to the training data.
   - Finally, an output layer with sigmoid activation is added for binary classification task.

3. **Compiling and Training the Models:**
   - Each model is compiled using RMSprop optimizer with a learning rate of 0.0001.
   - The chosen loss function for training the networks is `binary_crossentropy`, suitable for binary classification problems.
   - Accuracy metric is used to evaluate the performance of the networks.

The main purpose of this code is to prepare and train three deep learning models using pretrained networks for crack detection in images. By doing so, it is hoped that the trained models will demonstrate good capabilities in distinguishing images containing cracks and perform better compared to training from scratch.

1. **Splitting Data into Training and Testing Sets:**
   - Initially, out of all data available in `all_df`, 40,000 samples are randomly selected and split into training and testing sets with a ratio of 70:30.
   - This split is performed using `sample` with `random_state=1` setting.

2. **Preparing Input Data for Neural Networks:**
   - For training and validation, `tf.keras.preprocessing.image.ImageDataGenerator` is used to batch the data input.
   - `train_gen` is defined for training and validation data, and `test_gen` is defined for testing data.
   - Both `train_gen` and `test_gen` are set with `rescale=1./255` to scale the data appropriately for input into neural networks.

## Generating Training, Validation, and Test Data:

• `train_data` and `val_data`: Using `flow_from_dataframe`, training and validation data are batched with 32 samples per batch from the specified DataFrame (`train_df`). Images are resized to 227x227 pixels and input into the model in RGB format.

• `test_data`: Similarly, test data (`test_df`) is generated using the same method.

This section of the code is used to prepare data for training and evaluating deep learning models on image data. By generating batched data and scaling it, suitable conditions for training and validating models are provided to effectively identify cracks in images.

This part of the code is dedicated to training deep learning models on image data. The main objective of this section is to train the pretrained VGG16, InceptionV3, and MobileNet models on the training data (`train_data`) and then evaluate their performance on the validation data (`val_data`).

For each model, these steps are carried out:

1. **Model Definition**: The pretrained models (VGG16, InceptionV3, MobileNet) are prepared with additional layers for fine-tuning on specific data.

2. **Model Compilation**: Models are compiled using appropriate loss functions (`binary_crossentropy` for binary classification problems), suitable optimizers (such as RMSprop), and performance metrics (like accuracy).

3. **Model Training**: Using the `fit` function, models are trained on the training data (`train_data`). This process involves multiple epochs, where each epoch passes through all the data once.

4. **Model Evaluation**: After each epoch of training, the model's performance on the validation data (`val_data`) is evaluated to assess whether the model is learning correctly. This evaluation includes metrics such as accuracy and loss function.

## Plotting Model Learning:

In this section, the performance of the models during training epochs on both training and validation data is visualized. These plots include accuracy and loss metrics for all three models. These visualizations are useful to monitor whether the models are learning correctly, detect potential issues like overfitting, and assess their generalization capabilities.

## Evaluating Learning:

In this section, the final performance of each model on the test data (`test_data`) is evaluated. This evaluation includes metrics such as accuracy, loss function, confusion matrix, and classification report. These metrics help analyze the model's performance in classifying different categories effectively.

Overall, this section of the code focuses on assessing and examining the performance of deep learning models on image classification tasks, ensuring they are properly trained and capable of generalizing to new data.

## Conclusion

Through this project, a deep learning model has been developed capable of detecting cracks under various lighting conditions and at different distances, using images with various textures. This model can be valuable in assessing the condition of concrete structures and determining suitable repair methods.

The current project demonstrated that by combining positive and negative images in a matrix format, it is possible to generate zoomed-in and zoomed-out images. These images can be beneficial for training deep learning models to detect cracks under different lighting conditions using textures.

### Requirements
- Python 3.x
- keras
- NumPy
- Pandas
- gdown

### Contribution
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

### Contact
For any inquiries or support, please contact [za.shahlaie@gmail.com](mailto:za.shahlaie@gmail.com).

