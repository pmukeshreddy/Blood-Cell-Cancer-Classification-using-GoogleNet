# Blood-Cell-Cancer-Classification-using-GoogleNet
Project Overview
This project focuses on the classification of blood cell cancer types using deep learning with the GoogleNet architecture. The goal is to accurately classify different categories of blood cell cancer from microscopic images, aiding in the early detection and diagnosis of cancer.

Dataset
The dataset consists of labeled images of blood cells, categorized into different classes based on cancer type. Each image has been pre-processed to ensure compatibility with the GoogleNet architecture.

Model Architecture
We use GoogleNet (Inception v1), a deep convolutional neural network, which is known for its efficiency and accuracy in image classification tasks. The architecture includes:

Convolutional layers for feature extraction
Inception modules to capture multi-scale features
Global Average Pooling to reduce the number of parameters
Softmax layer for multi-class classification
Requirements
To run this project, you will need:

Python 3.x
PyTorch
Torchvision
NumPy
Matplotlib
Jupyter Notebook (optional, but recommended for running the notebook files)
Training and Evaluation
The model is trained using categorical cross-entropy loss and the Adam optimizer. The notebook provides evaluation metrics like accuracy, precision, recall, and F1-score. A confusion matrix is generated to assess the performance across different classes.

Results
Training Accuracy: (Update based on your results)
Validation Accuracy: (Update based on your results)
The model achieves good performance in classifying blood cell cancer images, showing promise for real-world applications in medical diagnosis.

Conclusion
This project demonstrates the potential of using GoogleNet for the classification of blood cell cancer. Further improvements can be made by experimenting with different architectures or hyperparameters.

Acknowledgments
The dataset was obtained from (https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class).
GoogleNet architecture is inspired by the original paper: Going Deeper with Convolutions (Inception v1).
