Detecting Spam Emails Using Tensorflow In Python:
 Used NLP and deep learning to classify spam and non-spam emails with high accuracy.- Improved email filtering and user experience
Code explanation - 
It aims to classify spam and non-spam emails using TensorFlow in Python. It utilizes natural language processing (NLP) techniques along with deep learning to achieve high accuracy in email classification. Here's a breakdown of the code:

1. Importing Libraries:
    - numpy: Numerical operations in Python.
    - pandas: Data manipulation and analysis.
    - tensorflow: Deep learning library developed by Google.
    - train_test_split from sklearn.model_selection: Function to split the dataset into training and testing sets.
    - CountVectorizer from sklearn.feature_extraction.text: To convert text data into numerical feature vectors.
    - Sequential and Dense from tensorflow.keras.models: For building the neural network model.
    - Dropout from tensorflow.keras.layers: Regularization technique to prevent overfitting.

2. Loading Dataset:
    - Reads the dataset from the CSV file located at "C:/Users/dharm/Downloads/archive (4)/spam_ham_dataset.csv" using `pd.read_csv()` function.

3. Preprocessing:
    - `CountVectorizer` converts the text data into a matrix of token counts, where each row represents an email and each column represents a unique word in the corpus.
    - `X` represents the input features (email text), and `y` represents the target variable (label_num), which indicates whether an email is spam (1) or not spam (0).

4. Splitting Data:
    - Splits the dataset into training and testing sets using `train_test_split()` function from scikit-learn. 80% of the data is used for training and 20% for testing.

5. Building the Model:
    - Creates a Sequential model, a linear stack of layers.
    - Adds three Dense layers with ReLU activation function. The first two layers have 64 units each, and the last layer has 1 unit with sigmoid activation function, which gives the probability of an email being spam.

6. Compiling the Model:
    - Compiles the model using Adam optimizer and binary cross-entropy loss function.

7. Training the Model:
    - Fits the model to the training data for 5 epochs with a batch size of 32. Also, it uses 20% of the training data as validation data for monitoring the model's performance during training.

8. Evaluating the Model:
    - Evaluates the trained model on the test data to compute the loss and accuracy.

9. Printing Results:
    - Prints the test loss and accuracy obtained by the model.

It demonstrates how to build a basic neural network model using TensorFlow to classify spam and non-spam emails accurately. To further improve the model, you can experiment with different architectures, hyperparameters, and preprocessing techniques. Additionally, you can deploy the model to enhance email filtering and user experience.
<img width="770" alt="image" src="https://github.com/DSTAR15/DKT_Project1/assets/128448451/f31ffc1a-15c2-404c-bcaa-e481b8ce82c3">

