# Customer Churn Prediction

This project focuses on predicting customer churn using a Deep Learning model built with TensorFlow and Keras. The project includes exploratory data analysis, data preprocessing, model training, and inference demonstrations.

## Project Structure

- **Notebooks/**
  - `Phase-1.ipynb`: This notebook contains the Exploratory Data Analysis (EDA), Data Preprocessing, and the training of the TensorFlow/Keras neural network model.
  - `Prediction.ipynb`: This notebook demonstrates how to perform inference. It loads the saved preprocessors, encoders, and the trained model (`model.h5`) to make churn predictions on new customer data.
  - `model.h5`: The trained Keras neural network model.
  - `*.pkl`: Various serialized pickle files (e.g., `preprocessor.pkl`, `label_Encoder.pkl`, `one_hot_encoder.pkl`, `scalar.pkl`) used for data transformation and scaling during both training and inference.
- **data/**
  - `Telco_customer_churn.csv`: The dataset used for training and evaluating the model.
- **requirements.txt**: Contains the necessary Python packages and dependencies required to run the project.

## Setup and Installation

1. Ensure you have Python installed. It is recommended to create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Training & Exploration:** Open `Notebooks/Phase-1.ipynb` in Jupyter Notebook or JupyterLab to explore the data, see the preprocessing steps, and understand how the neural network was structured and trained.
- **Prediction:** Open `Notebooks/Prediction.ipynb` to see a practical example of loading the pre-trained model and preprocessing artifacts to predict churn on new data.

## Technologies Used

- **Python**
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-Learn**: Data preprocessing (scaling, encoding)
- **TensorFlow & Keras**: Building and training the deep learning model
- **Jupyter Notebook**: Interactive development and visualization
