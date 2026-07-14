# Customer Churn Prediction

This project focuses on predicting customer churn using a Deep Learning model built with TensorFlow and Keras. The project includes exploratory data analysis (EDA), data preprocessing, model training, and a Streamlit-based web application to perform real-time churn predictions.

## Project Structure

- **Notebooks/**
  - `app.py`: A Streamlit web application providing a user-friendly interface to input customer details and predict churn probability in real time.
  - `Phase-1.ipynb`: Contains the Exploratory Data Analysis (EDA), Data Preprocessing, and training of the TensorFlow/Keras neural network model.
  - `Prediction.ipynb`: Demonstrates how to load the preprocessors, encoders, and the trained model (`model.h5`) to make churn predictions programmatically on new customer data.
  - `model.h5`: The trained Keras neural network model.
  - `*.pkl`: Various serialized pickle files (e.g., `preprocessor.pkl`, `label_Encoder.pkl`, `one_hot_encoder.pkl`, `scalar.pkl`) used for data transformation and scaling.
- **data/**
  - `Telco_customer_churn.csv`: The IBM Telco dataset used for training and evaluating the model.
- **requirements.txt**: List of Python dependencies.

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

### 📊 Streamlit Web Application
To run the interactive prediction interface:
```bash
streamlit run Notebooks/app.py
```
This application organizes customer parameters into four intuitive sections:
1. **👤 Demographic Information:** Profile inputs like Gender, Senior Citizen, Partner, and Dependents.
2. **📞 Services Subscribed:** Subscribed plans and add-ons (Phone Service, Internet Service, Online Security, Backup, Tech Support, etc.).
3. **📄 Contract & Billing:** Contract terms (Month-to-month, One year, Two year), Paperless Billing, and Payment Method.
4. **💵 Financial & Valuation Metrics:** Monthly Charges, Total Charges, and CLTV (Customer Lifetime Value).

It processes inputs through the preprocessor pipeline and uses the deep learning model to output the churn probability percentage along with a clear determination of whether the customer is likely to churn or stay.

### 📓 Jupyter Notebooks
- **Training & Exploration:** Open `Notebooks/Phase-1.ipynb` in Jupyter Notebook or JupyterLab to explore the data and see model training.
- **Prediction:** Open `Notebooks/Prediction.ipynb` to see an example of programmatic inference.

## Technologies Used

- **Python**
- **Streamlit**: Web application framework for interactive ML tools
- **TensorFlow & Keras**: Building and training the deep learning model
- **Scikit-Learn**: Data preprocessing, scaling, and encoding pipelines
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Jupyter Notebook**: Interactive development and visualizations

