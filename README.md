# Applied Techniques of Data Mining and Machine Learning To Understand and Predict the Development of Sepsis in Humans

## DMML - Ruaidhri Brown's Contribution to Group Project

This project leverages various data mining and machine learning techniques to predict the onset of sepsis in patients using a comprehensive dataset. The primary focus is on applying different models, including RNN, LSTM, and FNN, with feature selection and data preprocessing steps to enhance prediction accuracy.

### Sepsis DataSet

The dataset used for this project is sourced from Kaggle: [Prediction of Sepsis](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis).

### Files and Descriptions

#### Data Sampling and Preprocessing

1. **1.1 Sample_Data.py**: This script samples the larger dataset down to a specified amount for ease of handling and processing.

2. **2. Preprocess_Data.py**: This script preprocesses the data to make it ready for model training. It includes handling missing data, outliers, and other essential data cleaning steps.

#### Model Training Scripts

1. **3.0 LSTM_Model_Training.py**: This script trains an LSTM model combined with Random Forest for feature selection.
   
2. **3.0 RNN_Model_Training.py**: This script trains an RNN model combined with Random Forest for feature selection.
   
3. **3.0 FNN_Model_Training.py**: This script trains an FNN model as an extra model for comparison.

#### Data Visualization

1. **stats.py**: This script contains various statistical analyses and visualizations of the dataset.
   
2. **plot-Preprocessed-Data.py**: This script generates plots for the preprocessed data to help understand the transformations and cleaning steps applied.
   
3. **plot-averages-preprocessing.py**: This script creates visualizations of averages in the preprocessing steps to highlight the impact of different preprocessing techniques.

### How to Use

1. **Clone the repository:**
    ```sh
    git clone https://github.com/RuaidhriBrown/sepsis-prediction.git
    ```
2. **Navigate to the project directory:**
    ```sh
    cd sepsis-prediction
    ```
3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
4. **Run the sampling script to create a manageable dataset:**
    ```sh
    python 1.1_Sample_Data.py
    ```
5. **Preprocess the data:**
    ```sh
    python 2_Preprocess_Data.py
    ```
6. **Train the models:**
    - For LSTM:
      ```sh
      python 3.0_LSTM_Model_Training.py
      ```
    - For RNN:
      ```sh
      python 3.0_RNN_Model_Training.py
      ```
    - For FNN:
      ```sh
      python 3.0_FNN_Model_Training.py
      ```
7. **Generate data visualizations:**
    - Statistical Analysis:
      ```sh
      python stats.py
      ```
    - Preprocessed Data Plots:
      ```sh
      python plot-Preprocessed-Data.py
      ```
    - Averages in Preprocessing:
      ```sh
      python plot-averages-preprocessing.py
      ```

### Contributing

Contributions are welcome! Please fork the repository and use a feature branch. Pull requests are warmly welcome.

### License

This project is licensed under the MIT License. See the LICENSE file for details.
