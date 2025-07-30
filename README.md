#  End-to-End Property Price Prediction Engine

A complete data science application that scrapes real estate data, cleans it using a robust set of logical rules, and trains a machine learning model to predict property prices. The entire workflow is managed through an interactive web application built with Streamlit.

---

## 🚀 Key Features

* **Automated Data Scraping**: A powerful scraper using Selenium and Undetected Chromedriver navigates multiple pages across 44 different real estate locations to gather raw data.
* **Interactive Data Cleaning**: An all-in-one Streamlit app allows users to upload raw data and perform initial cleaning steps like removing duplicates and missing values.
* **Advanced Outlier Detection**: The application uses a sophisticated, rule-based system to identify and remove logically inconsistent data (e.g., impossible bedroom density, unrealistic price-to-area ratios) while preserving valid luxury property listings.
* **In-App EDA**: The app automatically generates and displays a comprehensive Exploratory Data Analysis (EDA) with enhanced visualizations (histograms, violin plots, heatmaps) to understand the cleaned data.
* **Live Model Training**: Users can train a RandomForest Regressor model directly within the application with a single click.
* **Interactive Price Prediction**: A dedicated tab allows users to input property features and get an instant price prediction from the newly trained model.
* **Confidence Score**: The prediction interface includes a confidence score and a prediction interval, providing insight into the model's certainty for each estimate.

---

## 🛠️ Technology Stack

* **Language**: Python
* **Data Scraping**: Selenium, Undetected Chromedriver, BeautifulSoup
* **Data Analysis & Manipulation**: Pandas, NumPy
* **Machine Learning**: Scikit-learn (RandomForestRegressor)
* **Data Visualization**: Matplotlib, Seaborn
* **Web Application**: Streamlit

---

## 📂 Project Structure

```
harga_properti/
│
├── 📂 data/
│   ├── properti_data.csv             # Raw data from scraper
│   ├── dropped_data.csv            # Outliers removed during preprocessing
│   └── properti_data_cleaned.csv   # Clean data ready for model training
│
├── 📂 notebooks/
│   ├── preprocessing.ipynb         # Jupyter Notebook for data cleaning
│   ├── eda.ipynb                   # Jupyter Notebook for EDA
│   └── model_trainer.ipynb         # Jupyter Notebook for model training
│
├──  scraper.py                      # Standalone script for live data scraping
├──  app.py                          # The main all-in-one Streamlit application
└──  README.md                       # Project documentation
```

*(**Note:** It's good practice to organize your files into folders like `data/` and `notebooks/`)*

---

## ⚙️ Setup and Usage

### 1. Installation

First, clone the repository and install the required Python packages.

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

# Install dependencies
pip install pandas scikit-learn streamlit matplotlib seaborn undetected-chromedriver joblib
```

### 2. Data Collection (Optional)

If you need to re-scrape the data, you can run the standalone scraper. This is a long process and requires manual intervention at the start to solve a security check.

```bash
python scraper.py
```
This will produce the raw `properti_data.csv` file.

### 3. Running the Application

The main way to use this project is through the all-in-one Streamlit application.

**To launch the app, run the following command in your terminal:**
```bash
streamlit run app.py
```
Your web browser will open with the application running.

#### **Workflow within the App:**

1.  **Tab 1: 📂 Data Loader & EDA**
    * Click "Browse files" and upload your raw `properti_data.csv` file.
    * Review the **Data Summary** and **Data Quality Checks**.
    * Optionally, select the checkboxes to remove duplicates or missing values.
    * Click the **"Process Data and Run EDA"** button. The app will clean the data, remove outliers, and display a full set of visualizations.

2.  **Tab 2: 🛠️ Model Training**
    * Once the data has been processed in the first tab, go to this tab.
    * Click the **"Train RandomForest Model"** button.
    * Wait for the training to complete. The model's performance (Mean Absolute Error) will be displayed.

3.  **Tab 3: 🏠 Price Predictor**
    * After the model is trained, go to this final tab.
    * Use the sidebar controls to input the features of a property.
    * Click the **"Predict Price"** button to get an estimated price, a confidence score, and a prediction range from your newly trained model.
