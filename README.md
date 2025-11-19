## 🚗 BMW Sales Classification Model (2010-2024)

This project aims to classify BMW sales records into "High" or "Low" sales performance categories using historical data spanning from 2010 to 2024. The core of the solution involves an **XGBoost Classifier** model, built upon a robust data preprocessing and feature engineering pipeline.

### 🎯 Project Goal

The primary objective is to develop a machine learning model capable of accurately predicting whether a given set of BMW sales attributes (Model, Price, Region, Mileage, etc.) will result in a **High** or **Low** sales volume classification.

---

### 📂 Dataset

The analysis is performed on the `BMW sales data (2010-2024).csv` dataset, which includes the following key features:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Year` | Numeric | Model Year |
| `Model` | Categorical | Specific BMW Model (e.g., M5, X3) |
| `Region` | Categorical | Sales Region |
| `Price_USD` | Numeric | Sale Price in USD |
| `Mileage_KM` | Numeric | Vehicle Mileage in Kilometers |
| `Fuel_Type` | Categorical | Fuel used by the vehicle |
| `Engine_Size_L` | Numeric | Engine displacement in Liters |
| **`Sales_Classification`** | Target | **High** or **Low** sales performance (Target Variable) |

### 🛠️ Methodology and Pipeline

The solution employs a comprehensive machine learning pipeline to ensure data integrity and maximize model performance.

1.  **Exploratory Data Analysis (EDA):** Initial analysis covered feature distributions, correlation matrices, and visualized the relationship between key features and the target variable.
2.  **Feature Selection & Correction:** The highly predictive `Sales_Volume` feature was identified as a source of **data leakage** and **dropped** from the feature set, transitioning the model to a realistic performance baseline.
3.  **Feature Engineering:** New temporal features, such as **`Car_Age`** (Current Year - Model Year), were created to enhance predictive power.
4.  **Preprocessing:**
    * **Scaling:** Numeric features (e.g., `Price_USD`, `Mileage_KM`, `Car_Age`) are standardized using **`StandardScaler`**.
    * **Encoding:** Nominal categorical features (e.g., `Model`, `Region`, `Fuel_Type`) are converted using **`OneHotEncoder`** within a `ColumnTransformer` and integrated into the pipeline.
5.  **Model Training:** An **XGBoost Classifier** was utilized for its efficiency and performance on tabular data.
6.  **Hyperparameter Tuning:** **`RandomizedSearchCV`** was reapplied to optimize parameters after the data leakage fix, focusing on achieving the highest reliable accuracy.

### 📊 Key Findings and Results

After removing the structural error, the model established a stable and reliable performance level.

* **Initial Leakage Accuracy:** 99.9% (Structural Error/Discarded)
* **Realistic Test Accuracy:** **~69%**
* **Class Imbalance:** The target variable exhibits significant class imbalance (approx. 70% Low vs. 30% High). The current performance serves as a solid baseline for future optimization efforts.

### 🚀 Future Work (To Improve Accuracy)

1.  **Class Imbalance Handling:** Implement **`scale_pos_weight`** within the XGBoost parameters to properly weight the minority class.
2.  **Advanced Feature Engineering:** Explore creation of interaction features (e.g., annual average mileage, price-to-size ratios).
3.  **Extensive Hyperparameter Search:** Conduct a deeper search (e.g., using `GridSearchCV`) focusing on regularization and learning rate to achieve optimal model generalization.

---

### 💻 Requirements

To run this project, you need the following Python libraries:

### ⚙️ How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository Link Here]
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    ```
3.  **Obtain Data:** Ensure the `BMW sales data (2010-2024).csv` file is placed in the project directory.
4.  **Execute the Notebook:** Open and run the cells in `Model.ipynb` to replicate the full pipeline, training, and evaluation steps.
