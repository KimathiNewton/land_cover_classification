# Land Cover Classification

## Overview
This project aims to develop a robust predictive model for land cover classification using geospatial, environmental, and remote sensing data. Our goal is to classify each observation into one of three land cover types:
- **Buildings**
- **Cropland**
- **Woody Vegetation Cover** (with >60% cover)

The final deliverable includes a submission CSV file with occurrence probabilities for each class and a technical report summarizing our methodology, findings, and recommendations.

## Project Structure
- **README.md**: This file.
- **crop_Data_Preprocessing.ipynb**: Jupyter Notebook detailing data preprocessing, feature engineering, target encoding, SMOTE balancing, model training, and evaluation.
- **model/**: Directory to store saved models (e.g., `rf_model_tuned.pkl`).
- **submission.csv**: Final submission file with predicted probabilities.
- **requirements.txt**: List of Python dependencies.

## Data Description
The dataset contains 15,811 training samples and a corresponding test set. Key columns include:
- **Identifiers & Coordinates**: `subid`, `lat`, `lon` (and projected coordinates `X`, `Y`).
- **Target Indicators**: `building`, `cropland`, `wcover`
- **Environmental & Remote Sensing Features**: e.g., `bcount`, `bd20`, `bio1`, `bio12`, `bio7`, `bio15`, `cec20`, `dni`, `dnlt`, `dnpa`, `dor1`, `dor2`, `fpara`, `fpars`, `lcc10`–`lcc21`, `lstd`, `lstn`, `mb1`, `mb2`, `mb3`, `mb7`, `mdem`, `nppm`, `npps`, `ph20`, `sirm`, `sirs`, `slope`, `snd20`, `soc20`, `tim`

Several features (e.g., `bcount`, `dnlt`, `nppm`, `sirs`) did not have documentation, but exploratory analysis (including correlation and feature importance analysis) indicated they provide non-negligible predictive power.

## Methodology

### Data Preprocessing
1. **Exploratory Data Analysis (EDA):**  
   - Visualized missing values using heatmaps.
   - Examined distributions and correlations among features.
   - Verified that nearly all columns were complete, except for a few temperature measurements.

2. **Target Engineering & Encoding:**  
   - Combined the target indicators into a unified target variable `landcover` using a hierarchical rule:
     - If `building` is "Yes" → label as **building**
     - Else if `cropland` is "Yes" → label as **cropland**
     - Else if `wcover` indicates ">60%" → label as **woody**
     - (A revised rule was applied to avoid a generic "other" category.)
   - Converted `landcover` to numeric labels using `LabelEncoder`.

3. **Handling Class Imbalance:**  
   - The initial class distribution was imbalanced (e.g., building: 1,308, cropland: ~5,232, woody: 7,511).
   - SMOTE was applied to oversample the minority class (buildings) to achieve balance across classes.

4. **Feature Scaling:**  
   - StandardScaler was used to standardize numeric features to ensure all variables contribute equally during model training.

### Model Training and Evaluation
1. **Model Selection:**  
   We evaluated multiple classifiers:
   - Random Forest
   - Gradient Boosting
   - K-Nearest Neighbors
   - Decision Tree
   - XGBoost  
   
   Random Forest emerged as the top performer based on cross-validation metrics.

2. **Hyperparameter Tuning:**  
   GridSearchCV was used to optimize Random Forest parameters. The best parameters found were:
   - `n_estimators`: 300
   - `max_depth`: None
   - `min_samples_split`: 2
   - `min_samples_leaf`: 1

3. **Feature Importance Analysis:**  
   - The model revealed that **bcount** (building count) was the most important feature (≈32.6% importance), followed by variables such as `lcc21`, `dor2`, and certain MODIS bands (`mb7`, `mb2`, `mb1`).
   - Even undocumented features contributed meaningfully, justifying their retention.

4. **Evaluation Metrics:**  
   - Evaluated using Accuracy, Macro-averaged F1 Score, and multi-class ROC-AUC.
   - Random Forest achieved ~74.2% accuracy, ~81.1% macro F1, and ~87.8% ROC-AUC on the evaluation set.

### Test Predictions and Final Submission
- The final tuned Random Forest model was applied to the test set (after scaling) to predict class probabilities.
- The submission file includes columns: `subid`, `building_prob`, `cropland_prob`, and `wcover_prob`.

### Model Saving
- The final model was saved using joblib (`rf_model_tuned.pkl`) for future inference without retraining.

## Critical Findings and Recommendations

### Critical Findings
- **bcount is Dominant:** Building count is the single most influential feature.
- **SMOTE Effectiveness:** Balancing classes improved performance, particularly for the minority building class.
- **Model Robustness:** Random Forest provided robust performance among the models tested.
- **Value of Undocumented Features:** Even features lacking formal description contributed useful predictive information.

### Recommendations
1. **Adopt the Tuned Random Forest Model:**  
   Use the optimized Random Forest for final predictions.
2. **Maintain a Consistent Preprocessing Pipeline:**  
   Ensure scaling, SMOTE balancing, and feature encoding are applied uniformly across datasets.
3. **Document Assumptions:**  
   Clearly document interpretations of undocumented features (e.g., treat `bcount` as a proxy for urban density).
4. **Feature Engineering Iterations:**  
   Further explore feature engineering, such as creating interaction terms or aggregated indices, to potentially boost performance.
5. **Ongoing Evaluation:**  
   Regularly revalidate model performance with cross-validation and error analysis, especially for underrepresented classes.

## How to Run the Project
1. **Preprocessing and Feature Engineering:**  
   Open and run `crop_Data_Preprocessing.ipynb` to preprocess data, encode targets, balance classes with SMOTE, and scale features.
2. **Model Training and Evaluation:**  
   Execute model training, hyperparameter tuning, and evaluation sections within the notebook.
3. **Generate Predictions and Create Submission:**  
   Use the final tuned model to predict test set probabilities and create `submission.csv`.
4. **Save and Load Models:**  
   Use joblib to save the model for future use:
   ```python
   import joblib
   joblib.dump(rf_model_tuned, 'rf_model_tuned.pkl')
   # To load later:
   # rf_model_tuned = joblib.load('rf_model_tuned.pkl')
