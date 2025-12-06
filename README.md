💻 Laptop Price Predictor — Machine Learning Project
A machine learning project that predicts laptop prices using hardware specifications such as brand, CPU, RAM, storage type, GPU, and display features. This end-to-end system includes data cleaning, feature engineering, model training, evaluation, and a Streamlit web app for live predictions.

🧩 1. Problem / Objective
Laptop prices vary widely depending on hardware specs, build quality, brand value, and technology features.
Consumers often struggle to estimate whether a laptop is fairly priced.

🎯 Objective
Build a machine learning model that can predict the price of a laptop using its specifications.
This helps in:
- Assisting buyers in identifying fair prices
- Helping sellers price their products correctly
- Supporting e-commerce platforms with automated valuation

📊 2. Dataset Description
Dataset used: laptop_data.csv

Key Columns:
Company – Brand name
TypeName – Category (Gaming, Ultrabook, Notebook…)
Inches – Screen size
ScreenResolution – Resolution + panel info
Cpu – Full processor name
Ram – Memory size
Memory – SSD/HDD combinations
Gpu – Graphics processor
OpSys – Operating system
Weight – Device weight
Price – Target variable

The dataset contained complex textual fields requiring extensive preprocessing.

🧰 3. Tools & Techniques Used

🔧 Technologies:
Python
Pandas & NumPy
Matplotlib / Seaborn
Scikit-learn
XGBoost
Streamlit
Pickle

🧠 Techniques
Data Cleaning
Feature Engineering
One-Hot Encoding
Train-Test Split
Regression Modeling
Pipeline Creation
Model Serialization

🔄 4. Process Breakdown

Step 1 — Data Cleaning
- Removed extra characters (“GB”, ".0")
- Normalized storage values (TB → GB)
- Extracted categorical and numerical features

Step 2 — Feature Engineering
Display Features:
- Extracted Touchscreen, IPS, X_res, Y_res
- Calculated PPI (Pixels Per Inch)
- Dropped raw resolution fields

CPU Features:
- Extracted first three words → CPU Name

Grouped into:
- Intel i7
- Intel i5
- Intel i3
- Other Intel
- AMD
- Dropped raw CPU fields

Memory Features:
- Converted “128GB SSD + 1TB HDD” into:
- HDD
- SSD
- Hybrid
- Flash Storage

GPU Features:
- Extracted GPU brand (Nvidia, AMD, Intel)
- Removed rare values (ARM)
- OS Simplification:

Grouped into:
- Windows
- Mac
- Others / Linux / No OS

Step 3 — Target Transformation
- Applied log transform on price for better model performance

Step 4 — ColumnTransformer + Pipeline
- Encoded categorical data
- Passed numeric features directly
- Ensured preprocessing and modeling stay synchronized

Step 5 — Model Training
Tested models:
- Linear Regression
- Ridge / Lasso
- KNN
- Decision Tree
- RandomForest
- ExtraTrees
- AdaBoost
- Gradient Boosting
- Support Vector Regressor
- XGBoost 

Step 6 — Final Model
A Pipeline with:
- Preprocessing (ColumnTransformer)
- XGBRegressor with tuned hyperparameters
- Saved using Pickle (pipe.pkl).

⚠️ 5. Challenges Faced
- Parsing and normalizing messy textual data (Memory, CPU, Resolution)
- Handling rare categories without causing model bias
- Preventing overfitting on high-cardinality categorical features
- Maintaining consistent preprocessing between training & deployment
- Tuning XGBoost without overcomplicating the model

🎓 6. Learnings & Takeaways
- Real-world datasets require heavy feature engineering
- Pipelines ensure clean deployment and reproducibility
- XGBoost is powerful for structured/tabular data
- Log transformation improves regression stability
- Streamlit makes ML models easy to demo and use
