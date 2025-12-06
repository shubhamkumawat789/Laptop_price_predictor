💻 Laptop Price Predictor (Machine Learning)

This project predicts the price of a laptop based on its hardware specifications using machine learning.
You give inputs like brand, RAM, CPU, storage, screen details etc., and the model returns an estimated price.

🧠 Project Idea:

We have a dataset of many laptops with:

Their specifications (brand, RAM, CPU, GPU, storage, screen, etc.)

Their actual selling price

The model learns the relationship between specs and price.

After training, we can give it a new laptop’s specs, and it will guess the price.

🗂️ Dataset & Important Columns

The original CSV is laptop_data.csv.
Some important original columns:

Company – Laptop brand (Dell, HP, Apple, etc.)

TypeName – Type of laptop (Gaming, Ultrabook, Notebook, etc.)

Inches – Screen size

ScreenResolution – Resolution + extra info (e.g. “1920x1080 IPS”, “Touchscreen”)

Cpu – Full CPU name (e.g. Intel Core i5 7200U)

Ram – RAM size (e.g. 8GB, 16GB)

Memory – Storage (e.g. “128GB SSD + 1TB HDD”)

Gpu – GPU information

OpSys – Operating system (Windows, macOS, Linux, etc.)

Weight – Weight of the laptop

Price – Target variable (what we want to predict)

🧹 Data Cleaning & Feature Engineering (what you did to the data)

To make the data suitable for ML, the notebook performs several steps:

1️⃣ Handling screen features

From ScreenResolution and Inches you created:

Touchscreen –

1 if “Touchscreen” is present

0 otherwise

Ips –

1 if “IPS” is present

0 otherwise

X_res, Y_res – numeric resolution values (e.g. 1920 and 1080)

ppi – Pixels Per Inch
Computed as:

𝑝
𝑝
𝑖
=
𝑋
_
𝑟
𝑒
𝑠
2
+
𝑌
_
𝑟
𝑒
𝑠
2
Inches
ppi=
Inches
X_res
2
+Y_res
2
	​

	​


Then you drop the original columns:

ScreenResolution, Inches, X_res, Y_res

So the model uses Touchscreen, IPS, and PPI instead of raw resolution text.

2️⃣ CPU simplification

From Cpu you created:

Cpu Name – first 3 words (e.g. “Intel Core i5”)

Cpu brand – grouped into:

Intel Core i7

Intel Core i5

Intel Core i3

Other Intel Processor

AMD Processor

Then you drop:

Cpu, Cpu Name

This gives a simple categorical feature for CPU power.

3️⃣ Memory → HDD / SSD / Hybrid / Flash

Memory is messy (like “128GB SSD + 1TB HDD”). You cleaned it step by step:

Remove .0, GB, TB (TB is converted to 000 GB).

Split into two parts (first drive and second drive).

Detect whether each layer is:

HDD

SSD

Hybrid

Flash Storage

Finally create numeric columns:

HDD – total HDD storage (in GB)

SSD – total SSD storage (in GB)

Hybrid – total hybrid storage (in GB)

Flash_Storage – total flash storage (in GB)

Drop helper columns used in the process.

Now the model gets clean numerical storage features.

4️⃣ GPU brand

From Gpu:

Extract Gpu brand = first word (e.g. Intel, Nvidia, AMD)

Remove rows where Gpu brand == 'ARM' (rare/unwanted)

Drop the original Gpu column

5️⃣ Operating system grouping

From OpSys you define a new column os:

Windows – for Windows 7 / 10 / 10 S

Mac – for macOS / Mac OS X

Others/No OS/Linux – everything else

Then drop OpSys.

6️⃣ Final features and target

Features X = all columns except Price

Target y = log of Price

Taking log smooths the distribution and helps the model.

You then split into train & test:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=2
)

🤖 Models & Approach

You use scikit-learn Pipelines with a ColumnTransformer:

ColumnTransformer:

One-hot encodes the categorical columns (Company, TypeName, Cpu brand, Gpu brand, os)

Keeps the remaining numeric columns as-is (RAM, Weight, Touchscreen, Ips, ppi, HDD, SSD, Hybrid, Flash_Storage, etc.)

You try multiple regression models:

Linear models: LinearRegression, Ridge, Lasso

KNN: KNeighborsRegressor

Tree model: DecisionTreeRegressor

Ensemble models:

RandomForestRegressor

ExtraTreesRegressor

AdaBoostRegressor

GradientBoostingRegressor

SVM: SVR

Gradient boosting library: XGBRegressor (XGBoost)

For each model, you:

Fit on X_train, y_train

Predict on X_test

Evaluate using:

R² score (how well it explains variance)

MAE (Mean Absolute Error) (average error in log price)

✅ Final chosen model

From the last cells in the notebook:

Final pipe is a Pipeline with:

Step 1: ColumnTransformer (one-hot encode selected columns)

Step 2: XGBRegressor with tuned parameters (e.g. n_estimators=45, max_depth=5, learning_rate=0.5)

This final pipe is what you export and use for predictions.

💾 Saving the model

At the end of the notebook:

import pickle

pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))


df.pkl – processed dataset

pipe.pkl – full pipeline (preprocessing + XGBoost model)

In your Streamlit app, you will load pipe.pkl and call:

pred_log = pipe.predict(input_df)[0]
pred_price = np.exp(pred_log)  # convert back from log to actual price

🏗️ Project Structure (suggested)

You can organize the repo like:

.
├── app.py                      # Streamlit app
├── laptop-price-predictor.ipynb
├── laptop_data.csv             # Raw data
├── pipe.pkl                    # Trained model pipeline
├── df.pkl                      # Processed dataset
├── requirements.txt
└── README.md

⚙️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt


Typical packages used:

numpy

pandas

matplotlib

seaborn

scikit-learn

xgboost

streamlit

2️⃣ (Optional) Retrain the model

If you want to retrain:

Open laptop-price-predictor.ipynb in Jupyter / VS Code.

Run all cells.

It will recreate df.pkl and pipe.pkl.

3️⃣ Run the Streamlit app
streamlit run app.py


Then:

A browser window opens.

You select:

Company, TypeName, CPU brand, GPU brand, OS

RAM, storage (HDD/SSD), weight, touchscreen yes/no, IPS yes/no, etc.

Click the Predict button.

The app shows the predicted laptop price.

🎯 Goal of the Project

Understand how different laptop specs affect price.

Practice data cleaning, feature engineering, and model comparison.

Build a deployable ML model using a Pipeline so:

Preprocessing and model stay together

You can easily load it in a web app (Streamlit) and make predictions.
