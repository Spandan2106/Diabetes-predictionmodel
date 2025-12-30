🩺 Diabetes Prediction using SVM (Support Vector Machine)


📘 Overview  

This project uses a Support Vector Machine (SVM) model to predict whether a person is diabetic or not based on medical diagnostic measurements.
It utilizes the Pima Indians Diabetes Database, a widely used dataset for binary classification problems in healthcare analytics.


🧠 Objective

To build and evaluate a machine learning model that can accurately classify individuals as diabetic (1) or non-diabetic (0) using their health-related parameters.


🧩 Technologies Used

Python 3.x
NumPy
Pandas
Seaborn
Matplotlib
scikit-learn


📂 Dataset

The dataset used in this project is named diabetes-1.csv.
It contains several medical features such as:

 
Pregnancies ==>	Number of times pregnant
Glucose ==>	Plasma glucose concentration
BloodPressure ==>	Diastolic blood pressure (mm Hg)
SkinThickness ==>	Triceps skin fold thickness (mm)
Insulin ==>	2-Hour serum insulin (mu U/ml)
BMI ==>	Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction ==>	Diabetes pedigree function (genetic influence)
Age ==>	Age of the person (years)
Outcome	1 = Diabetic, 0 = Non-diabetic


⚙️ Steps Performed

Importing Libraries
Imported all required Python libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

Loading the Dataset

diabetes_dataset = pd.read_csv('/diabetes-1.csv')


Data Exploration and Visualization

Displayed the first few rows using head()

Checked dataset shape and summary statistics

Visualized relationships using Seaborn box plots:

sns.catplot(x='Pregnancies', y='Age', data=diabetes_dataset, kind='box')


Data Splitting

Divided the dataset into features (X) and target (Y)

Performed a train-test split (80%-20%) with stratification for balanced class distribution.

Feature Scaling

Standardized the data using StandardScaler to ensure all features have equal influence.

Model Training

Used an SVM Classifier with a Linear Kernel:

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)


Model Evaluation

Calculated accuracy on both training and testing datasets using:

accuracy_score()


Prediction on New Data

The model accepts user input (medical parameters) and predicts whether the person is diabetic or not.



🧾 Example Input
input_data = (1,189,60,23,846,30.1,0.398,59)


After standardization and prediction:

if prediction[0] == 0:
    print("Person is Non-Diabetic")
else:
    print("Person is Diabetic")



📊 Results
Dataset	Accuracy
Training Data	~Accuracy score shown at runtime
Test Data	~Accuracy score shown at runtime

(Note: Actual accuracy may vary based on the dataset and preprocessing.)



🧠 Insights

The SVM Linear Kernel performed well in classifying the data.

Proper feature scaling significantly improved model performance.

Useful for medical data analysis and predictive modeling.

🚀 Future Improvements

Add hyperparameter tuning using GridSearchCV.

Implement other ML algorithms (Logistic Regression, Random Forest).








