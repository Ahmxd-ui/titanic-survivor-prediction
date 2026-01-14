# 🚢 Titanic Survival Prediction

## 🎯 Objective
Build a **Machine Learning model** that predicts passenger survival on the Titanic with high accuracy.  
The project progresses from a **baseline model** to an **optimized, production-ready ML pipeline** using hyperparameter tuning.

---

## ❓ Problem
This is a **binary classification task**.

Using passenger features such as:
- Age
- Sex
- Passenger Class (Pclass)
- Fare

The goal is to predict:
- **0 → Did not survive**
- **1 → Survived**

The test dataset does not include labels, simulating a real-world inference scenario.

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn  
  - Logistic Regression  
  - Random Forest  
  - GridSearchCV
- **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

titanic_survivor_project/
├── data/ # train.csv and test.csv
├── notebooks/ # EDA and visualizations
├── src/
│ ├── main.py # Baseline ML pipeline
│ └── main_tuned.py # Optimized model with hyperparameter tuning
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🚀 How to Run

### Clone the Repository
```bash
git clone https://github.com/Ahmxd-ui/titanic_survivor_project.git
cd titanic_survivor_project

Install Dependencies

pip install -r requirements.txt

Run the Tuned Model (Best Accuracy)

python src/main_tuned.py

📁 Output:

    Generates submission_tuned.csv

    Uses the best model found via GridSearchCV

📋 Examples
Input Data (Raw)
PassengerId	Pclass	Sex	Age
1	3	male	22.0
2	1	female	38.0
Output File (submission_tuned.csv)
PassengerId	Survived
892	0
893	1
🧩 Machine Learning Pipeline

This project follows a standard ETL + ML workflow:

Raw Data
   ↓
Data Cleaning & Feature Engineering
   ↓
Train / Validation Split
   ↓
GridSearchCV (Hyperparameter Tuning)
   ↓
Best Model (Random Forest)
   ↓
Final Predictions

🧪 Tests & Evaluation

    Validation Strategy: 80 / 20 Train–Test Split

    Evaluation Metrics:

        Accuracy Score

        Confusion Matrix

📊 Model Performance

    Baseline (Logistic Regression): ~79%

    Tuned Random Forest: ~83.3%

💡 Key Takeaways

    Built a reusable data preprocessing pipeline

    Handled missing values (Age, Embarked)

    Applied automated hyperparameter tuning

    Improved accuracy using ensemble learning

    Eliminated environment issues using a dedicated virtual environment

🔐 Authentication

No authentication required.
This is a standalone local machine learning project.
📜 License

This project is licensed under the MIT License.
See the LICENSE file for more details.