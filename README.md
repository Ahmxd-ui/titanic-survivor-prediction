# 🚢 Titanic Survival Prediction

## 🎯 Objective
To build a Machine Learning model that predicts which passengers survived the Titanic shipwreck with high accuracy, moving from a baseline model to an optimized production-ready pipeline.

## ❓ Problem
The challenge is a binary classification problem: using passenger data (such as Age, Sex, Class, and Fare) to predict survival (0 = Died, 1 = Survived) for a test set where the outcome is hidden.

## 🛠 Tech Stack
* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, Logistic Regression, GridSearchCV)
* **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
```text
├── data/                  # Contains train.csv and test.csv
├── notebooks/             # Jupyter Notebooks for EDA and visuals
├── src/
│   ├── main.py            # Baseline model pipeline
│   └── main_tuned.py      # Optimized model with Hyperparameter Tuning
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

🚀 How to Run

    Clone the repository:
    Bash
 git clone https://github.com/Ahmxd-ui/titanic_survivor_project.git
 cd titanic_survivor_project

 Install dependencies:
 Bash

 pip install -r requirements.txt

 Run the Tuned Model (Best Accuracy):
 Bash

    python src/main_tuned.py

    This will perform Grid Search and save submission_tuned.csv.


📋 Examples

Input Data (Raw): | PassengerId | Pclass | Sex | Age | | :--- | :--- | :--- | :--- | | 1 | 3 | male | 22.0 | | 2 | 1 | female | 38.0 |

Output File (submission_tuned.csv): | PassengerId | Survived | | :--- | :--- | | 892 | 0 | | 893 | 1 |


🧩 Pattern Placeholder

This project follows a standard ETL (Extract, Transform, Load) Machine Learning pipeline:
Code snippet

graph LR
    A[Raw Data] --> B(Clean Data Function)
    B --> C{Training Phase}
    C -->|Train Split| D[GridSearchCV Tuning]
    D --> E[Best Model: Random Forest]
    E --> F[Final Predictions]
    
    style A fill:#f9f,stroke:#333
    style E fill:#bbf,stroke:#333


🧪 Tests

    Validation Strategy: 80/20 Train-Test split.

    Metrics Used: Accuracy Score, Confusion Matrix.

    Current Performance:

        Baseline (Logistic Regression): ~79%

        Tuned Random Forest: ~83.3%


💡 Key Takeaways

    Built a reusable data cleaning pipeline (handling missing values for Age/Embarked).

    Implemented automated Hyperparameter Tuning using GridSearchCV.

    Solved "Environment Hell" by using a dedicated Python venv.


🔐 Authentication

No authentication is required. This is a standalone local script.


📜 License

This project is licensed under the MIT License - see the LICENSE file for details.


