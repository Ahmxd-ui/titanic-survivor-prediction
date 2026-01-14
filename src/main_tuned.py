import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def clean_data(df,age,mode):
    
    #fill age with age median 
    df['Age'] = df['Age'].fillna(age) 
    
    #fill embarked with most popular
    df['Embarked'] = df['Embarked'].fillna(mode)
    
    #drop the junk columns 
    df = df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], errors='ignore')

    #binary encode sex
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    
    #map embarked to numbers 
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    return df

#import train and test data 
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

#age median 
age_median = train_data['Age'].median()

#embarked mode 
embarked_mode = train_data['Embarked'].mode()[0] 

#clean the data 
train_clean = clean_data(train_data,age_median,embarked_mode)
test_clean = clean_data(test_data, age_median,embarked_mode)

#prepare training data 
X_train = train_clean.drop(columns='Survived', axis=1)
y_train = train_clean['Survived']

#tuning
print("Starting Grid Search... (This might take a moment)")

param_grid = {
    'n_estimators': [50, 100, 200],      #try 50, 100, and 200 trees
    'max_depth': [3, 5, 10, None],       #try different depths
    'min_samples_split': [2, 5, 10]      #try different split rules
}

#search
grid_search = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=5, verbose=1)

#fit 
grid_search.fit(X_train, y_train)


#winner 
print(f"Best Accuracy during training: {grid_search.best_score_}")
print(f"Best Parameters: {grid_search.best_params_}")

#save the best model 
best_model = grid_search.best_estimator_
predictions = best_model.predict(test_clean)


#create submission file and we use the old test data which still has the passengerId
output = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : predictions
    })

#save the file 
output.to_csv('data/submission_tuned.csv', index=False)
print('Success !!!')