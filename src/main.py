import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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

#build the random forest 

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train,y_train)

#predictions 
predictions = model.predict(test_clean)

#create submission file and we use the old test data which still has the passengerId
output = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : predictions
    })

#save the file 
output.to_csv('data/submission.csv', index=False)
print('Success !!!')