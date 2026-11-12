#ex 19 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

df = pd.read_csv("fake_news_detection - fake_news_detection.csv")

X = df.drop(["label","news_type"],axis=1)
y = df['label']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)  

pred = model.predict(x_test)
print(pred)
print("Accuracy:", accuracy_score(y_test,pred))

with open("model.pkl","wb") as file:
    pickle.dump(model, file)
    print("model.pkl has been created")