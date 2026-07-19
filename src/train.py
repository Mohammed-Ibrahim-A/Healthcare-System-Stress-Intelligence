import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

DATA="data/processed/final_dataset.csv"
df=pd.read_csv(DATA)
X=df.drop("mh_crisis_index",axis=1)
y=df["mh_crisis_index"]
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model=RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train,y_train)
joblib.dump(model,"models/model.pkl")
print("Model Training Completed!")