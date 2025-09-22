from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json
import joblib
from datetime import datetime
import pandas as pd
#1. Load the dataset
df = pd.read_csv('data/synthetic_fraud_dataset.csv')

#2. Split features and target
X = df.drop('label', axis=1)
y = df['label']

#3 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. Preprocessing
numeric_features = ['amount', 'is_foreign', 'prev_frauds', 'hour']
categorical_features = ['channel']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')  # 'passthrough' leaves numeric features as-is

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Fit pipeline
pipeline.fit(X_train, y_train)

# 6. Save model (artifact)
model_path = 'models/model_v1.pkl'
joblib.dump(pipeline, model_path)

# 7. Save metadata
metrics = {
    'train_score': float(pipeline.score(X_train, y_train)),
    'test_score': float(pipeline.score(X_test, y_test)),
    'version': 'v1',
    'created_at': datetime.utcnow().isoformat()
}
with open('models/metadata_v1.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Saved model to", model_path, "with metrics:", metrics)

