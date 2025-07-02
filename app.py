import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

# Load your synthetic dataset
df = pd.read_csv("your_dataset.csv")  # Replace with the actual filename

# ----------- PREPROCESSING -----------

# Label encoding for categorical columns
label_cols = ['Age Group', 'Gender', 'City', 'Occupation', 'Income Range',
              'Shopping Frequency', 'Purchase Factor', 'Willing to Try New Platform',
              'Subscribed to Loyalty Program', 'Referral Likelihood',
              'Preferred Shopping Time', 'Preferred Shopping Day',
              'Experienced Delivery Issues', 'Willing to Pay for Fast Delivery']

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# ----------- CLASSIFICATION -----------
# Predict Willingness to Try New Platform

X_cls = df.drop(columns=['Willing to Try New Platform', 
                         'Preferred Categories', 'Payment Methods',
                         'Loyalty Engagement Factors', 'Offer Preferences', 
                         'Delivery Issue Types'])
y_cls = df['Willing to Try New Platform']

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred_cls = clf.predict(X_test)

print("🔍 Classification Report:")
print(classification_report(y_test, y_pred_cls))

# ----------- REGRESSION -----------
# Predict Expected Monthly Spend

X_reg = df.drop(columns=['Expected Monthly Spend', 
                         'Preferred Categories', 'Payment Methods',
                         'Loyalty Engagement Factors', 'Offer Preferences', 
                         'Delivery Issue Types'])
y_reg = df['Expected Monthly Spend']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
print("\n📈 Regression Mean Squared Error (MSE):", round(mse, 2))

# ----------- CLUSTERING -----------
# Segment customers

cluster_features = df[['Age Group', 'Income Range', 'Shopping Frequency',
                       'Spend Per Transaction', 'Subscribed to Loyalty Program']]
scaler = StandardScaler()
X_cluster = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

print("\n🔎 Sample of Clustered Data:")
print(df[['Age Group', 'Income Range', 'Spend Per Transaction', 'Cluster']].head())

# ----------- OPTIONAL: Export clustered data -----------
df.to_csv("clustered_consumer_data.csv", index=False)
