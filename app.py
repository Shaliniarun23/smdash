import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Set page config
st.set_page_config(page_title="Consumer Dashboard", layout="wide")

st.title("🧠 Consumer Insight Dashboard")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("clustered_consumer_data.csv")

st.sidebar.markdown("Download base dataset:")
with open("clustered_consumer_data.csv", "rb") as f:
    st.sidebar.download_button("Download CSV", f, file_name="clustered_consumer_data.csv")

# Label encode utility
def label_encode(df, cols):
    df = df.copy()
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Visualization", "Classification", "Clustering", "Association Rules", "Regression"])

# Placeholder for other tabs (logic will be added in external files in the actual zip)
tab1.markdown("To be continued in final package...")
tab2.markdown("To be continued in final package...")
tab3.markdown("To be continued in final package...")
tab4.markdown("To be continued in final package...")
tab5.markdown("To be continued in final package...")
