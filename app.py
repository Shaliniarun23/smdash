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
with tab1:
    st.subheader("🔍 Exploratory Data Insights")

    # 1. Distribution of Spend per Transaction
    fig, ax = plt.subplots()
    sns.histplot(df['Spend Per Transaction'], kde=True, bins=30, ax=ax)
    ax.set_title("Distribution of Spend Per Transaction")
    st.pyplot(fig)

    # 2. Monthly Spend vs Occupation
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Occupation", y="Expected Monthly Spend", ax=ax)
    ax.set_title("Monthly Spend by Occupation")
    st.pyplot(fig)

    # 3. Age Group vs Willingness
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Age Group", hue="Willing to Try New Platform", ax=ax)
    ax.set_title("Age Group vs Willingness to Try Platform")
    st.pyplot(fig)

    # 4. Spend vs Subscription to Loyalty Program
    fig, ax = plt.subplots()
    sns.violinplot(data=df, x="Subscribed to Loyalty Program", y="Spend Per Transaction", ax=ax)
    ax.set_title("Spending by Loyalty Subscription")
    st.pyplot(fig)

    # 5. Top Categories
    st.markdown("### 🛍️ Most Common Preferred Categories")
    all_cats = ";".join(df["Preferred Categories"].dropna()).split(";")
    top_cats = pd.Series(all_cats).value_counts().head(10)
    st.bar_chart(top_cats)

    # 6. Preferred Payment Methods
    st.markdown("### 💳 Preferred Payment Methods")
    all_pm = ";".join(df["Payment Methods"].dropna()).split(";")
    top_pm = pd.Series(all_pm).value_counts()
    st.bar_chart(top_pm)

    # 7. Loyalty Engagement Preferences
    st.markdown("### 🎯 Loyalty Engagement Preferences")
    all_loy = ";".join(df["Loyalty Engagement Factors"].dropna()).split(";")
    top_loy = pd.Series(all_loy).value_counts()
    st.bar_chart(top_loy)

    # 8. Spend vs Time of Day
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Preferred Shopping Time", y="Spend Per Transaction", ax=ax)
    ax.set_title("Spend vs Time of Day")
    st.pyplot(fig)

    # 9. Delivery Issues by City
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="City", hue="Experienced Delivery Issues", ax=ax)
    ax.set_title("Delivery Issues by City")
    st.pyplot(fig)

    # 10. Referral Likelihood vs Spend
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Referral Likelihood", y="Expected Monthly Spend", ax=ax)
    ax.set_title("Spend vs Referral Likelihood")
    st.pyplot(fig)
