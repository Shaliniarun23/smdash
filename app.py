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

with tab2:
    st.subheader("🧪 Classification Models – Predict Willingness")

    model_dict = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    # Encode categorical features
    encode_cols = ['Age Group', 'Gender', 'City', 'Occupation', 'Income Range',
                   'Shopping Frequency', 'Purchase Factor', 'Subscribed to Loyalty Program',
                   'Referral Likelihood', 'Preferred Shopping Time', 'Preferred Shopping Day',
                   'Experienced Delivery Issues', 'Willing to Pay for Fast Delivery']
    
    df_cls = label_encode(df, encode_cols + ['Willing to Try New Platform'])

    X = df_cls.drop(columns=['Willing to Try New Platform', 'Preferred Categories',
                              'Payment Methods', 'Loyalty Engagement Factors',
                              'Offer Preferences', 'Delivery Issue Types'])
    y = df_cls['Willing to Try New Platform']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = []
    probs = {}
    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(y_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1 Score": f1_score(y_test, y_pred, average="weighted")
        })
        probs[name] = y_prob

    st.dataframe(pd.DataFrame(results).round(3))

    # Confusion matrix
    model_selected = st.selectbox("Choose model to display Confusion Matrix", list(model_dict.keys()))
    clf_model = model_dict[model_selected]
    clf_model.fit(X_train, y_train)
    y_pred = clf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_selected} - Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve
   from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

st.markdown("### 🧪 ROC Curve for Multiclass")

# Binarize output
classes = y.unique()
y_bin = label_binarize(y_test, classes=sorted(classes))
n_classes = y_bin.shape[1]

fig, ax = plt.subplots()

for name, model in model_dict.items():
    clf = OneVsRestClassifier(model)
    clf.fit(X_train, label_binarize(y_train, classes=sorted(classes)))
    if hasattr(clf, 'predict_proba'):
        y_score = clf.predict_proba(X_test)
    else:
        continue  # skip model if no probability method

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} – Class {sorted(classes)[i]} (AUC = {roc_auc:.2f})")

ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Multiclass ROC Curve")
ax.legend(loc="lower right")
st.pyplot(fig)

with tab3:
    st.subheader("🧩 Customer Segmentation – KMeans Clustering")

    # Encode and scale relevant features
    cluster_cols = ['Age Group', 'Income Range', 'Shopping Frequency',
                    'Spend Per Transaction', 'Subscribed to Loyalty Program']
    df_cluster = label_encode(df, cluster_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[cluster_cols])

    # Elbow chart
    st.markdown("### 📈 Elbow Chart")
    sse = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        sse.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K_range, sse, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("SSE")
    ax.set_title("Elbow Chart")
    st.pyplot(fig)

    # Slider to choose number of clusters
    num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=4)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

    # Persona table
    st.markdown("### 🧠 Cluster Persona Summary")
    persona = df_cluster.groupby('Cluster')[cluster_cols].mean().round(2)
    st.dataframe(persona)

    # Download full clustered data
    df_out = df.copy()
    df_out['Cluster'] = df_cluster['Cluster']
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Clustered Data", csv, file_name="clustered_output.csv")

with tab4:
    st.subheader("🔗 Association Rule Mining – Apriori")

    cols_for_rules = st.multiselect("Select columns for Apriori", 
        ['Preferred Categories', 'Payment Methods', 'Loyalty Engagement Factors', 'Offer Preferences', 'Delivery Issue Types'],
        default=['Preferred Categories', 'Payment Methods'])

    min_support = st.slider("Minimum Support", 0.01, 1.0, 0.05, 0.01)
    min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.4, 0.05)

    # Preprocess transactions
    transactions = df[cols_for_rules].fillna('').apply(lambda row: ';'.join(row.values.astype(str)), axis=1)
    transaction_lists = [t.split(';') for t in transactions]

    te = TransactionEncoder()
    te_ary = te.fit(transaction_lists).transform(transaction_lists)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # Run Apriori
    frequent = apriori(df_trans, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)

    # Show top 10
    top_rules = rules.sort_values(by="confidence", ascending=False).head(10)
    st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

with tab5:
    st.subheader("📉 Predicting Monthly Spend – Regression Models")

    model_dict = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Decision Tree Regressor": DecisionTreeRegressor()
    }

    reg_cols = ['Age Group', 'Gender', 'Occupation', 'Income Range',
                'Shopping Frequency', 'Purchase Factor', 'Subscribed to Loyalty Program',
                'Referral Likelihood', 'Preferred Shopping Time']

    df_reg = label_encode(df, reg_cols)
    X = df_reg[reg_cols + ['Spend Per Transaction']]
    y = df_reg['Expected Monthly Spend']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.markdown(f"### 🔎 {name}")
        st.write(f"Mean Squared Error: {round(mse, 2)}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Spend")
        ax.set_ylabel("Predicted Spend")
        ax.set_title(f"{name} – Actual vs Predicted")
        st.pyplot(fig)
