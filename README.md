# 🧠 Consumer Insight Engine – Synthetic Dataset Analysis

This project demonstrates how to generate and analyze a **realistic synthetic dataset** to test the feasibility of a business idea focused on:
- Identifying key drivers of gross income
- Improving consumer segmentation and loyalty effectiveness
- Evaluating operational patterns across cities, time, and payment methods

---

## 📊 Business Use Case

A startup is exploring launching a personalized e-commerce platform. This dataset helps test:
- **Customer segmentation** for targeting
- **Classification** to predict user interest in a new platform
- **Regression** to estimate potential revenue
- **Association rule mining** to understand shopping behaviors

---

## 🧪 Dataset Overview

The dataset includes **1000+ realistic consumer responses**, mimicking actual sentiments and behaviors:
- Demographic variables: Age, City, Gender, Income
- Purchase behavior: Spend per transaction, preferred categories
- Operational insights: Time of shopping, delivery issues
- Loyalty metrics: Referral interest, program subscription

Columns with multi-choice responses (e.g. “Preferred Categories”) are stored in **semicolon-separated format** for association rule mining.

---

## 📁 Files Included

| File                          | Description |
|-------------------------------|-------------|
| `synthetic_consumer_data.csv` | The generated dataset |
| `analysis_pipeline.py`        | Python script for clustering, classification, regression |
| `README.md`                   | Project documentation |

---

## ⚙️ Requirements

To run this project, install the following dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
