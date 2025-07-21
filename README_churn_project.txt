
# Customer Churn Prediction for SaaS CRM Company

This project focuses on predicting customer churn for a SaaS-based CRM product using a synthetic yet realistic dataset. I explored different models, compared their performance, and selected the one that performed best.

---

## Project Objective
To identify which customers are likely to churn (i.e., not renew their subscription) based on their activity and account-related features.

---

## Dataset Summary
The dataset includes key user and behavioral features:

- `CustomerID`, `SignupDate`, `LastLoginDate`, `AccountAgeDays`
- `TotalLogins`, `InvoicesCount`, `OpportunitiesCount`
- `SubscriptionType`, `Industry`, `AvgInvoiceAmount`
- `Churned` (target variable)

Some logic was applied during data generation:
- Customers with no recent logins are more likely to churn
- Opportunities > Invoices, assuming a typical sales pipeline
- Older accounts usually have higher login counts

---

##  Data Preparation
- Transformed `LastLoginDaysAgo` into actual `LastLoginDate`
- Calculated aggregated values for login, invoice, and opportunity history
- Cleaned and encoded categorical variables
- Dropped non-informative columns: `CustomerID`, `SignupDate`, `LastLoginDate`

---

##  Exploratory Analysis
I performed initial analysis using:
- Histograms, scatter plots, box plots
- Correlation matrix (heatmap)
- Feature distributions by churn class

---

## Models I Tested
I trained and evaluated the following models:

### 1. Logistic Regression
- Performed the best overall
- Fast and interpretable

### 2. XGBoost
- Performance wasn’t great on this dataset
- Possibly overfitting due to synthetic nature of the data

### 3. Random Forest
- Decent results
- Useful for feature importance insights

---

## Final Decision
I chose **Logistic Regression** as the final model due to its high accuracy and simplicity. Despite testing more complex models, this one gave the most stable and interpretable results on this dataset.

---

## Files in This Repo
├── churn_dataset_raw_realistic.csv     # The synthetic dataset
├── LR.py                               # Logistic Regression implementation
├── XGBoost.py                          # XGBoost model
├── RandomForest.py                     # Random Forest model
├── README.md                           # This documentation

---

## Tools Used
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost

---

## What I Learned
- Good data matters more than complex models
- Feature engineering based on domain logic pays off
- Model interpretability helps when communicating with non-technical stakeholders

---

##  Next Steps
- Try more real-world datasets
- Explore cost-sensitive churn modeling
- Deploy this project using Streamlit or Flask

---


Pegah Kashani

---

Thanks for checking out the project! Feel free to explore or connect with me if you're interested.
