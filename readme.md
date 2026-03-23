# Customer Churn Prediction & Analysis
## Description
This project is an end-to-end Exploratory Data Analysis (EDA) and Machine Learning pipeline designed to:
- Identify high-risk customers  
- Understand the key drivers of churn  
- Build a predictive model to proactively reduce churn  
- Provide actionable, data-driven business recommendations  
---
## Dependencies & Installation
All required libraries are listed in the `requirements.txt` file.
---
## Problem Statement
Customer churn is a major source of revenue loss in the banking sector. Because acquiring new customers is significantly more expensive than retaining existing ones, relying on reactive strategies—such as attempting to win customers back after they leave—is often inefficient.
This project focuses on building a **proactive, data-driven retention engine**. By analyzing historical demographics and behavioral data, the model successfully identifies customers who are likely to churn, enabling early and targeted intervention.
---
## Methodology & Architecture
The pipeline executes a full data science lifecycle:
### Data Preprocessing:
- Automated data cleaning and validation  
- Handling missing values  
- Encoding categorical variables
### Class Balancing:
Because churn datasets are typically imbalanced (most customers stay), the initial model was naturally biased. To fix this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data, drastically improving the model’s ability to detect true flight risks.
### Predictive Engine:
The model is built using a **Random Forest Classifier**, chosen for its ability to:
- Capture non-linear relationships  
- Handle mixed data types effectively  
- Remain robust to outliers  
---
## Key Insights & Findings
To ensure reliable business insights, this project goes beyond default feature importance methods (which can sometimes be biased). Instead, **Permutation Importance** was utilized to better evaluate the true impact of each feature.
Key findings include:
- **Product usage is a major driver of churn** — Customers with more products show higher churn risk, possibly due to potential dissatisfaction with product integration or financial over-leveraging 
- **Customer inactivity is a strong churn signal**  
- **Older customers are more likely to churn**  
- Demographic factors such as **geography and gender have less influence** compared to behavioral features  
---
## Model Performance
Accuracy alone is not a reliable metric for churn prediction due to class imbalance. Therefore, this project focuses on **Recall**, which measures how well the model identifies actual churners.
- **Baseline Model (Random Forest):**
  - Accuracy: 85%  
  - Recall: 53%  
  → Misses nearly half of churn-risk customers  
- **Improved Model (with SMOTE):**
  - Accuracy: 79%  
  - Recall: 67%  
  → Significantly better at identifying churners  
### Business Trade-off
The improved model prioritizes detecting churn (higher Recall), even at the cost of more false positives.
This is intentional — the cost of reaching out to a few extra customers is far lower than losing high-value customers. Therefore, this approach aligns better with real-world business goals.
---
## Model Deployment
The final trained model is saved using `joblib` as a `.pkl` file.  
It is fully separated from the training pipeline and can be integrated into:
- Backend systems  
- APIs  
- Web applications  
for real-time churn prediction.
---
## Technical Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn, Joblib  
---
## Limitations
- Dataset is static (no time-based behavioral trends)  
- SMOTE may introduce synthetic patterns  
- Model may require retraining for new data  
- External business factors (pricing, competition, market trends) are not included  
---
## How to Run the Project
1. Clone the repository  
2. Place the dataset file (`Churn_Modelling.xlsx`) in the root directory  
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
