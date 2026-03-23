import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import LabelEncoder as LE
from scipy.stats import pearsonr as pr
from sklearn.metrics import confusion_matrix as c_m
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc
import joblib

print("\n======PREDICTION OF CHURN CUSTOMERS=======\n")
#Load the raw data
df = pd.read_excel("Churn_Raw.xlsx")
name_ = "Churn_Raw.xlsx"
print(f"Opening the dataset: {name_}\n")

#Counting
total_rows = len(df)

#Checking duplicates
print("\nChecking for duplicates...\n")
dupl = df.duplicated()
dup = df.duplicated().sum()
print(dupl.sort_values(ascending=True))
if dup == 0:
    print(f"There are no duplicates in the dataset: {name_}\n")
else:
    print(f"There are {dup} duplicated values in the dataset: {name_}.\n")

#Handling missing values
print("\nChecking for missing values...\n")
missing = df.isna().sum().sum()
if missing == 0:
    print(f"There are no missing values in the dataset: {name_}\n")
elif missing < (0.05 * total_rows * len(df.columns)):
    print(f"Minor cleaning needed. There are {missing} values in the dataset: {name_}\n")
else:
    print(f"Large missing values: {missing} in the dataset: {name_}. Consider dropping rows...\n")
    
for col in df.columns:
    missing_col = df[col].isnull().sum()
    if missing_col > 0:
        print(f"Missing columns: {missing_col}")
        perc_missing_col = round((missing_col/total_rows * 100), 1)
        if perc_missing_col > 20:
            print(f"Missing values in the dataset: {name_} on {col} is {perc_missing_col}%. Consider dropping the row.\n")
        else:
            print(f"Missing values are not significant. Consider filling it up.\n")

#------Analyzing Customers
print("\n----Looking For Insights----\n")
#Geography
countries = df['Geography'].value_counts()
print(f"Regions Available: {countries}\n")
perc_countries = countries/len(df['Geography']) * 100
print(perc_countries)

#-------Pie Chart of Geograhy Distribution
plt.figure(figsize=(5,3))
perc_countries.plot(kind='pie', autopct = '%1.2f%%')
plt.title('Distribution of Customers by Geography')
plt.ylabel('')
plt.show()

#Gender
print("Gender Available:\n")
gender = df['Gender'].value_counts()
perc_gender = gender/len(df['Gender']) * 100
print(gender, perc_gender)

#--------Pie Chart of Gender Distribution
plt.figure(figsize=(5,3))
perc_gender.plot(kind='pie', autopct = '%1.2f%%')
plt.title('Distribution of Customers by Gender')
plt.ylabel('')
plt.show()

#--------First GroupBy Analysis----------
print("\n------RAW GROUPBY ANALYSIS------\n")
geo_churn_analyze = df.groupby([
    'Geography',
    'Gender',
    'Exited'
]).agg({
    'Age':'mean',
    'CreditScore':'mean',
    'Tenure':'mean',
    'Balance':'max',
    'NumOfProducts':'mean',
    'HasCrCard': 'mean',
    'IsActiveMember':'mean',
    'EstimatedSalary':'mean'
}).round(2)

#------Correlation Matrix--------to see if any column is related to another

num_col = ['CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]
num_col_corr = df[num_col].corr()

sns.heatmap(num_col_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap on Num_Data', fontsize=16)
plt.show()

#Identify features most correlated with churn
#------Key Drivers of Churn by Geography-------
print("\n-----CHURN DRIVERS-----\n")
for country in df['Geography'].unique():
    country_ = df[df['Geography'] == country]
    col_to_check = num_col + ['Exited']
    correlation = country_[col_to_check].corr()['Exited']
    correlations = correlation.drop('Exited')
    #We use .abs() because a strong negative score (-0.5) is just as important as a strong positive (+0.5)
    top_3 = correlations.abs().sort_values(ascending=False).head(5)
    
    print(f"{country.strip().upper()} (Total Customers: {len(country_)})")
    for feature in top_3.index:
        #We grab the REAL correlation value (with its + or - sign) to print
        real_value = correlations[feature]
        print(f"   - {feature}: {real_value:.3f}")
        
    print("-" * 35)

#Influence of each Driving Col on Churn
#Crosstab analysis
#--------Geo vs Churn
Per_Geo = (pd.crosstab(index=df['Geography'], columns=df['Exited'], normalize='index')*100).round(1)
print("\n----Churn Rates By Geography----\n")
print(f"{Per_Geo}")
Per_Geo.plot(
    kind='bar', 
    stacked=True, 
    figsize=(8, 6), 
    color=['#2ca02c', '#d62728'] # Green for staying (0), Red for leaving (1)
)
plt.title('Customer Churn Rate by Geography', fontsize=16, fontweight='bold')
plt.xlabel('Geography', fontsize=12)
plt.ylabel('Percentage of Customers (%)', fontsize=12)
plt.legend(title='Churn Status (1 = Exited)', bbox_to_anchor=(1.05, 1), loc='upper left') #Moves the legend outside the chart so it doesn't cover the bars
plt.xticks(rotation=0) #Keep the country names flat (not sideways)
plt.tight_layout()
plt.show()

#-------Gender vs Churn
Per_Gender = (pd.crosstab(index=df['Gender'], columns=df['Exited'], normalize='index')*100).round(1)
print("\n----Churn Rates By Gender----\n")
print(f"{Per_Gender}")
Per_Gender.plot(
    kind='bar', 
    stacked=True, 
    figsize=(8, 6), 
    color=['#2ca02c', '#d62728'] # Green for staying (0), Red for leaving (1)
)
plt.title('Customer Churn Rate by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Percentage of Customers (%)', fontsize=12)
plt.legend(title='Churn Status (1 = Exited)', bbox_to_anchor=(1.05, 1), loc='upper left') #Moves the legend outside the chart so it doesn't cover the bars
plt.xticks(rotation=0) #Keep the country names flat (not sideways)
plt.tight_layout()
plt.show()

#Geo vs Gender by Churn
print("\n-------Churn by Gender and Geography-------\n")
geo_gender_pivot = df.pivot_table(
    columns='Gender',
    index='Geography',
    values=['Exited'],
    aggfunc={'Exited': 'count'}
)
geo_gender_pivot.plot(kind='barh', stacked=True)
plt.title('Exited Customers by Gender and Geography')
plt.ylabel('')
plt.xlabel('Count of Exited')
plt.show()

#--------Age vs Churn
print("\n----Churn Rates By Age----\n")
sns.boxplot(x='Exited', y='Age', data=df)
plt.show()
print("---Interpretation and Insight:\n")
print("1. Non-churned (0): Median age Customers who stay is 35 years \n2. Exited (1): Median age of Customers who exited is 45 years\n")
print("Insight: Older Customers exit more\n")

#-------Geo and Active by Churn
print("\n----Churn Rates By Activeness and Geography----\n")
activee = round(df.pivot_table(
    values='Exited',
    index='Geography',
    columns='IsActiveMember',
    aggfunc='mean'
) * 100, 2)
print(activee)
sns.heatmap(activee, annot=activee.round(1).astype(str) + '%', cmap='coolwarm', fmt='')
plt.title('Churn Rate (%) by Geography & Activity - Heatmap')
plt.show()
print("---Interpretation and Insight:\n")
print("-Non-active Customers exit more than Active Customers")

#--------Num of Products vs Churn
print("\n----Churn Rates By NumOfProducts----\n")
sns.boxplot(x='Exited', y='NumOfProducts', data=df)
plt.show()
print("---Interpretation and Insight:\n")
print("\033[1mNumber of Products is not a strong reason why Customers leave, but Customers who use more services are more “locked in” and less likely to leave\033[0m\nInterpretations...\n1. One outlier is an exited customer with 4 products\n2. Customers with 2+ products churn slightly less\n3. Customers with only 1 product are more likely to churn\n")

#---------MACHINE LEARNING MODELS-----------
print("--------MACHINE LEARNING---------")
df_ml = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')
df_ml = pd.get_dummies(df_ml, columns=['Geography', 'Gender'], drop_first=True, dtype=int)
vip_features = [
    'Age', 
    'IsActiveMember', 
    'Balance', 
    'NumOfProducts', 
    'Geography_Germany', 
    'Geography_Spain', 
    'Gender_Male'
]
X = df_ml[vip_features]
y = df_ml['Exited']
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)
#----Random Forest----
print("\n------Random Forest in action------\n")
#Training the model
rf_model = RFC(n_estimators=100, random_state=42, class_weight='balanced') #class-weight to balance the algorithm
rf_model.fit(X_train, y_train)
#Test
predictions = rf_model.predict(X_test)
print("Optimized Model Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
#-------Confusion Matrix For RFC--------
print("\n------CONFUSION MATRIX FOR RANDOM FOREST-------\n")
cm = c_m(y_test, predictions)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix For Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# SMOTE Algorithm to balance the algorithm
print("\n-----SMOTE ALGORITHM-------\n")
smote = SMOTE(random_state=42)
#Resampling without our initial test variables
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Original Churners in Training: {sum(y_train == 1)}")
print(f"SMOTE Churners in Training: {sum(y_train_smote == 1)}\n")

#Back to random forest using the SMOTE result
rf_smote = RFC(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

smote_preds = rf_smote.predict(X_test)

print("Optimized Model Accuracy:", accuracy_score(y_test, smote_preds))
print("Classification Report (SMOTE):\n", classification_report(y_test, smote_preds))
print("WITHOUT SMOTE")
print(classification_report(y_test, predictions))

print("\nWITH SMOTE")
print(classification_report(y_test, smote_preds))
print("\n\033[1mAccuracy Score should be overlooked since no model can be accurate in this contex\033[0m")
#-------Confusion Matrix for SMOTE--------
print("\n------CONFUSION MATRIX FOR SMOTE-------\n")
cm = c_m(y_test, smote_preds)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix For SMOTE')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#--------FEATURE IMPORTANCE---------
print("\n--- GENERATING FEATURE IMPORTANCE CHART ---")

importances = rf_smote.feature_importances_
feature_names = X_train_smote.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=importance_df, 
    palette='magma'
)
plt.title('Top Drivers of Customer Churn (Random Forest)', fontsize=16, fontweight='bold')
plt.xlabel('Relative Importance Score', fontsize=12)
plt.ylabel('')
plt.tight_layout()
plt.show()

print("\n--- FEATURE IMPORTANCE INTERPRETATION ---")
for i, row in importance_df.head(5).iterrows():
    print(f"{row['Feature']} is a strong driver of churn (importance: {row['Importance']:.3f})")
    
print("Because of the SMOTE Algorithm, our data is making balance to cheat the system. Hence, we rely on the importance of the raw data: Permutation Importance.\n")
    
#-------PERMUTATION IMPORTANCE--------
print("\n---RUNNING PERMUTATION IMPORTANCE---")

perm_results = permutation_importance(
    rf_smote, 
    X_test, 
    y_test, 
    n_repeats=10, #10 to shuffle each col 10x to get a fairer average
    random_state=42
)
true_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Accuracy_Drop': perm_results.importances_mean
})
true_importance = true_importance.sort_values(by='Accuracy_Drop', ascending=False)
print(f"Permutation Importance: {true_importance}")
print("\n--- PERMUTATION IMPORTANCE INTERPRETATION---\n")
for i, row in true_importance.iterrows():
    print(f"{row['Feature']} is strong driver of churn (importance: {row['Accuracy_Drop']:.3f})")
    
proba = rf_smote.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, proba)

plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print("\n===== FINAL INSIGHTS =====")

print("""
1. Customers in Germany have the highest churn rate
2. Older customers are more likely to churn
3. Inactive members are at highest risk
4. Customers with fewer products are more likely to leave

RECOMMENDATIONS:
- Target inactive customers with engagement campaigns
- Focus retention efforts on Germany region
- Offer bonuses to high-balance customers
- Cross-sell products to increase retention
""")

#Saving to joblib
#joblib.dump(rf_smote, "churn_prediction_model.pkl")
#print("Model sucessfully saved to disk!")

def predict_churn(input_data):
    model = joblib.load("churn_prediction_model.pkl")
    return model.predict(input_data)