## Dataset Structure
1. Data set shape 614 records and13 columns
2. Dataset information
RangeIndex: 614 entries, 0 to 613
Data columns (total 13 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Loan_ID            614 non-null    str    
 1   Gender             601 non-null    str    
 2   Married            611 non-null    str    
 3   Dependents         599 non-null    str    
 4   Education          614 non-null    str    
 5   Self_Employed      582 non-null    str    
 6   ApplicantIncome    614 non-null    int64  
 7   CoapplicantIncome  614 non-null    float64
 8   LoanAmount         592 non-null    float64
 9   Loan_Amount_Term   600 non-null    float64
 10  Credit_History     564 non-null    float64
 11  Property_Area      614 non-null    str    
 12  Loan_Status        614 non-null    str    
dtypes: float64(4), int64(1), str(8)

## No Duplicates In The Dataset

## Columns With Null Values To Handle
1. Categorical Features
Gender: 13 null values , missing percentage(2.12%)
Married: 3 null values , missing percentage(0.49%)
Dependents: 15 null values , missing percentage(2.44%)
Self_Employed: 32 null values , missing percentage(5.21%)

2. Numerical Features
LoanAmount: 22 null values, missing percentage(3.58%)
Loan_Amount_Term: 14 null values, missing percentage(2.28%)
Credit_History: 50 null values, missing percentage(8.14%)

## Column Names With Inconsistencies
ApplicantIncome
CoapplicantIncome 
LoanAmount 

## Data Inconsistencies
Dependents column has an input of 3+ and it needs to be specified with the maximum number so that it cannot disturb calculations

## Data Distribution
1. Categorical Distribution
![alt text](<Images/Gender Distribution.png>)
![alt text](<Images/Marriage Distribution.png>)
![alt text](<Images/Dependents Distribution.png>)
![alt text](<Images/Self Employed Distribution.png>)

2. Numerical Distribution