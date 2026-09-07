## Dataset Structure
1. Dataset shape 614 records and 13 columns

## Dataset information

<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; text-align: left; font-family: sans-serif;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>#</th>
      <th>Column</th>
      <th>Non-Null Count</th>
      <th>Dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td><strong>Loan_ID</strong></td>
      <td>614 non-null</td>
      <td>str</td>
    </tr>
    <tr>
      <td>1</td>
      <td><strong>Gender</strong></td>
      <td>601 non-null</td>
      <td>str</td>
    </tr>
    <tr>
      <td>2</td>
      <td><strong>Married</strong></td>
      <td>611 non-null</td>
      <td>str</td>
    </tr>
    <tr>
      <td>3</td>
      <td><strong>Dependents</strong></td>
      <td>599 non-null</td>
      <td>str</td>
    </tr>
    <tr>
      <td>4</td>
      <td><strong>Education</strong></td>
      <td>614 non-null</td>
      <td>str</td>
    </tr>
    <tr>
      <td>5</td>
      <td><strong>Self_Employed</strong></td>
      <td>582 non-null</td>
      <td>str</td>
    </tr>
    <tr>
      <td>6</td>
      <td><strong>ApplicantIncome</strong></td>
      <td>614 non-null</td>
      <td>int64</td>
    </tr>
    <tr>
      <td>7</td>
      <td><strong>CoapplicantIncome</strong></td>
      <td>614 non-null</td>
      <td>float64</td>
    </tr>
    <tr>
      <td>8</td>
      <td><strong>LoanAmount</strong></td>
      <td>592 non-null</td>
      <td>float64</td>
    </tr>
    <tr>
      <td>9</td>
      <td><strong>Loan_Amount_Term</strong></td>
      <td>600 non-null</td>
      <td>float64</td>
    </tr>
    <tr>
      <td>10</td>
      <td><strong>Credit_History</strong></td>
      <td>564 non-null</td>
      <td>float64</td>
    </tr>
    <tr>
      <td>11</td>
      <td><strong>Property_Area</strong></td>
      <td>614 non-null</td>
      <td>str</td>
    </tr>
    <tr>
      <td>12</td>
      <td><strong>Loan_Status</strong></td>
      <td>614 non-null</td>
      <td>str</td>
    </tr>
  </tbody>
</table>



## No Duplicates In The Dataset

## Columns With Null Values To Handle
### Categorical Features
1. Gender: 13 null values , missing percentage(2.12%)
2. Married: 3 null values , missing percentage(0.49%)
3. Dependents: 15 null values , missing percentage(2.44%)
4. Self_Employed: 32 null values , missing percentage(5.21%)

### Numerical Features
1. LoanAmount: 22 null values, missing percentage(3.58%)
2. Loan_Amount_Term: 14 null values, missing percentage(2.28%)
3. Credit_History: 50 null values, missing percentage(8.14%)

## Columns With Inconsistencies
1. ApplicantIncome
2. CoapplicantIncome 
3. LoanAmount 

## Data Inconsistencies
Dependents column has an input of 3+ and it needs to be specified with the maximum number so that it cannot disturb calculations

## Data Distribution
1. Categorical Distribution

![alt text](<Images/Gender Distribution.png>)

![alt text](<Images/Marriage Distribution.png>)

![alt text](<Images/Dependents Distribution.png>)

![alt text](<Images/Self Employed Distribution.png>)

2. Numerical Distribution