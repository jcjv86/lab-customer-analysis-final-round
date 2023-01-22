#Library imports

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set_theme()
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sklearn.metrics as metrics

#Functions for labs


def clean_customer_lifetime_value(x):
    '''
This function is called in Lab 1 as an alternative to pd.to_numeric. Since the goal of that lab is creating the function 
I will leave it outside (as if I nested it inside it would not be able to overwrite the dataframe inside the main function).
I could also define it as a loop inside the main function but the purpose of this exercise would be lost, so I opted to leave it
outside and call it from inside the main function.
'''
    if type(x) == str:
        x = x.replace('%', '')
    return float(x)
        
def lab_1():
    file1 = pd.read_csv('./files_for_lab/csv_files/file1.csv')
    file2 = pd.read_csv('./files_for_lab/csv_files/file2.csv')
    file3 = pd.read_csv('./files_for_lab/csv_files/file3.csv')
    print('File 1 head and shape\n')
    display(file1.head())
    display(file1.shape)
    print('\nFile 2 head and shape\n')
    display(file2.head())
    display(file2.shape)
    print('\nFile 3 head and shape\n')
    display(file3.head())
    display(file3.shape)
    
    #Standardize columns
    cols1 = []
    for col in file1.columns:
        cols1.append(col.lower().replace(' ', '_'))
    file1.columns = cols1
    cols2 = []
    for col in file2.columns:
        cols2.append(col.lower().replace(' ', '_'))
    file2.columns = cols2
    file3.rename(columns = {'State':'st'}, inplace = True)
    cols3 = []
    for col in file3.columns:
        cols3.append(col.lower().replace(' ', '_'))
    file3.columns = cols3
    print('\nThese are now the files standardized columns: \n')
    display(file1.columns)
    display(file2.columns)
    display(file3.columns)
    file2 = file2[['customer', 'st', 'gender', 'education', 'customer_lifetime_value', 'income', 'monthly_premium_auto', 'number_of_open_complaints', 'policy_type', 'vehicle_class', 'total_claim_amount']]
    file3 = file3[['customer', 'st', 'gender', 'education', 'customer_lifetime_value', 'income', 'monthly_premium_auto', 'number_of_open_complaints', 'policy_type', 'vehicle_class', 'total_claim_amount']]
    
    #Concat 3 files into DF
    data = pd.concat([file1, file2, file3], axis=0)
    print('\nThis is the new dataframe after concatenating all 3 files:\n')
    data = data.reset_index(drop=True)
    display(data.head())
    display(data.shape)
    
    #Numerical and categorical
    print('\nThese are the numerical columns:\n')
    numerical = data.select_dtypes(np.number)
    display(numerical.head())
    print('\nThese are the categorical columns (please note some that should be numerical are listed as categorical since their format is wrong):\n')
    categorical_data = data.select_dtypes(object)
    display(categorical_data.head())
    
    print('''
    #customer: customer reference
    #st: state
    #gender: gender of the customer
    #education: educational level of the customer
    #customer lifetime value: estimation of the amount of revenue a customer will generate over the course of their relationship with the brand.
    #income: income of the customer
    #monthly_premium_auto: amount customer pays to insurance company on a regular basis, often every month or every six months, in exchange for insurance coverage.
    #number_of_open_complaints: unsure, could be internal info or data be corrupt.
    #policy_type: as it states, type of policy depending on the car type (in this dataframe: personal, corporate or special).
    #vehicle_class: see below as there are 9 types
    #total_claim_amount: This is the dollar amount an insurance company paid for damages to or replacement of an insured vehicle
    ''')
    print('\n\nThese are the possible values for the column vehicle class\n')
    display(data['vehicle_class'].value_counts())
    print('\nDropping Education and Number of open complaints columns')
    data = data.drop(['education', 'number_of_open_complaints'], axis=1)
    display(data.head())
    data_copy = data.copy()

    #Clean customer lifetime value
    print('\nCleaning Customer lifetime value: with function and pd.to_numeric:\n\n')
    print('\nAfter applying our user-defined function:\n')
    data['customer_lifetime_value'] = data['customer_lifetime_value'].apply(clean_customer_lifetime_value)
    display(data.head())
    data.info()
    print('\nWe can clearly see they have been converted to float successfully\n\n')
    print('\nAfter applying pd.to_numeric:\n')
    data_copy['customer_lifetime_value'] =  pd.to_numeric(data_copy['customer_lifetime_value'], errors='coerce')
    display(data_copy.head())
    print('\nWe can clearly see they have been converted to float successfully as well\n\n')
    data_copy.info()
    print('\nNAs aafter applying the function:')
    display(data['customer_lifetime_value'].isna().sum())
    print('\nNAs aafter applying pd.to_numeric:')
    display(data_copy['customer_lifetime_value'].isna().sum())
    print('\nWe will stick with the database we applied our user defined function to. We will now divide by 100 and round to 2 decimals:\n')
    data['customer_lifetime_value'] = round(data['customer_lifetime_value']/100, 2)
    display(data)
    display(data.shape)
    print('\nWe will remove the duplicates')
    data = data.drop_duplicates()
    display(data)
    display(data.shape)
    print('\nWe will finally filter out the customers with income 0 or less:\n')
    data = data[(data['income']>0)]
    display(data)
    display(data.shape)
    return


def lab_2():
    df = pd.read_csv('./files_for_lab/csv_files/marketing_customer_analysis_lab2.csv')
    display(df)
    df.info()
    df = df.drop_duplicates()
    df = df.drop([df.columns[0]], axis=1) # Dropped unname column as it is irrelevant
    df = df.drop([df.columns[3]], axis=1) #Dropped Response column
    df = df.drop([df.columns[-1]], axis=1) #Dropped vehicle type since the data is massively corrupted
    df = df.drop([df.columns[-6]], axis=1) #Dropped policy column as it has the same info as Policy Type
    df = df.drop([df.columns[-8]], axis=1) #Dropped number_of open complaints as it is non relevant
    #Since the column drops modify the columns indexes, we need to consider them in order to remove the desired ones.
    df = df.reset_index(drop = True)

    df.columns
    #shape
    print("Shape: ", df.shape, "\n")
    #standardize headers
    cols = []
    for col in df.columns:
        cols.append(col.lower().replace(' ', '_'))
    df.columns = cols
    print("The new headers are: \n", cols, "\n")
    #displays the numerical columns
    numerical = df.select_dtypes(np.number)
    print("The numerical columns are: \n")
    display(numerical)
    #which columns are cathegorical? Therefore non numerical
    categorical = df.select_dtypes(object)
    print("The categorical columns are: \n")
    display(categorical)
    #Rounding up total claim amount and customer lifetime value columns to 2 decimals
    df['total_claim_amount'] = df['total_claim_amount'].round(decimals = 2)
    df['customer_lifetime_value'] = df['customer_lifetime_value'].round(decimals = 2)
    #Dealing with NaN values - we decided to fill in vehicle class and vehicle size with the mode values.
    df['vehicle_class'] = df['vehicle_class'].fillna(df['vehicle_class'].mode()[0])
    df['vehicle_size'] = df['vehicle_size'].fillna(df['vehicle_size'].mode()[0])
    #Adding a column with the month inside "effective_to_date" column:
    df['month'] = pd.DatetimeIndex(df['effective_to_date']).month
    print("\nThis is the fully transformed dataframe: \n")
    display(df)
    print("\nThis is the information for the first quarter: \n")
    display(df[(df['month'] <=3)])
    
    print('\n\n**Dealing with NaN values - Considerations**\n')
    print('\nSome info for context:')
    df.info()
    
    jobs = df[(df['income'] > 0) & (df['employmentstatus'] !='Unemployed')]
    nojobs = df[(df['income'] == 0) & (df['employmentstatus'] =='Unemployed')]
    print('\nThese are the customers with jobs:')
    display(jobs.shape)
    print('\nAnd these are the customers without jobs:')
    display(nojobs.shape)
    print('''\n\n
    We can clearly see that the number of unemployed and also income 0 customers is bvery significant. Some may even have high customer values
    and expensive policies, so as long as they pay their quotas they are relevant data. We will not get rid of these values.
    
    We can consider to fill out Vehicle Size and Vehicle Class NaN values, since they are around 600 (6% of the total). Since we are talking about a 
    categorical value, we cannot use mean or median to fill them out, so we have used the mode. We have done this with the function already.
    \n\n''')
    display(df['vehicle_size'].value_counts(dropna = False))
    display(df['vehicle_class'].value_counts(dropna = False))
    print('\nThere are no longer NaN values on both columns')
    
    return    


def lab_3(df):
    df = df.drop_duplicates()
    df = df.reset_index(drop = True)
    print('This is the original dataframe as loaded from the .csv file')
    display(df)
    print('These are the original Dataframe column names')
    display(df.columns)
    #Standardize headers
    cols = []
    for col in df.columns:
        cols.append(col.lower().replace(' ', '_'))
    df.columns = cols
    #Rounding up total claim amount and customer lifetime value columns to 2 decimals
    df['total_claim_amount'] = df['total_claim_amount'].round(decimals = 2)
    df['customer_lifetime_value'] = df['customer_lifetime_value'].round(decimals = 2)
    #Displaying Df info
    print("This is the dataframe info: \n")
    display(df.info())
    #Describe Df
    print("This is the dataframe description: \n")
    display(df.describe())
    fig, axes = plt.subplots(2,2, figsize=(10,8), dpi=200)
    #Total responses
    sns.histplot(data=df, x='response', hue='response', ax = axes[0,0])
    axes[0,0].set_xlabel('Response')
    axes[0,1].set_ylabel('')
    axes[0,0].set_title('Total Responses (%)')
    #Sales channel VS response
    sns.histplot(data=df, x='response', hue='sales_channel', ax = axes[0,1])
    axes[0,1].set_xlabel('Response')
    axes[0,1].set_ylabel('')
    axes[0,1].set_title('Sales Channel VS response')
    #Total claim amount VS Response
    sns.histplot(data = df, x='total_claim_amount', hue='response', ax = axes[1,0])
    axes[1,0].set_xlabel('Total claim amount')
    axes[1,0].set_ylabel('')
    axes[1,0].set_title('Total claim amount VS Response')
    #Response VS income
    sns.histplot(data = df, x='income', hue='response', ax = axes[1,1])
    axes[1,1].set_xlabel('Income')
    axes[1,1].set_ylabel('')
    axes[1,1].set_title('Income VS Response')
    plt.tight_layout()
    return

def lab_4(df):
    df = df.drop_duplicates()
    df = df.reset_index(drop = True)
    print('This is the original dataframe as loaded from the .csv file:')
    display(df)
    print('\nThese are the original Dataframe column names:\n')
    display(df.columns)
    #Standardize headers
    cols = []
    for col in df.columns:
        cols.append(col.lower().replace(' ', '_'))
    df.columns = cols
    #Rounding up total claim amount and customer lifetime value columns to 2 decimals
    df['total_claim_amount'] = df['total_claim_amount'].round(decimals = 2)
    df['customer_lifetime_value'] = df['customer_lifetime_value'].round(decimals = 2)
    #Data types
    print('\nThese are the data types for all columns (headers already standardized):\n')
    display(df.dtypes)
    #Numerical and categorical columns
    numerical = df.select_dtypes(include = np.number)
    categorical = df.select_dtypes(include='object')
    print('\nThese are the numerical columns')
    display(numerical)
    print('These are the categorical columns')
    display(categorical)
    #Plots made with Seaborn
    print('\nThese are the graphs made with Seaborn library\n')
    fig, axes = plt.subplots(4,2,figsize=(20,20), dpi=300)
    sns.histplot(data = numerical, x='customer_lifetime_value', ax = axes[0,0])
    axes[0,0].set_xlabel('Customer Lifetime Value')
    sns.histplot(data = numerical, x='income', ax = axes[0,1])
    axes[0,1].set_xlabel('Income')
    sns.histplot(data = numerical, x='monthly_premium_auto', ax = axes[1,0])
    axes[1,0].set_xlabel('Monthly Premium Auto')
    sns.histplot(data = numerical, x='months_since_last_claim', ax = axes[1,1])
    axes[1,1].set_xlabel('Months Since Last Claim')
    sns.histplot(data = numerical, x='months_since_policy_inception', ax = axes[2,0])
    axes[2,0].set_xlabel('Months Since Policy Inception')
    sns.histplot(data = numerical, x='number_of_open_complaints', ax = axes[2,1])
    axes[2,1].set_xlabel('Number of Open Complaints')
    sns.histplot(data = numerical, x='number_of_policies', ax = axes[3,0])
    axes[3,0].set_xlabel('Number of Policies')
    sns.histplot(data = numerical, x='total_claim_amount', ax = axes[3,1])
    axes[3,1].set_xlabel('Total Claim Amount')
    plt.tight_layout()
    plt.show()
    #Plots made with matplotlib
    print('\nThese are the histograms made in matplotlib')
    fig, axes = plt.subplots(4,2,figsize=(20,20), dpi=300)
    axes[0,0].hist(x=numerical['customer_lifetime_value'], bins=50)
    axes[0,0].set_xlabel('Customer Lifetime Value')
    axes[0,1].hist(x=numerical['income'], bins=50)
    axes[0,1].set_xlabel('Income')
    axes[1,0].hist(x=numerical['monthly_premium_auto'], bins=50)
    axes[1,0].set_xlabel('Monthly Premium Auto')
    axes[1,1].hist(x=numerical['months_since_last_claim'], bins=50)
    axes[1,1].set_xlabel('Months Since Last Claim')
    axes[2,0].hist(x=numerical['months_since_policy_inception'], bins=50)
    axes[2,0].set_xlabel('Months Since Policy Inception')
    axes[2,1].hist(x=numerical['number_of_open_complaints'], bins=50)
    axes[2,1].set_xlabel('Number of Open Complaints')
    axes[3,0].hist(x=numerical['number_of_policies'], bins=50)
    axes[3,0].set_xlabel('Number of Policies')
    axes[3,1].hist(x=numerical['total_claim_amount'], bins=50)
    axes[3,1].set_xlabel('Total Claim Amount')
    plt.tight_layout()
    plt.show()
    #Do the distributions for different numerical variables look like a normal distribution?
    print('\nDo the distributions for different numerical variables look like a normal distribution?')
    sns.pairplot(numerical)
    plt.show()
    print('Far from it')
    print('\nWe will check Months since last claim and Months since policy inception as they look quite evently distributed')
    fig, axes = plt.subplots(1,2,figsize=(15,5), dpi=200)
    sns.boxplot(data = numerical, x='months_since_last_claim', ax = axes[0])
    axes[0].set_xlabel('Months Since Last Claim')
    sns.boxplot(data = numerical, x='months_since_policy_inception', ax = axes[1])
    axes[1].set_xlabel('Months Since Policy Inception')
    plt.tight_layout()
    plt.show()
    print('\nCheking how different the Income column is by including or excluding 0 income values')
    income0excl = []
    for i in df['income']:
        if i>0:
            income0excl.append(i)
        else:
            pass
    income_positive = pd.DataFrame (income0excl, columns = ['income'])
    fig, axes = plt.subplots(1,2,figsize=(15,5), dpi=200)
    sns.boxplot(data = df, x='income', ax=axes[0])
    axes[0].set_xlabel('Income - including 0 values')
    sns.boxplot(data = income_positive, x='income', ax=axes[1])
    axes[1].set_xlabel('Income - excluding 0 values')
    plt.tight_layout()
    plt.show()
    print('\nWe can also see the differences via histograms:')
    fig, axes = plt.subplots(1,2,figsize=(15,5), dpi=200)
    sns.histplot(data = df, x='income', ax = axes[0])
    axes[0].set_xlabel('Income - including 0 values')
    sns.histplot(data = income_positive, x='income', ax = axes[1])
    axes[1].set_xlabel('Income - excluding 0 values')
    plt.tight_layout()
    plt.show()
    print('\nWe will check the multicorrelation via matrix and heatmap')
    correlations_matrix = numerical.corr()
    display(correlations_matrix)
    sns.heatmap(correlations_matrix, annot=True)
    plt.show()
    print('Total claim amount and monthly premium auto are the only values with a minimal correlation (0.63), so we will not drop any columns')
    return

def lab_5(df):
    df = df.drop_duplicates()
    df = df.reset_index(drop = True)
    #Standardize headers
    cols = []
    for col in df.columns:
        cols.append(col.lower().replace(' ', '_'))
    df.columns = cols
    #Rounding up total claim amount and customer lifetime value columns to 2 decimals
    df['total_claim_amount'] = df['total_claim_amount'].round(decimals = 2)
    df['customer_lifetime_value'] = df['customer_lifetime_value'].round(decimals = 2)
    df = df.select_dtypes(include = np.number)
    X = df.drop(['total_claim_amount'], axis=1)
    y = df['monthly_premium_auto']
    print('This is dataframe X')
    display(X)
    print('\nThis is dataframe y')
    display(y)
    X_nrm = X.copy()
    X_std = X.copy()
    transformer = MinMaxScaler().fit(X_nrm)
    x_normalized = transformer.transform(X_nrm)
    x_normalized = pd.DataFrame(x_normalized, columns=X_nrm.columns)
    print('\nThis is X after normalization with MinMaxScaler')
    display(x_normalized)
    transformer2 = StandardScaler().fit(X_std)
    x_standardized = transformer2.transform(X_std)
    x_standardized = pd.DataFrame(x_standardized, columns=X_std.columns)
    print('\nThis is X after Standardization with StandardScaler')
    display(x_standardized)
    fig, axes = plt.subplots(7, 3, figsize=(15,15), dpi=500)
    #Raw data graphs
    sns.histplot(data=X, x='customer_lifetime_value',ax=axes[0][0])
    axes[0,0].set_xlabel('Customer Lifetime Value')
    sns.histplot(data=X, x='income',ax=axes[1][0])
    axes[1,0].set_xlabel('Income')
    sns.histplot(data=X, x='monthly_premium_auto',ax=axes[2][0])
    axes[2,0].set_xlabel('Monthly Premium Auto')
    sns.histplot(data=X, x='months_since_last_claim',ax=axes[3][0])
    axes[3,0].set_xlabel('Months Since Last Claim')
    sns.histplot(data=X, x='months_since_policy_inception',ax=axes[4][0])
    axes[4,0].set_xlabel('Months Since Policy Inception')
    sns.histplot(data=X, x='number_of_open_complaints',ax=axes[5][0])
    axes[5,0].set_xlabel('Number of Policies')
    sns.histplot(data=X, x='number_of_policies',ax=axes[6][0])
    axes[6,0].set_xlabel('Total Claim Amount')
    #Data normalized graphs
    sns.histplot(data=x_normalized, x='customer_lifetime_value',ax=axes[0][1])
    axes[0,1].set_xlabel('Customer Lifetime Value\nNormalized')
    sns.histplot(data=x_normalized, x='income',ax=axes[1][1])
    axes[1,1].set_xlabel('Income\nNormalized')
    sns.histplot(data=x_normalized, x='monthly_premium_auto',ax=axes[2][1])
    axes[2,1].set_xlabel('Monthly Premium Auto\nNormalized')
    sns.histplot(data=x_normalized, x='months_since_last_claim',ax=axes[3][1])
    axes[3,1].set_xlabel('Months Since Last Claim\nNormalized')
    sns.histplot(data=x_normalized, x='months_since_policy_inception',ax=axes[4][1])
    axes[4,1].set_xlabel('Months Since Policy Inception\nNormalized')
    sns.histplot(data=x_normalized, x='number_of_open_complaints',ax=axes[5][1])
    axes[5,1].set_xlabel('Number of Policies\nNormalized')
    sns.histplot(data=x_normalized, x='number_of_policies',ax=axes[6][1])
    axes[6,1].set_xlabel('Total Claim Amount\nNormalized')
    #Data standardized graphs
    sns.histplot(data=x_standardized, x='customer_lifetime_value',ax=axes[0][2])
    axes[0,2].set_xlabel('Customer Lifetime Value\nStandardized')
    sns.histplot(data=x_standardized, x='income',ax=axes[1][2])
    axes[1,2].set_xlabel('Incomen\nStandardized')
    sns.histplot(data=x_standardized, x='monthly_premium_auto',ax=axes[2][2])
    axes[2,2].set_xlabel('Monthly Premium Auton\nStandardized')
    sns.histplot(data=x_standardized, x='months_since_last_claim',ax=axes[3][2])
    axes[3,2].set_xlabel('Months Since Last Claim\nStandardized')
    sns.histplot(data=x_standardized, x='months_since_policy_inception',ax=axes[4][2])
    axes[4,2].set_xlabel('Months Since Policy Inception\nStandardized')
    sns.histplot(data=x_standardized, x='number_of_open_complaints',ax=axes[5][2])
    axes[5,2].set_xlabel('Number of Policies\nStandardized')
    sns.histplot(data=x_standardized, x='number_of_policies',ax=axes[6][2])
    axes[6,2].set_xlabel('Total Claim Amount\nStandardized')
    plt.tight_layout()
    plt.show()
    return

def lab_6(df):
    df = df.drop_duplicates()
    df = df.reset_index(drop = True)
    #Standardize headers
    cols = []
    for col in df.columns:
        cols.append(col.lower().replace(' ', '_'))
    df.columns = cols
    df = df.select_dtypes(include = np.number)
    d2 = df.copy()
    X = df.drop(['total_claim_amount'], axis=1)
    y = df['total_claim_amount']
    X.describe().T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)
    X_train.head()
    y_train.head()
    print('\nLinear regression, X and y train / test\n')
    print('This is the shape of X train: \n', X_train.shape)
    print('This is the shape of X test: \n', X_test.shape)
    print('This is the shape of y train: \n', y_train.shape)
    print('This is the shape of y test: \n', y_test.shape)
    print()
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    y_pred_train = lm.predict(X_train)
    y_pred_test = lm.predict(X_test)
    print('\nError tests\n')
    print(f'R2 train = {r2_score(y_train, y_pred_train):.4f}')
    print(f'R2 test = {r2_score(y_test, y_pred_test):.4f}')
    print()
    print(f'RMSE train = {(np.sqrt(mean_squared_error(y_train,y_pred_train))):.4f}')
    print(f'RMSE test = {(np.sqrt(mean_squared_error(y_test,y_pred_test))):.4f}')
    print()
    print (f'MAE train = {(metrics.mean_absolute_error(y_train, y_pred_train)):.4f}')
    print (f'MAE test = {(metrics.mean_absolute_error(y_test, y_pred_test)):.4f}')
    print()
    print (f'MSE train = {(metrics.mean_squared_error(y_train, y_pred_train)):.4f}')
    print (f'MAE test = {(metrics.mean_squared_error(y_test, y_pred_test)):.4f}')
    print()
    
    #POWER TRANSFORMER
    transformer1 = PowerTransformer().fit(d2)
    dpt = transformer1.transform(d2)
    dpt = pd.DataFrame(dpt, columns=d2.columns)
    print('\nThis is the dataframe after using PowerTransformer')
    display(dpt)
    ypt = dpt['total_claim_amount']
    Xpt = dpt.drop(['total_claim_amount'], axis=1)
    Xpt_train, Xpt_test, ypt_train, ypt_test = train_test_split(Xpt, ypt, test_size=0.2, random_state=86)
    print('\nLinear regression, X and y train / test\n')
    print('This is the shape of X train: \n', Xpt_train.shape)
    print('This is the shape of X test: \n', Xpt_test.shape)
    print('This is the shape of y train: \n', ypt_train.shape)
    print('This is the shape of y test: \n', ypt_test.shape)
    print()
    lmpt = LinearRegression()
    lmpt.fit(Xpt_train,ypt_train)
    ypt_pred_train = lmpt.predict(Xpt_train)
    ypt_pred_test = lmpt.predict(Xpt_test)
    print('\nError tests after applying PowerTransformer to the data:\n')
    print(f'R2 train = {r2_score(ypt_train, ypt_pred_train):.4f}')
    print(f'R2 test = {r2_score(ypt_test, ypt_pred_test):.4f}')
    print()
    print(f'RMSE train = {(np.sqrt(mean_squared_error(ypt_train,ypt_pred_train))):.4f}')
    print(f'RMSE test = {(np.sqrt(mean_squared_error(ypt_test,ypt_pred_test))):.4f}')
    print()
    print (f'MAE train = {(metrics.mean_absolute_error(ypt_train, ypt_pred_train)):.4f}')
    print (f'MAE test = {(metrics.mean_absolute_error(ypt_test, ypt_pred_test)):.4f}')
    print()
    print (f'MSE train = {(metrics.mean_squared_error(ypt_train, ypt_pred_train)):.4f}')
    print (f'MAE test = {(metrics.mean_squared_error(ypt_test, ypt_pred_test)):.4f}')
    
    #MinMaxScaler (Normalized)
    transformer2 = MinMaxScaler().fit(dpt)
    dnrm = transformer2.transform(dpt)
    dnrm = pd.DataFrame(dnrm, columns=d2.columns)
    ynrm = dnrm['total_claim_amount']
    Xnrm = dnrm.drop(['total_claim_amount'], axis=1)
    print('\nThis is the dataframe after applying MinMaxScaler to the PowerTransfored data:')
    display(dnrm)
    Xnrm_train, Xnrm_test, ynrm_train, ynrm_test = train_test_split(Xnrm, ynrm, test_size=0.2, random_state=86)
    print('\nLinear regression, X and y train / test\n')
    print('This is the shape of X train: \n', Xnrm_train.shape)
    print('This is the shape of X test: \n', Xnrm_test.shape)
    print('This is the shape of y train: \n', ynrm_train.shape)
    lmnrm = LinearRegression()
    lmnrm.fit(Xnrm_train,ynrm_train)
    ynrm_pred_train = lmpt.predict(Xnrm_train)
    ynrm_pred_test = lmpt.predict(Xnrm_test)
    print('\nError tests after applying Standard Scaler to the Power Transformed dataset:\n')
    print(f'R2 train = {r2_score(ynrm_train, ynrm_pred_train):.4f}')
    print(f'R2 test = {r2_score(ynrm_test, ynrm_pred_test):.4f}')
    print()
    print(f'RMSE train = {(np.sqrt(mean_squared_error(ynrm_train,ynrm_pred_train))):.4f}')
    print(f'RMSE test = {(np.sqrt(mean_squared_error(ynrm_test,ynrm_pred_test))):.4f}')
    print()
    print (f'MAE train = {(metrics.mean_absolute_error(ynrm_train, ynrm_pred_train)):.4f}')
    print (f'MAE test = {(metrics.mean_absolute_error(ynrm_test, ynrm_pred_test)):.4f}')
    print()
    print (f'MSE train = {(metrics.mean_squared_error(ynrm_train, ynrm_pred_train)):.4f}')
    print (f'MAE test = {(metrics.mean_squared_error(ynrm_test, ynrm_pred_test)):.4f}')
    return
