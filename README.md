<H3>ENTER YOUR NAME: Naveen Kumar V</H3>
<H3>ENTER YOUR REGISTER NO.212223230140</H3>
<H3>EX. NO.1</H3>
<H3>DATE: -04-26</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")
df.head()

df.isnull().sum()    

df.duplicated().sum()

df.describe()

df = df.drop(['Surname', 'Geography', 'Gender'], axis=1)

scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)

x = df.iloc[:, :-1].values
x

y = df.iloc[:, -1].values
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:
<img width="1247" height="241" alt="image" src="https://github.com/user-attachments/assets/7af4a785-c84e-442d-8266-e33576e81237" /><br>
<img width="208" height="307" alt="image" src="https://github.com/user-attachments/assets/4d715a8e-3eed-4d6c-83a4-939ecaa98572" /><br>
<img width="1230" height="302" alt="image" src="https://github.com/user-attachments/assets/4eeff2f3-611d-4480-9612-54e629d05bc2" /><br>
<img width="776" height="579" alt="image" src="https://github.com/user-attachments/assets/e776a8c3-b8aa-43dd-83c0-29d4cbc6680b" /><br>
<img width="702" height="283" alt="image" src="https://github.com/user-attachments/assets/a7bc8482-44d8-4a30-8119-04ce9223e895" /><br>
<img width="422" height="32" alt="image" src="https://github.com/user-attachments/assets/55507813-c631-4e86-b0fe-62258ef307ad" /><br>
<img width="606" height="310" alt="image" src="https://github.com/user-attachments/assets/64b62485-dee3-4d53-b939-d87cb325e853" /><br>
<img width="602" height="305" alt="image" src="https://github.com/user-attachments/assets/4ff2b390-19e8-489d-8d83-0ba8e4c4e9ff" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


