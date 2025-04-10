import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df_median = pd.read_csv("C:\\MMO\\first\\Titanic\\train.csv")
df_mean = pd.read_csv("C:\\MMO\\first\\Titanic\\train_2.csv")
df_mode = pd.read_csv("C:\\MMO\\first\\Titanic\\train_3.csv")

df_median_test = pd.read_csv("C:\\MMO\\first\\Titanic\\test.csv")
df_mean_test = pd.read_csv("C:\\MMO\\first\\Titanic\\test _2.csv")
df_mode_test = pd.read_csv("C:\\MMO\\first\\Titanic\\test_3.csv")

pd.set_option('display.max_columns', None)
print(df_median.head())
print("Omissions \n" ,df_median.isna().sum())

num_cols_1 = df_median.select_dtypes(include = "number").columns
cat_cols_1 = df_median.select_dtypes(include=['object']).columns
num_cols_2 = df_mean.select_dtypes(include = "number").columns
cat_cols_2 = df_mean.select_dtypes(include=['object']).columns

df_median[num_cols_1] = df_median[num_cols_1].fillna(df_median[num_cols_1].median())
df_mean[num_cols_2] = df_mean[num_cols_2].fillna(df_mean[num_cols_2].mean())

pd.set_option('future.no_silent_downcasting', True)
df_median[cat_cols_1] = df_median[cat_cols_1].fillna(df_median[cat_cols_1].mode().iloc[0])
df_mean[cat_cols_2] = df_mean[cat_cols_2].fillna(df_mean[cat_cols_2].mode().iloc[0])

df_mode = df_mode.fillna(df_mode.mode().iloc[0])


print("First DataFrame \n ", df_median.head(18))
print("Omissions \n" ,df_median.isna().sum())
print("Second DataFrame\n", df_mean.head())
print("Omissions \n" ,df_mean.isna().sum())
print("Third DataFrame\n", df_mode.head())
print("Omissions \n" ,df_mode.isna().sum())


df_median['Deck'] = df_median['Cabin'].str[0]
df_median['RoomNumber'] = df_median['Cabin'].str.extract('(\d+)')
df_median['RoomNumber'] = df_median['RoomNumber'].astype(int)
df_median['Section'] = df_median['Cabin'].str[-1]
#df_median['Section'] = df_median['Section'].where(df_median['Section'].str.isalpha(), 'Unknown')
df_median.drop(columns=['Cabin'], inplace=True)

df_mean['Deck'] = df_mean['Cabin'].str[0]
df_mean['RoomNumber'] = df_mean['Cabin'].str.extract('(\d+)')
df_mean['RoomNumber'] = df_mean['RoomNumber'].astype(int)
df_mean['Section'] = df_mean['Cabin'].str[-1]
#df_mean['Section'] = df_mean['Section'].where(df_mean['Section'].str.isalpha(), 'Unknown')
df_mean.drop(columns=['Cabin'], inplace=True)

df_mode['Deck'] = df_mode['Cabin'].str[0]
df_mode['RoomNumber'] = df_mode['Cabin'].str.extract('(\d+)')
df_mode['RoomNumber'] = df_mode['RoomNumber'].astype(int)
df_mode['Section'] = df_mode['Cabin'].str[-1]
#df_mode['Section'] = df_mode['Section'].where(df_mode['Section'].str.isalpha(), 'Unknown')
df_mode.drop(columns=['Cabin'], inplace=True)

df1 = pd.get_dummies(df_median, columns=['HomePlanet','Destination','Deck', 'Section'], drop_first=False)
df2 = pd.get_dummies(df_mean, columns=['HomePlanet','Destination','Deck', 'Section'], drop_first=False)
df3 = pd.get_dummies(df_mode, columns=['HomePlanet','Destination','Deck', 'Section'], drop_first=False)

scaler_1 = MinMaxScaler()
scaler_2 = MinMaxScaler()
scaler_3 = MinMaxScaler()

num_cols = df1.select_dtypes(include=['number']).columns
df1[num_cols] = scaler_1.fit_transform(df1[num_cols])
num_cols = df2.select_dtypes(include=['number']).columns
df2[num_cols] = scaler_2.fit_transform(df2[num_cols])
num_cols = df3.select_dtypes(include=['number']).columns
df3[num_cols] = scaler_3.fit_transform(df3[num_cols])

pd.set_option('display.max_columns', None)
print("Answer: \n\n\n\n",df1.head())
print(df2.head())
print(df3.head())

#df1.to_excel('updated_data_1.xlsx', index=False)
#df2.to_excel('updated_data_2.xlsx', index=False)
#df3.to_excel('updated_data_3.xlsx', index=False)


#----------------------------

pd.set_option('display.max_columns', None)
print(df_mean_test.head())
print("Omissions \n" ,df_median_test.isna().sum())

num_cols_1 = df_median_test.select_dtypes(include = "number").columns
cat_cols_1 = df_median_test.select_dtypes(include=['object']).columns
num_cols_2 = df_mean_test.select_dtypes(include = "number").columns
cat_cols_2 = df_mean_test.select_dtypes(include=['object']).columns

df_median_test[num_cols_1] = df_median_test[num_cols_1].fillna(df_median_test[num_cols_1].median())
df_mean_test[num_cols_2] = df_mean_test[num_cols_2].fillna(df_mean_test[num_cols_2].mean())

pd.set_option('future.no_silent_downcasting', True)
df_median_test[cat_cols_1] = df_median_test[cat_cols_1].fillna(df_median_test[cat_cols_1].mode().iloc[0])
df_mean_test[cat_cols_2] = df_mean_test[cat_cols_2].fillna(df_mean_test[cat_cols_2].mode().iloc[0])

df_mode_test = df_mode_test.fillna(df_mode_test.mode().iloc[0])



df_median_test['Deck'] = df_median_test['Cabin'].str[0]
df_median_test['RoomNumber'] = df_median_test['Cabin'].str.extract('(\d+)')
df_median_test['RoomNumber'] = df_median_test['RoomNumber'].astype(int)
df_median_test['Section'] = df_median_test['Cabin'].str[-1]
#df_median['Section'] = df_median['Section'].where(df_median['Section'].str.isalpha(), 'Unknown')
df_median_test.drop(columns=['Cabin'], inplace=True)

df_mean_test['Deck'] = df_mean_test['Cabin'].str[0]
df_mean_test['RoomNumber'] = df_mean_test['Cabin'].str.extract('(\d+)')
df_mean_test['RoomNumber'] = df_mean_test['RoomNumber'].astype(int)
df_mean_test['Section'] = df_mean_test['Cabin'].str[-1]
#df_mean['Section'] = df_mean['Section'].where(df_mean['Section'].str.isalpha(), 'Unknown')
df_mean_test.drop(columns=['Cabin'], inplace=True)

df_mode_test['Deck'] = df_mode_test['Cabin'].str[0]
df_mode_test['RoomNumber'] = df_mode_test['Cabin'].str.extract('(\d+)')
df_mode_test['RoomNumber'] = df_mode_test['RoomNumber'].astype(int)
df_mode_test['Section'] = df_mode_test['Cabin'].str[-1]
#df_mode['Section'] = df_mode['Section'].where(df_mode['Section'].str.isalpha(), 'Unknown')
df_mode_test.drop(columns=['Cabin'], inplace=True)

df1_test = pd.get_dummies(df_median_test, columns=['HomePlanet','Destination','Deck', 'Section'], drop_first=False)
df2_test = pd.get_dummies(df_mean_test, columns=['HomePlanet','Destination','Deck', 'Section'], drop_first=False)
df3_test = pd.get_dummies(df_mode_test, columns=['HomePlanet','Destination','Deck', 'Section'], drop_first=False)


num_cols = df1_test.select_dtypes(include=['number']).columns
df1_test[num_cols] = scaler_1.transform(df1_test[num_cols])
num_cols = df2_test.select_dtypes(include=['number']).columns
df2_test[num_cols] = scaler_2.transform(df2_test[num_cols])
num_cols = df3_test.select_dtypes(include=['number']).columns
df3_test[num_cols] = scaler_3.transform(df3_test[num_cols])

pd.set_option('display.max_columns', None)
print("Answer: \n\n\n\n",df1_test.head())
print(df2_test.head())
print(df3_test.head())


df1_test.drop(columns=['Name','PassengerId'], inplace=True)
df2_test.drop(columns=['Name','PassengerId'], inplace=True)
df3_test.drop(columns=['Name','PassengerId'], inplace=True)

df1_test.to_excel('updated_test_data_1.xlsx', index=False)
df2_test.to_excel('updated_test_data_2.xlsx', index=False)
df3_test.to_excel('updated_test_data_3.xlsx', index=False)