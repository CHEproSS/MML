import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C:\\MMO\\first\\Titanic\\train.csv")

pd.set_option('display.max_columns', None)
print(df.head())
print(df.isna().sum())

num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.median()))

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode_value = df[col].mode()[0]
    df.loc[df[col].isna(), col] = mode_value

df['Cabin'] = df['Cabin'].fillna('Unknown')

df['Deck'] = df['Cabin'].str[0]

df['RoomNumber'] = df['Cabin'].str.extract('(\d+)')
df['RoomNumber'] = df['RoomNumber'].fillna(0).astype(int)

df['Section'] = df['Cabin'].str[-1]
df['Section'] = df['Section'].where(df['Section'].str.isalpha(), 'Unknown')

df.drop(columns=['Cabin'], inplace=True)

df = pd.get_dummies(df, columns=['Deck', 'Section'], drop_first=True)

scaler = MinMaxScaler()
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = scaler.fit_transform(df[num_cols])

pd.set_option('display.max_columns', None)
print(df.head())
