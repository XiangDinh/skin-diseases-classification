import pandas as pd

# df_1 = pd.read_csv('train_2019.csv')
# df_2 = pd.read_csv('train_2020.csv')
# print(len(df_1))
# print(len(df_2))
df = pd.concat(
    map(pd.read_csv, ['train_2019.csv','train_2020.csv']), ignore_index=True)
print(len(df))
df.to_csv('train.csv',index=True)


