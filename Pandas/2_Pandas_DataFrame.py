import pandas as pd

print("DataFrame with list")
data=[1,2,3,4,5]
df=pd.DataFrame(data)
print(df)

print("dataframe with list")
data=[['PPA',4],['LB',3],['Python',3],['Angular',3]]
df=pd.DataFrame(data,columns=['Name','Duration'])
print(df)

data={'Name':['PPA','LB','Python','Angular'],'Duration ':[4,3,3,3]}
df=pd.DataFrame(data)
print(df)

data=[{'Name':'PPA','Duration':3,'Fess':16500},{'Name':'LB','Duration':3,'Fess':16500}
,{'Name':'Python','Duration':3,'Fess':16500}
,{'Name':'Angular','Duration':3,'Fess':16500}
]

df=pd.DataFrame(data)
print(df)
