import pandas as pd

df=pd.DataFrame({'Data':[11,21,51,101,111,121]})

writer=pd.ExcelWriter('MarvellousPandas.xlsx',engine='xlsxwriter')

df.to_excel(writer,sheet_name='sheet1')

#save that file in the secondary storage
writer.save()