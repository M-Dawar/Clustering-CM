import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from sklearn.preprocessing import StandardScaler, normalize

df=pd.read_csv('testData.csv',index_col='DEPTH_MD')
print(df.head())

df.dropna(inplace=True)

print(df.head())

data_scaled=normalize(df)


scaler = StandardScaler()

df[['RHOB_T','NPHI_T','GR_T','PEF_T','DTC_T']]= scaler.fit_transform(df[['RHOB','NPHI','GR','PEF','DTC']])

plt.figure(figsize=(10,7))
plt.title("Dendrograms Standard Scaler")
dendD = shc.dendrogram(shc.linkage(df[['RHOB_T','NPHI_T','GR_T','PEF_T','DTC_T']],method='ward'))
plt.show()

plt.figure(figsize=(10,7))
plt.title("Dendrograms Normalized Data")
dendS = shc.dendrogram(shc.linkage(data_scaled,method='ward'))
plt.show()


