import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import correlation as cor



# df = pd.read_csv('output.csv',index_col='stdId')

df=pd.read_csv('output.csv', usecols=['stdId','totalSessions','totalActivities','totalAnnotations'],index_col=0)
print(df)

# remove NAN
#df.dropna(inplace=True)

# Scaling the data by choosing the columns we want to scale (Scalar Transform)
print(df.describe())

scaler = StandardScaler()

df[['totalSessions_T','totalActivities_T','totalAnnotations_T']]= scaler.fit_transform(df[['totalSessions','totalActivities','totalAnnotations']])
print(df)
print((df.describe()))


# Elbow method, selecting number of clusters
def optimise_k_means(data, max_k):
    means=[]
    inertias=[]
    for k in range(1,max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
    # Generate elbow plot
    fig = plt.subplots(figsize=(10,5))
    plt.plot(means,inertias,'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Intertia')
    plt.grid(True)
    plt.show()

# Here we will have to use dimensionality reduction to convert multidimentional data to 2D, since scatter plot is 2D
#optimise_k_means(df[['totalSessions_T','totalActivities_T','totalAnnotations_T']],10)

optimise_k_means(df[['totalSessions_T','totalAnnotations_T']],10)

# Applying KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(df[['totalSessions_T','totalAnnotations_T']])

# kmeans.labels_ are the labels of clusters to which each datapoint belongs
df['kmeans_3']=kmeans.labels_
print(df)# Plot the clustering reults

plt.scatter(x=df['totalAnnotations_T'],y=df['totalSessions_T'],c=df['kmeans_3'])
plt.xlim(-2,5)
plt.ylim(-2,5)
plt.show()

'''
# Comparing different K values

for k in range(1,6):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(df[['RHOB_T','NPHI_T']])
    df[f'KMeans_{k}'] = kmeans.labels_

#df
fig, axs = plt.subplots(nrows=1,ncols=5,figsize=(20,5))

for i, ax in enumerate(fig.axes,start=1):
    ax.scatter(x=df['NPHI'],y=df['RHOB'],c=df[f'KMeans_{i}'])
    ax.set_ylim(3,1.5)
    ax.set_xlim(0,1)
    ax.set_title(f'N Clusters: {i}') 
    
cor.correlation_matrix(df[['RHOB_T','NPHI_T','GR_T','PEF_T','DTC_T']])

cor.correlation_matrix(df[['RHOB','NPHI','GR','PEF','DTC']])'''