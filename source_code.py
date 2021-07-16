import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

creditcard_df=pd.read_csv('/content/creditcard dataset.csv')
creditcard_df
sns.heatmap(creditcard_df.isnull(),yticklabels=False,cbar=False,cmap='Blues')
creditcard_df.isnull().sum()
creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()
creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = creditcard_df['CREDIT_LIMIT'].mean()

creditcard_df.duplicated().sum()
creditcard_df.drop('CUST_ID',axis=1,inplace=True)

plt.figure(figsize=(10,50))
plt.figure(figsize=(10,50))
for i in range(len(creditcard_df.columns)):
  plt.subplot(17,1,i+1)
  sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws={"color":'b',"lw":3,"label":"KDE"},hist_kws={"color":"g"})
  plt.title(creditcard_df.columns[i])

  plt.tight_layout()
  
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)
creditcard_df_scaled

kmeans=KMeans(7)
kmeans.fit(creditcard_df_scaled)
labels=kmeans.labels_
cluster_cemters=pd.DataFrame(data=kmeans.cluster_centers_,columns=[creditcard_df.columns])
cluster_cemters
cluster_cemters=scaler.inverse_transform(cluster_cemters)
cluster_cemters=pd.DataFrame(data=kmeans.cluster_centers_,columns=[creditcard_df.columns])
cluster_cemters

y_kmeans=kmeans.fit_predict(creditcard_df_scaled)
y_kmeans
creditcard_df_cluster=pd.concat([creditcard_df,pd.DataFrame({'cluster':labels})],axis=1)

for i in creditcard_df.columns:
 plt.figure(figsize= (35,5))
for j in range(7):
  plt.subplot(1,7,j+1)
  cluster=creditcard_df_cluster[creditcard_df_cluster['cluster'] == j] 
  cluster[i].hist(bins=20)
  plt.title('{}  \nCluster {} '.format(i,j))

  
  plt.show()
  
pca=PCA(n_components=2)
p_comp=pca.fit_transform(creditcard_df_scaled)
p_comp
pca_df=pd.DataFrame(data=p_comp,columns=['pca1','pca2'])
pca_df=pd.concat([pca_df,pd.DataFrame({'cluster':labels})],axis=1)
plt.figure(figsize=(10,10))
ax=sns.scatterplot(x='pca1',y='pca2',hue="cluster",data=pca_df,palette=['red','green','blue','pink','yellow','gray','purple'])
