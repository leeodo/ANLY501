# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 00:04:49 2021

@author: Yannian Liu
"""

#%%
#import packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

#%%
#reading in the text data
fire_news = pd.read_csv("fire_news.csv")
rain_news = pd.read_csv("rain_news.csv")
snow_news = pd.read_csv("snow_news.csv")

textDF = pd.concat([fire_news, rain_news, snow_news], axis = 0)
Description = textDF["title"].tolist()
labels = ['fire'] * 20 + ['rain'] * 20 + ['snow'] * 20


#%%
#vectorizing text
CV = CountVectorizer(stop_words='english', max_features=100)
DTM = CV.fit_transform(Description)
colnames = CV.get_feature_names()
textDF = pd.DataFrame(DTM.toarray(), columns=colnames)
textMX = textDF.to_numpy()
textMX_norm = normalize(textMX)


#%%
#kmeans
n_clusters = range(2, 6)
intertias = []
silht_score  = []
for i in n_clusters:
    KM = KMeans(n_clusters = i)
    KM.fit(textMX_norm)
    intertias.append(KM.inertia_)
    silht_score.append(silhouette_score(textMX_norm, KM.labels_))
    
#%%
plt.plot(n_clusters, intertias)
plt.ylabel('Inertia')
plt.xlabel('Clusters')
plt.title('Elbow Method')
plt.savefig("news_elbow.png")
plt.show()

plt.plot(n_clusters, silht_score)
plt.ylabel('Silhoutte Score')
plt.xlabel('Clusters')
plt.title('Silhoutte Method')
plt.savefig("news_silht.png")
plt.show()

#%%
#hierarchical clustering
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(textMX_norm, method='ward'))
plt.savefig("news_dendro.png")
plt.show()

#%%
#DBSCAN
pca = PCA(n_components=4)
trans = pca.fit_transform(textMX_norm)
trans = pd.DataFrame(trans, columns = ['comp1', 'comp2', 'comp3', 'comp4'])
fig = px.scatter_3d(trans, x='comp1',y='comp2',z='comp3', color='comp4')
fig.show()
fig.write_html('DBpoints.html')

#%%
db = DBSCAN(0.4)
db.fit(trans)
labels = list(map(str,db.labels_))
fig = px.scatter_3d(trans, x='comp1',y='comp2',z='comp3', color=labels)
fig.show()
fig.write_html('DBcluster.html')
