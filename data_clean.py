#import packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#creating corpus
fire_news = pd.read_csv("fire_news.csv")

climate_news = pd.read_csv("climate_news.csv")

fire_corpus = fire_news["description"].tolist()

climate_corpus = climate_news["description"].to_list()

#vectorizing the corpuses

##initiate the CountVectorizers
fireCV = CountVectorizer()
climateCV = CountVectorizer()

##fit matrices
fireMatrix = fireCV.fit_transform(fire_corpus)
climateMatrix = climateCV.fit_transform(climate_corpus)

#transform to DataFrames
fireColNames = fireCV.get_feature_names()
fireDF = pd.DataFrame(fireMatrix.toarray(), columns=fireColNames)

climateColNames = climateCV.get_feature_names()
climateDF = pd.DataFrame(climateMatrix.toarray(), columns=climateColNames)

#store on the disk
fireDF.to_csv("fire_news_DF.csv", index=False)
climateDF.to_csv("climate_news_DF.csv", index=False)