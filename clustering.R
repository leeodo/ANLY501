fire = read.csv("fire_history_cleaned.csv")
df = fire[c("CalculatedAcres","DailyAcres","DiscoveryAcres","EstimatedCostToDate", "FireCause")]
data = df[c("CalculatedAcres","DailyAcres","DiscoveryAcres","EstimatedCostToDate")]
label = df["FireCause"]

#KMeans
nclusters = 2:5
wss = rep(0, 4)
for (i in nclusters) {
  km = kmeans(data, centers = i, nstart = 25)
  wss[i-1] = km$withinss
}

wss = cbind(nclusters, wss)
wss = as.data.frame(wss)

library(ggplot2)
ggplot(wss, aes(x=nclusters, y=wss)) +
  geom_line()

#using too much memory
library(factoextra)
fviz_nbclust(data, kmeans, method = "silhouette")

#Hierarchical
#using too much memory subset data
data_sub = sample(1:nrow(data), 5000)
data_sub = data[data_sub,]
set.seed(2021)
data_sc = scale(data_sub)
dist_mat = dist(data_sc)
hclust_avg = hclust(dist_mat, method = 'average')
plot(hclust_avg)

cut_avg = cutree(hclust_avg, k = 4)