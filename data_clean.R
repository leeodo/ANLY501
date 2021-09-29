#World Bank Climate Change Data API
wb39 = read.csv("mavg_USA_39.csv", header = TRUE)
wb59 = read.csv("mavg_USA_59.csv", header = TRUE)
wb79 = read.csv("mavg_USA_79.csv", header = TRUE)
wb99 = read.csv("mavg_USA_99.csv", header = TRUE)

wb_weather = rbind(wb39, wb59, wb79, wb99)

library(tidyverse)
wb_weather = wb_weather %>%
  mutate(annual_avg = rowMeans(wb_weather[,5:16]))

#write cleaned data back to disk
write.csv(wb_weather, "wb_weather_cleaned.csv")


#NFIC
fire_hist = read.csv("WFIGS_-_Wildland_Fire_Locations_Full_History.csv")
colnames(fire_hist)[1] = "X"

#found one column don't have any values
which(colSums(is.na(fire_hist)) == length(fire_hist$X))

#removing the empty column
fire_hist = fire_hist[,-13]

#removing useless columns for the purpose of this project
keeps = c("X", "Y", "CalculatedAcres", "ContainmentDateTime", "ControlDateTime",
          "DailyAcres", "DiscoveryAcres", "EstimatedCostToDate", "FireCause",
          "FireCauseGeneral", "FireCauseSpecific", "FireDiscoveryDateTime",
          "POOFips", "POOCounty", "POOState", "TotalIncidentPersonnel",
          "IncidentTypeCategory", "IncidentTypeKind", "POOLandownerKind")
fire_hist = fire_hist[, (names(fire_hist) %in% keeps)]

#Fill empty CalculatedAcres with 0s
length(which(is.na(fire_hist$CalculatedAcres)))
fire_hist$CalculatedAcres[which(is.na(fire_hist$CalculatedAcres))] = 0

#Fill empty FireCause with Unknown
unique(fire_hist$FireCause)
length(which(fire_hist$FireCause == ""))
fire_hist$FireCause[which(fire_hist$FireCause == "")] = "Unknown"

#Fill empty EstimatedCostToDate with 0s
length(which(is.na(fire_hist$EstimatedCostToDate)))
fire_hist$EstimatedCostToDate[which(is.na(fire_hist$EstimatedCostToDate))] = 0

#write cleaned data back to disk
write.csv(fire_hist, "fire_history_cleaned.csv")


#meteostat
LAweather = read.csv("LAweather.csv")
SACweather = read.csv("SACweather.csv")

length(which(is.na(LAweather$tsun)))
length(which(is.na(SACweather$tsun)))
#remove tsun for both data
LAweather = LAweather[,1:10]
SACweather = SACweather[,1:10]

write.csv(LAweather, "LAweather_cleaned.csv")
write.csv(SACweather, "SACweather_cleaned.csv")
