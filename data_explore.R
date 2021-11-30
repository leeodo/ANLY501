#load all the data
wb_weather = read.csv("wb_weather_cleaned.csv")

fire_hist = read.csv("fire_history_cleaned.csv")

LAweather = read.csv("LAweather_cleaned.csv")

SACweather = read.csv("SACweather_cleaned.csv")

library(tidyverse)

#world bank climate
wb_weather %>%
  filter(GCM == "bccr_bcm2_0") %>%
  select(from_year, annual_avg) %>%
  ggplot(aes(x = from_year, y = annual_avg)) +
  geom_line() +
  xlab("Year") +
  ylab("Annual Average bccr bcm2 0")


#NFIC fire history
fire_hist %>%
  select(CalculatedAcres) %>%
  filter(CalculatedAcres > 10000) %>%
  ggplot(aes(x = CalculatedAcres)) +
  geom_histogram(bins = 50, fill = "#00AFBB") +
  xlab("Calculated Acres") +
  ggtitle("Histogram of Calculated Damaged Acres")


fire_hist %>%
  select(FireCause) %>%
  ggplot(aes(x = as.factor(FireCause))) +
  geom_bar(fill = "#FFDB6D") +
  xlab("Fire Cause") +
  ggtitle("Bar Plot of Fire Cause")

fire_hist %>%
  select(IncidentTypeCategory) %>%
  ggplot(aes(x = as.factor(IncidentTypeCategory))) +
  geom_bar(fill = "#D16103") +
  xlab("Incident Type Category") +
  ggtitle("Bar Plot of Incident Type Category")

ggplot(fire_hist, aes(x = as.factor(IncidentTypeKind))) +
  geom_bar(fill = "#C3D7A4") +
  xlab("Incident Type Kind") +
  ggtitle("Bar Plot of Incident Type Kind")

ggplot(fire_hist, aes(y = as.factor(POOState))) +
  geom_bar(fill = "#4E84C4") +
  ylab("States") +
  ggtitle("Histgram of States")


#meteostat
LAweather$time = as.Date(LAweather$time)
SACweather$time = as.Date(SACweather$time)

ggplot(LAweather, aes(x = time, y = tavg)) +
  geom_line() +
  ggtitle("LA Average Temperature")

ggplot(LAweather, aes(x = time, y = tmin)) +
  geom_line() +
  ggtitle("LA Minimum Temperature")

ggplot(LAweather, aes(x = time, y = tmax)) +
  geom_line() +
  ggtitle("LA Maximum Temperature")

ggplot(LAweather, aes(x = time, y = prcp)) +
  geom_line() +
  ggtitle("LA Precipitation")

ggplot(SACweather, aes(x = time, y = tavg)) +
  geom_line() +
  ggtitle("Sacramento Average Temperature")

ggplot(SACweather, aes(x = time, y = tmin)) +
  geom_line() +
  ggtitle("Sacramento Minimum Temperature")

ggplot(SACweather, aes(x = time, y = tmax)) +
  geom_line() +
  ggtitle("Sacramento Maximum Temperature")

ggplot(SACweather, aes(x = time, y = prcp)) +
  geom_line() +
  ggtitle("Sacramento Precipitation")
