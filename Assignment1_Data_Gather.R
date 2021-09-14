#World Bank Climate API
library(httr)

response_pr = GET("http://climatedataapi.worldbank.org/climateweb/rest/v1/country/mavg/tas/1980/1999/USA.csv")

tas = content(response_pr)
write(tas, "mavg_USA.csv")

#Meteostat API

#curl "https://bulk.meteostat.net/v2/hourly/full/10637.csv.gz" --output "10637.csv.gz"

