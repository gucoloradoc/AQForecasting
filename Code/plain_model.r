
library("readxl") # to read excel files


# Data cleaning
pressure <- read_excel("../Data/Mexico/CDMX/PRESION/2018PA.xls") # Hourly lectures, imported as list

# "Plain Model": Input variables used by Ruiz-Suarez, 1995: 
#               Meteorologycal: Wind Direction, Wind Velocity, Temperature. 
#               RAMA Varaibles: CO,SO2, NO2, NOx

#REDMET data gathering, hourly data
HR2017 <- read_excel("../Data/Mexico/CDMX/REDMET/17REDMET/2017RH.xls") #Idea: To import the date as character and import that space as three columns: Use chmod 
TEMP2017<- read_excel("../Data/Mexico/CDMX/REDMET/17REDMET/2017TMP.xls")
WDR2017<- read_excel("../Data/Mexico/CDMX/REDMET/17REDMET/2017WDR.xls")
WSP2017<- read_excel("../Data/Mexico/CDMX/REDMET/17REDMET/2017WSP.xls")

#REDMET data cleaning: Data imported as List, opportunity to organize everything in one dataset
all(TEMP2017[[1]]==WDR2017[[1]]) #Testing that all the values of date (1) and hours (2) are exactly the same, useful to ammed data.
test <-month.day.year(TEMP2017[[1]][1:10])

#RAMA data gathering, hourly data
CO2017 <- read_excel("../Data/Mexico/CDMX/RAMA/17RAMA/2017CO.xls")
NO2017 <- read_excel("../Data/Mexico/CDMX/RAMA/17RAMA/2017NO.xls")
NO22017 <- read_excel("../Data/Mexico/CDMX/RAMA/17RAMA/2017NO2.xls")
NOx2017 <- read_excel("../Data/Mexico/CDMX/RAMA/17RAMA/2017NO2.xls")
SO22017 <- read_excel("../Data/Mexico/CDMX/RAMA/17RAMA/2017SO2.xls")

#################
O32017 <- read_excel("../Data/Mexico/CDMX/RAMA/17RAMA/2017O3.xls") #Expected output of the model##################

#Data cleaning
# Filtering the stations that measure the interest variables:
commonstations <- function(...){
    #Input is comma separated lists, where the names attribute is going to be intercepted.
    ldbases = list(...)
    #for each dbs, intersect the common stations.
    for (i in 1:(length(ldbases)-2)){
        if (i== 1){
            temp=intersect(names(ldbases[[1]]),names(ldbases[[2]]))
        }
        temp=intersect(temp,names(ldbases[[i+2]]))
    }
    return(temp)
}
#Function testing
#all(intersect(intersect(names(O32017),names(SO22017)),names(HR2017))==commonstations(O32017,SO22017,HR2017))
commonstations(O32017,HR2017,TEMP2017,WDR2017,WSP2017,CO2017, NO2017, NO22017, NOx2017, SO22017)