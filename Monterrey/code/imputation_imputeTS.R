library(readr)
library(imputeTS)

stacked.imputation <- function(tseries) {
  tseriesimp <- na.seadec(tseries,  algorithm = "interpolation")
  if (length(tseriesimp[tseriesimp <0])>0) {
    tseriesimp[tseriesimp <0] <-NA
    tseriesimp <- na.seadec(tseriesimp, algorithm = "mean")
  }
  return(tseriesimp)
}

Gerimputation <- function(datapath,station) {
  #Fuction to read the csv file
  df.station <-read_csv(paste(datapath, station,".csv",sep=""), 
    col_types = cols(FECHA = col_datetime(format = "%Y-%m-%d %H:%M:%S")))
  
  df.tsstation<-ts(df.station[,2:16], start = c(2012,0), frequency = 365.25*24)
  df.tsstation_imputed=df.tsstation #initialization
  for (i in 1:(dim(df.tsstation)[2])){
    df.tsstation_imputed[,i] <- stacked.imputation(df.tsstation[,i])
  }

  #for each variable perform the imputation and save in a csv and plot 
  #in a folder imputedG with subfolders data and plots.
  return(df.tsstation_imputed)
}
#Working directory on AQPForecasting
NOROESTE <- read_csv("Monterrey/data/cleaned_station/NOROESTE.csv",
    col_types = cols(FECHA = col_datetime(format = "%Y-%m-%d %H:%M:%S")))
View(NOROESTE)

tsNOROESTE<-ts(NOROESTE, start = c(2012,0), frequency = 365.25*24)

imputed <-Gerimputation("Monterrey/data/cleaned_station/","NOROESTE")
save(imputed, file="Monterrey/data/imputed/data/NOROESTE.RData")
write.csv(cbind(NOROESTE[,1],imputed), file="Monterrey/data/imputed/data/NOROESTE.csv", row.names = FALSE)
plotNA.imputations(tsNOROESTE[,'TOUT'],imputed[,"TOUT"])


#Seasonalidad, diaria no capturada
imputeddecomposed <-decompose(imputed[,'PM2.5'])
plot(imputeddecomposed)

