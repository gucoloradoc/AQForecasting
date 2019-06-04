#Data imputation based in single station observations
#Libraries used
library(readr)
library(mice)
library(VIM)
library(data.table)

#setwd("/Users/gucoloradoc/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/Tecnológico de Monterrey/Maestría/Tesis/2019-I/AQForecasting/")

NOROESTE <- read_csv("OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/Tecnológico de Monterrey/Maestría/Tesis/2019-I/AQForecasting/NOROESTE.csv", 
                     col_types = cols(FECHA = col_datetime(format = "%Y-%m-%d %H:%M:%S")))
View(NOROESTE)

#Counting of missing data

mdtableNOROESTE<-md.pattern(NOROESTE)
mdtableNOROESTE2 <-setDT(as.data.frame(mdtableNOROESTE), keep.rownames = TRUE)[]

#imputation of data (default predictive mean matching)
imp5 <- mice(NOROESTE[,2:16], m=5, seed = 23109)
save(imp5, file="NOROESTE_method_pmm_all.RData")
NOROESTEimp<-cbind(NOROESTE[,1],complete(imp5, 5))
NORESTEimp[,1]<-format(NORESTEimp$FECHA,"%Y-%m-%d %H:%M:%S")
write_csv(NORESTEimp,'NOROESTEimp.csv')

imp5norm <- mice(NOROESTE[,2:16], m=5, seed = 23109, method = "norm")
save(imp5norm, file="NOROESTE_method_norm_all.RData")
NOROESTEimp<-cbind(NOROESTE[,1],complete(imp5norm, 5))
NOROESTEimp[,1]<-format(NOROESTEimp$FECHA,"%Y-%m-%d %H:%M:%S")
write_csv(NORESTEimp,'NOROESTEimp_norm.csv')

imp5lnorm <- mice(NOROESTE[,2:16], m=5, seed = 23109, method = "2l.norm")
save(imp5lnorm, file="NOROESTE_method_norm_all.RData")
NOROESTEimp<-cbind(NOROESTE[,1],complete(imp5, 5))
NOROESTEimp[,1]<-format(NOROESTEimp$FECHA,"%Y-%m-%d %H:%M:%S")
write_csv(NORESTEimp,'NOROESTEimp_norm.csv')

decompose(NOROESTE$PM2.5)
