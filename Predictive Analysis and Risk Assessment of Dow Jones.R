library(readr)
library(lubridate)
library(quantmod)
library(tseries)
library(dynlm)
library(dplyr)
library(caret)
library(tree)
library(e1071)
library(tseries)
library(quantmod)
library(readxl)
library(tidyverse)


#Creation of the dataframe, it contains information for 30 companies for 1st and 2nd quarter of 2011
dow_jones_index <- read.csv("F:/Master's Courses UTSA/Spring 2019/Data Applications/Datasets/dow_jones_index.data")

#Creation of a duplicate to alter and to test on
dowjones <- dow_jones_index

#Omit NAs in the data set, only 30 observations were removed (750->730)
dowjones <- na.omit(dowjones)

#Change open, high, low, close, volume, next_weeks_open, new_weeks_close to usable numbers
dowjones$open = as.numeric(gsub("\\$", "", dowjones$open))
dowjones$high = as.numeric(gsub("\\$", "", dowjones$high))
dowjones$low = as.numeric(gsub("\\$", "", dowjones$low))
dowjones$close = as.numeric(gsub("\\$", "", dowjones$close))
dowjones$volume = as.numeric(dowjones$volume)
dowjones$next_weeks_open = as.numeric(gsub("\\$", "", dowjones$next_weeks_open))
dowjones$next_weeks_close = as.numeric(gsub("\\$", "", dowjones$next_weeks_close))

#Change date to date class
dowjones$date <- as.Date(dowjones$date, format = "%m/%d/%Y")

#change quarter to a factor variable
dowjones$quarter <- as.factor(dowjones$quarter)


#Looking at log plots of variables to see correlation, these were the only ones ones that had correlation 
#based on the lag plot function
lag.plot(dowjones$open, lag = 5) #open
lag.plot(dowjones$high, lag = 5) #high
lag.plot(dowjones$low, lag = 5) #low
lag.plot(dowjones$close, lag = 5) #close
lag.plot(dowjones$next_weeks_open, lag = 5) #next_weeks_open
lag.plot(dowjones$next_weeks_close, lag = 5) #next_weeks_close
lag.plot(dowjones$percent_return_next_dividend, lag = 5) #percent_return_next_dividend

#Creation of new data frame to test on
dowjonesnew <- dowjones

#Make the stock variable a factor
dowjonesnew$stock <- as.factor(dowjonesnew$stock)

#Due to nature of data set wherein each company was represented and stacked on one another. I needed to create 
#lag variables in certain way so that the last observation of a company did not become the first observation
#for the next stacked company.  Needed to do this for the seven variables that we found using lag plots
#Creation of lags for variables
dowjonesnew <- 
  dowjonesnew %>%
  group_by(stock) %>%
  mutate(lag.open = dplyr::lag(open, n = 1, default = NA)) %>% 
  ungroup()

dowjonesnew <- 
  dowjonesnew %>%
  group_by(stock) %>%
  mutate(lag.high = dplyr::lag(high, n = 1, default = NA)) %>% 
  ungroup()

dowjonesnew <- 
  dowjonesnew %>%
  group_by(stock) %>%
  mutate(lag.low = dplyr::lag(low, n = 1, default = NA)) %>% 
  ungroup()

dowjonesnew <- 
  dowjonesnew %>%
  group_by(stock) %>%
  mutate(lag.close = dplyr::lag(close, n = 1, default = NA)) %>% 
  ungroup()

dowjonesnew <- 
  dowjonesnew %>%
  group_by(stock) %>%
  mutate(lag.next_weeks_open = dplyr::lag(next_weeks_open, n = 1, default = NA)) %>% 
  ungroup()

dowjonesnew <- 
  dowjonesnew %>%
  group_by(stock) %>%
  mutate(lag.next_weeks_close = dplyr::lag(next_weeks_close, n = 1, default = NA)) %>% 
  ungroup()

dowjonesnew <- 
  dowjonesnew %>%
  group_by(stock) %>%
  mutate(lag.percent_return_next_dividend = dplyr::lag(percent_return_next_dividend, 
                                                       n = 1, default = NA)) %>% 
  ungroup()

#Making a training set from 1st quarter data of a company and testing set from the 2nd quarter
dowjonesnew.train <- subset(dowjonesnew, dowjonesnew$quarter == 1) #330 observations
dowjonesnew.test <- subset(dowjonesnew, dowjonesnew$quarter == 2) #390 observations

#Splitting both the train and test set based on company, now its a list of 30 data frames (one for each
#company).
dowjonesnew.train <- split(dowjonesnew.train, dowjonesnew.train$stock)
dowjonesnew.test <- split(dowjonesnew.test, dowjonesnew.test$stock)


#General formula for a linear function, this one is based on the stock AA
#These variables were used because they were significant and created a low AIC value.
#May want to test what variables are significant and have a low AIC.
linear.model.AA <- lm(formula = percent_change_price ~ lag.open + lag.close+ lag.high + lag.low +
                        lag.percent_return_next_dividend, data = dowjonesnew.train$AA)
summary(linear.model.AA)

#Using the earlier linear model as an example made a function to speed up process of creation of linear 
#model and obtaining an RMSE(Root Mean Square Error) value .  Remember a lower RMSE value indicates a 
#better fit.
#Creation of a function that will take a specified company's train and test set then print a RMSE value
#Example of trainstock = dowjonesnew.train$AA or <train data>$<stock symbol>
#Example of teststock = dowjonesnew.test$AA <test data>$<stock symbol>
glmfxn <- function (trainstock, teststock) {
  set.seed(123)
  glmfit <- glm(formula = percent_change_price ~ lag.open + lag.close+ lag.high + lag.low +
                  lag.percent_return_next_dividend, 
                data = trainstock)
  
  linear.predict <- predict.glm(glmfit, newdata = teststock) 
  err <- mean((linear.predict-teststock$percent_change_price)^2)
  print(err)
}

#Make an empty vector for the linear errors
linerr <- NULL

#Loop using glm fxn to make a model, predict, and get a RMSE for all the stocks and put in empty vector
for (i in levels(dowjonesnew$stock)){
  x=dowjonesnew.train[[i]]
  y=dowjonesnew.test[[i]]
  linerr <- c(linerr, glmfxn(x,y))
  
#Uncomment out the next 4 lines if you want to save each value in seperate text files for each company
  # namefilmodel = paste0("Linear_Accuracy", i, ".txt")
  # sink(namefilmodel)
  # print(glmfxn(x,y))
  # sink()
}

#This will display the average RMSE value for the linear models 
avg.linerr <- mean(linerr); avg.linerr



#Creation of the Tree function to streamline modeling and obtaining RMSE values, will print out RMSE value
#Example of trainstock = dowjonesnew.train$AA or <train data>$<stock symbol>
#Example of teststock = dowjonesnew.test$AA <test data>$<stock symbol>
treefxn <- function (trainstock, teststock) {
  set.seed(123)
  suppressWarnings(treemodel <- tree(percent_change_price ~ ., data = trainstock))
  suppressWarnings(tree.predict <- predict(treemodel, 
                                           newdata = teststock))
  err <- mean((tree.predict-teststock$percent_change_price)^2)
  print(err)
}

#Make an empty vector for the tree errors
treeerr <- NULL

#Loop using tree fxn to make a model, predict and get a RMSE value for all the stocks and put in empty vector
for (i in levels(dowjonesnew$stock)){
  x=dowjonesnew.train[[i]]
  y=dowjonesnew.test[[i]]
  treeerr <- c(treeerr,treefxn(x,y))
  
#Uncomment out the next 4 lines if you want to save each value in seperate text files for each company
  # namefilmodel = paste0("Tree_Accuracy", i, ".txt")
  # sink(namefilmodel)
  # print(glmfxn(x,y))
  # sink()
}

#Get average RMSE for the tree models
avg.treeerr <- mean(treeerr); avg.treeerr




#Creation of SVR Model

#Creation of new data set to scale certain variables, can only scale numeric variables
dowjonesnewscale <- dowjonesnew

#Scaling variables 
dowjonesnewscale[,4:23] <- scale(dowjonesnew[,4:23])

#Splitting into Train and Test again based on quarter
dowjonesnewscale.train <- subset(dowjonesnewscale, dowjonesnewscale$quarter == 1) 
dowjonesnewscale.test <- subset(dowjonesnewscale, dowjonesnewscale$quarter == 2) 

#Splitting the data further by stock so that each stock is its own data frame
dowjonesnewscale.train <- split(dowjonesnewscale.train, dowjonesnewscale.train$stock)
dowjonesnewscale.test <- split(dowjonesnewscale.test, dowjonesnewscale.test$stock)

#Creation of SVR model function, prints a RMSE value, need to use new scaled data sets 
#Example of trainstock = dowjonesnewscale.train$AA or <scaled train data>$<company symbol>
#Example of teststock = dowjonesnewscale.test$AA or <scaled test data>$<company symbol>
svrfxn <- function (trainstock, teststock) {
  set.seed(123)
  svrmodel <- svm(percent_change_price ~ lag.open + lag.close+ lag.high + lag.low +
                    lag.percent_return_next_dividend, data = trainstock)
  svr.predict <- predict(svrmodel, 
                         newdata = teststock)
  err <- mean((svr.predict-teststock$percent_change_price)^2)
  print(err)
}

#Make an empty vector for the svr errors
svrerr <- NULL

#Loop using svr fxn to make a model, predict and get a RMSE value for all the stocks and put in empty vector
for (i in levels(dowjonesnew$stock)){
  x=dowjonesnewscale.train[[i]]
  y=dowjonesnewscale.test[[i]]
  svrerr <- c(svrerr,svrfxn(x,y))

#Uncomment out the next 4 lines if you want to save each value in seperate text files for each company  
  # namefilmodel = paste0("SVR_Accuracy_", i, ".txt")
  # sink(namefilmodel)
  # print(svrfxn(x,y))
  # sink()
}

#Get average RMSE for the SVR models 
avg.svr <- mean(svrerr); avg.svr


#Pull in data for the S&P 500 from yahoo finance
SP500 <- read.csv("SP500.csv", header = TRUE, sep = ",") 

#Split the orginal data set by stock
dowjonesreturn <- split(dowjones,dowjones$stock)

#Create returns for each of the 30 companies and the S&P 500 based off closing price
ReturnSP500 <-  na.omit(Delt(SP500[,5]))
ReturnAA <- na.omit(Delt(dowjonesreturn$AA[,7]))
ReturnAXP <- na.omit(Delt(dowjonesreturn$AXP[,7]))
ReturnBA <- na.omit(Delt(dowjonesreturn$BA[,7]))
ReturnBAC <- na.omit(Delt(dowjonesreturn$BAC[,7]))
ReturnCAT <- na.omit(Delt(dowjonesreturn$CAT[,7]))
ReturnCSCO <- na.omit(Delt(dowjonesreturn$CSCO[,7]))
ReturnCVX <- na.omit(Delt(dowjonesreturn$CVX[,7]))
ReturnDD <- na.omit(Delt(dowjonesreturn$DD[,7]))
ReturnDIS <- na.omit(Delt(dowjonesreturn$DIS[,7]))
ReturnGE <- na.omit(Delt(dowjonesreturn$GE[,7]))
ReturnHD <- na.omit(Delt(dowjonesreturn$HD[,7]))
ReturnHPQ <- na.omit(Delt(dowjonesreturn$HPQ[,7]))
ReturnIBM <- na.omit(Delt(dowjonesreturn$IBM[,7]))
ReturnINTC <- na.omit(Delt(dowjonesreturn$INTC[,7]))
ReturnJNJ <- na.omit(Delt(dowjonesreturn$JNJ[,7]))
ReturnJPM <- na.omit(Delt(dowjonesreturn$JPM[,7]))
ReturnKO <- na.omit(Delt(dowjonesreturn$KO[,7]))
ReturnKRFT <- na.omit(Delt(dowjonesreturn$KRFT[,7]))
ReturnMCD <- na.omit(Delt(dowjonesreturn$MCD[,7]))
ReturnMMM <- na.omit(Delt(dowjonesreturn$MMM[,7]))
ReturnMRK <- na.omit(Delt(dowjonesreturn$MRK[,7]))
ReturnMSFT <- na.omit(Delt(dowjonesreturn$MSFT[,7]))
ReturnPFE <- na.omit(Delt(dowjonesreturn$PFE[,7]))
ReturnPG <- na.omit(Delt(dowjonesreturn$PG[,7]))
ReturnT <- na.omit(Delt(dowjonesreturn$T[,7]))
ReturnTRV <- na.omit(Delt(dowjonesreturn$TRV[,7]))
ReturnUTX <- na.omit(Delt(dowjonesreturn$UTX[,7]))
ReturnVZ <- na.omit(Delt(dowjonesreturn$VZ[,7]))
ReturnWMT <- na.omit(Delt(dowjonesreturn$WMT[,7]))
ReturnXOM <- na.omit(Delt(dowjonesreturn$XOM[,7]))

#Combine all the Return Values into a data frame
MyData <- cbind(ReturnSP500, ReturnAA, ReturnAXP, ReturnBA, ReturnBAC, ReturnCAT, ReturnCSCO, 
                ReturnCVX, ReturnDD, ReturnDIS, ReturnGE, ReturnHD, ReturnHPQ, ReturnIBM, ReturnINTC, 
                ReturnJNJ, ReturnJPM, ReturnKO, ReturnKRFT, ReturnMCD, ReturnMMM, ReturnMRK, ReturnMSFT, 
                ReturnPFE, ReturnPG, ReturnT, ReturnTRV, ReturnUTX, ReturnVZ, ReturnWMT, ReturnXOM)

#Give the data frame column names
colnames(MyData) <- c("SP500","AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS",
                      "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "KRFT", "MCD",
                      "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UTX", "VZ", "WMT",
                      "XOM")

#Visualization of the expected returns of each company against the market (S&P 500)
boxplot(MyData, main = "Expected Return", xlab = "Stock Picks", ylab = "Return")

# Created a copy of data table that was just made then changed it to a data frame
MyStocks <- MyData
MyStocks <- as.data.frame(MyStocks)

#Creation of the beta function (risk assessment), will print beta value
#example of stock = MyStocks$AA or MyStocks$<Stock Symbol>
betafxn <- function(stock){
  
  lmmodel<- lm(formula = stock ~ SP500, data = MyStocks)
  
  Beta <- summary(lmmodel)$coefficients[2, 1]
  print(Beta)
}

#Create list of stocks for the loop
returns <- colnames(MyStocks[,2:31])

#Loop using beta fxn to make a model and get a beta value for each company
for (i in returns){
  
  x=MyStocks[[i]]
  
  betafxn(x)
#Uncomment out the next 4 lines if you want to save each value in seperate text files for each company   
  # namefilmodel = paste0("Beta_Value_", i, ".txt")
  # sink(namefilmodel)
  # print(betafxn(x))
  # sink()
}

#Using the RMSE Values for each model and Beta Values put them in a csv file
#Made this file into a data frame and gave it row names based on company
Values <- read_excel("F:/Master's Courses UTSA/Spring 2019/Data Applications/Data Applications/Case Study 3 accuracy scores/Values.xlsx") 
Values <- as.data.frame(Values)

rownames(Values) <- c("AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS",
                      "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "KRFT", "MCD",
                      "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UTX", "VZ", "WMT",
                      "XOM")

#Looked at summary statistics of each column of values
summary(linerr)
summary(treeerr)
summary(svrerr)
summary(Values$`Beta Values`)

#Looked at variance of each column of values
var(linerr)
var(treeerr)
var(svrerr)
var(Values$`Beta Values`)

#Looked at standard deviation of each column of values
sd(linerr)
sd(treeerr)
sd(svrerr)
var(Values$`Beta Values`)


#Creation of function that would print out last iteration of predicted percent change in price for
#each company using the best model type.  Based on earlier averages of RMSE the SVR was the best type of model
#Example of trainstock = dowjonesnewscale.train$AA or <scaled train data>$<company symbol>
#Example of teststock = dowjonesnewscale.test$AA or <scaled test data>$<company symbol>
predictsvrfxn <- function (trainstock, teststock) {
  set.seed(123)
  
  svrmodel <- svm(percent_change_price ~ lag.open + lag.close+ lag.high + lag.low +
                    lag.percent_return_next_dividend, data = trainstock)
  
  svr.predict <- predict(svrmodel, 
                         newdata = teststock)
  
  svr.predict <- as.data.frame(svr.predict)
  
  print(svr.predict[13,])
  
}

#Make an empty vector to hold all the predicted percent change in prices of each company
svrpredict <- NULL

#Loop using predicted SVR function to provide a predicted value for each stock
for (i in levels(dowjonesnew$stock)){
  x=dowjonesnewscale.train[[i]]
  y=dowjonesnewscale.test[[i]]
  svrpredict <- c(svrpredict, predictsvrfxn(x,y))
  
#Uncomment out the next 4 lines if you want to save each value in seperate text files for each company
  # namefilmodel = paste0("Predicted_SVR", i, ".txt")
  # sink(namefilmodel)
  # print(glmfxn(x,y))
  # sink()
}

#Made the vector housing all the predicted values into a data frame
svrpredict <- as.data.frame(svrpredict)

#Gave newly created data frame row names corresponding to their respective company
rownames(svrpredict) <- c("AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS",
                          "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "KRFT", "MCD",
                          "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UTX", "VZ", "WMT",
                          "XOM")

#It was found that the top 5 companies with the highest predicted percent changed in price 
#were DD, CVX, CAT, XOM, and GE

#Binded the predicted values of each company with their respective beta values in a new data frame
ReturnBeta <- cbind.data.frame(Values$`Beta Values`, svrpredict)

#Gave the new data frame row names based on company
rownames(ReturnBeta) <- c("AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS",
                          "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "KRFT", "MCD",
                          "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UTX", "VZ", "WMT",
                          "XOM")
#Gave the data frame column names to differentiate the values
colnames(ReturnBeta) <- c("Betas", "Predictions")

#Visual representation of the Beta Values vs the Predicted Values with each dot labeled by company stock symbol
ggplot(ReturnBeta, aes(x= ReturnBeta$Betas, y= ReturnBeta$Predictions, 
                       label=rownames(ReturnBeta)))+
  geom_point() +
  geom_text(aes(label=rownames(ReturnBeta)),hjust=1, vjust=1, color = "red") +
  xlab("Betas") + ylab("Returns") +
  theme_test()