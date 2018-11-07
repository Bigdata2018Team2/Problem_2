
setwd("/home/kostrian/BigData-2018-Problem2/data")
data1<-read.csv("2015-01-28.csv")

data = data1
data = data[-9]
data = data[-1]

data_fund = data[-27]
data_fund = data_fund[-27]
data_fund = data_fund[-27]

#데이터 전처리
#이상치 제거
data_fund = na.omit(data_fund)

col=0
#col = menu(colnames(data_fund), title = "라벨이 될 컬럼 이름을 입력하세요 ")
#lap = readline(prompt = "laplace값을 입력하세요 ")

data_fund_exam = data_fund

set.seed(6)

idx = sample(2,nrow(data_fund_exam),replace=TRUE,prob=c(0.7,0.3))
train_f = data_fund_exam[idx==1,]
test_f = data_fund_exam[idx==2,]

#library(e1071)

install.packages("aod")
library(aod)
install.packages("ggplot2")
library(ggplot2)

model2 = glm(fund~.,data = data_fund_exam)
summary(model2)
