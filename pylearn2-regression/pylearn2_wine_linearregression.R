wine=read.csv('winequality-red.csv',sep=";",header=T)
wine_train=wine[1:(1024+256),]
wine_test=wine[1281:1599,]
fit_train <- lm(quality ~ ., data=wine_train)
summary(fit_train)
mean(fit_train$residuals^2)

newdata=wine_test[,1:11]
q=predict(fit_train, newdata, interval="predict") 
mean((wine_test[12]-q[,1])^2)

#on_train: 0.43
#on_test:0.41

wine_test=wine[1281:1599,1:11]
write.csv(wine_test,'wine_test.csv',row.names=F)

wine_result=read.csv("output.csv",header=F)
