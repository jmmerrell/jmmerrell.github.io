library(GGally)
library(gridExtra)
library(ggplot2)
library(nlme)
setwd("C:/Users/merre/Desktop/jmmerrell.github.io/solar_AR1")
solar <- read.table("SolarSavings.csv",header = T,sep = ",")
solar$month <- as.factor(gsub("/","",substr(solar$Date,1,2)))
solar$Date <- as.Date(as.character(solar$Date),"%m/%d/%y")
solar[order(solar$Date),]
solar$winter <- ifelse(solar$month %in% c("12","1","2","3"),1,0)
solar$summer <- ifelse(solar$month %in% c("7","8","9","10"),1,0)
temps <- c(32,31,36,44.5,52,61,69.5,77.5,76,66.5,53.5,41.5)
temps <- abs(temps-65)
solar$temp<-c(rep(temps,4),temps[1:3])
df <- data.frame(cbind(solar$Date,fit.ar1$fitted))

pdf("solar_data.pdf")
 ggplot(data=solar, aes(x=Date, y=PowerBill, colour=Solar)) + 
   geom_line()+
   geom_point()
dev.off()

pdf("solar_fit.pdf")
ggplot(data=solar, aes(x=Date, y=PowerBill, colour=Solar)) + 
  geom_line()+
  geom_line(aes(x=Date, y=fit, colour="Fitted"))+
  geom_point()
dev.off()

simss <- function(nsims){
  out <- matrix(NA,nsims,53)
for(i in 1:nsims){ 
  rowz <-sample(1:nrow(solar),5,replace = F) 
train <- solar[-rowz,]
test <- solar[rowz,]
names(train) <- names(solar)
names(test) <- names(train)
 fit.ar1 <- gls(PowerBill ~ -1 + Solar+ Solar:winter + Solar:summer + summer:temp + winter:temp, corr = corAR1(form=~1|temp), method='ML',data = train)
coeffs <- t(t(coef(fit.ar1)))
X <- matrix(c(test$Solar=="N",test$Solar=="Y",test$Solar=="N"&test$winter==1,test$Solar=="Y"&test$winter==1,test$Solar=="N"&test$summer==1,test$Solar=="Y"&test$summer==1,(test$summer==1)*test$temp,(test$winter==1)*test$temp),5,8,byrow = F)
preds <- X %*% coeffs 
 
 out[i,1] <- mean(test$PowerBill-preds)
 out[i,2] <- sqrt(mean((test$PowerBill-preds)^2))
 out[i,(rowz+2)] <- preds 
}
out
  }

xxx <-cbind(solar,fit.ar1$fitted)
mean(xxx[38:49,8])
result <- simss(100)

quantile(result[,2],c(.025,.975), na.rm = T)
mean(result[,1])
mean(result[,2])

result[,3:53] <- ifelse(result[,3:53]==0,NA,result[,3:53])



fit.ar1 <- gls(PowerBill ~ Solar + Solar:winter + Solar:summer + summer:temp + winter:temp, corr = corAR1(form=~1|temp), method='REML',data = solar)
sum(fit.ar1$residuals^2)/sum((fit.ar1$fitted-mean(solar$PowerBill))^2)
summary(fit.ar1)


pdf("assump.pdf")
par(mfrow=c(1,2))
plot(fit.ar1$fitted,fit.ar1$residuals, xlab="Fitted Values",ylab="Residuals", main="Fitted Values v. Residuals")
hist(fit.ar1$residuals/29, xlab="Standardized Residuals", main="Standardized Residuals")
dev.off()
 ## Set up the R matrix for observations AND predictions
 N <- nrow(solar) #Number of observed time periods
 K <- 12 #number of time periods forward
 
 R <- diag(K+N)
 R <- (.03770666)^(abs(row(R)-col(R))) ## AR(1) correlation matrix
 R[1:6,1:6]
 pred.mn <- Xstar%*%bhat + R[N+(1:K), (1:N)]%*%solve(R[(1:N),(1:N)])%*%
   (Y-X%*%bhat) #conditional mean of MVN
 pred.var <- sig2*(R[N+(1:K),N+(1:K)]-R[N+(1:K),
                                        (1:N)]%*%solve(R[(1:N),(1:N)])%*%R[(1:N),N+(1:K)]) # conditional variance of MVN
 