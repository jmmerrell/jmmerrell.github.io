setwd("~/Documents/aSchool/536 - Stat Learning and Data Mining")
loc <- read.table("PredLocs.csv", header = T,sep=",")
cmaq <- read.table("CMAQ.csv",header=T,sep=",")
oz <- read.table("Ozone.csv",header=T,sep=",")

min.dist <- function(i){
  dist1 <- sqrt((cmaq$Longitude-loc$Longitude[i])^2+(cmaq$Latitude-loc$Latitude[i])^2)
  disC <- cbind(dist1,cmaq$CMAQ_O3)
  disC <- disC[order(disC[,1]),]
  c(disC[1:1000,1],disC[1:1000,2])
}



  
##data used to fit model

data <- loc
nrow(loc)
data[3:(2002)] <- 0
for(i in 1:2685){
  r <- min.dist(i)
  data[i,3:2002]<-r
}


allvars <- NULL

###lowest AIC 

less.nums <-c(1004:1012)

aa(less.nums)
all.nums <- setdiff(all.nums,less.nums)

allvars <- cbind(sapply(all.nums,function(x) paste("+ V",x,sep="")))
aic.mat <- matrix(0,length(allvars),2)
aa <- function(less.nums){
  
expr <- "lm(Ozone.8.hr.max. ~ -1 + "
for (ii in 1:length(less.nums)){
expr <- paste(expr, " + V",less.nums[ii],sep="")
}


expr.new <- paste(expr,allvars[i]," ,data=data)",sep="")
fit11 <-eval(parse(text=expr.new))
AIC(fit11)
print(expr.new)

}
#preds<- sapply(1:800,function(x) crossprod(1/as.numeric(data[x,4:1003]),as.numeric(data[x,1004:2003]))/sum(1/as.numeric(data[x,4:1003])))
####Subset selection

for(i in 1:length(allvars)){
expr.new <- paste(expr,allvars[i]," ,data=data)",sep="")
fit11 <-eval(parse(text=expr.new))
aic.mat[i,1]<-AIC(fit11)
aic.mat[i,2]<-allvars[i]
}

aic.mat[which(aic.mat[,1]==min(aic.mat[,1])),]

library(glmnet)
library(leaps)
x <- model.matrix(Ozone.8.hr.max. ~ .,data)[,-c(1:3,1004:2003)]
y <- data$Ozone.8.hr.max.

summary(cc)
##data set used for prediction
sqrt(mean((data$Ozone.8.hr.max.-data$V1004)^2))

mean((data$Ozone.8.hr.max.-data$V1004))
mean((data$Ozone.8.hr.max.-preds))

library(nlme)

up <- max(data$Latitude)
down <- min(data$Latitude)
lft <- min(data$Longitude)
rght <- max(data$Longitude)

lat <- down + (0:10)*(up-down)/10
lon <- lft + (0:10)*(rght-lft)/10

out <- matrix(0,20,4)
out2 <- out

library(ggplot2)
library(GGally)
pdf("spatial_data.pdf")
ggpairs(data[,c(3,1004:1006,1504:1506)],columnLabels = c("O3","Nearest(1)","Nearest(2)","Nearest(3)","Nearest(500)","Nearest(501)","Nearest(502)"))
dev.off()

for(i in 11:20){
nums <- which(data$Longitude<lon[i-9]&data$Longitude>=lon[i-10])
train <- data[-nums,]
test <- data[nums,1004:1020]
N <- nrow(train)
K <- nrow(test)
xxx <-gls(Ozone.8.hr.max.~-1+V1004+V1005+V1006+V1007+V1008+V1009+V1010+V1011+V1012+V1013+V1014+V1015+V1016+V1017+V1018+V1019+V1020,correlation = corExp(form = ~Longitude+Latitude,nugget = T),data=train)
coefz <-as.numeric(coef(xxx$modelStruct$corStruct,unconstrained = F))
R <-as.matrix(dist(cbind(data$Longitude,data$Latitude),method = "euclidean"))
w <- coefz[2]
r <- coefz[1]
R <- (-1/r)*R
R <- exp(R)
sig2 <- xxx$sigma^2
sigma <- sig2*((1-w)*R+w*diag(800))
X <- as.matrix(train[,1004:1020])
X.star <- as.matrix(test)
bhat <- as.matrix(coefficients(xxx))
y <- data$Ozone.8.hr.max.[-nums]
y.star <- data$Ozone.8.hr.max.[nums]

predmn <- X.star%*%bhat+R[N+(1:K),(1:N)]%*%solve(R[(1:N),(1:N)])%*%(y-X%*%bhat)
pred.var <-sig2*(R[N+(1:K),N+(1:K)]-R[N+(1:K),(1:N)]%*%solve(R[(1:N),(1:N)])%*%R[(1:N),N+(1:K)]) 
upp <- predmn + qt(0.975,1000)*sqrt(diag(pred.var)+sig2)
lowr <- predmn - qt(0.975,1000)*sqrt(diag(pred.var)+sig2) 

out[i,1] <- sqrt(mean((predmn-y.star)^2))
out[i,2] <- mean(y.star >= lowr & y.star <= upp)
out[i,3] <- mean(upp-lowr)
out[i,4] <- nrow(test)

}

crossprod(out[1:10,1],out[1:10,4])/800
crossprod(out[1:10,2],out[1:10,4])/800
crossprod(out[1:10,3],out[1:10,4])/800

mean(qq$fitted-data$Ozone.8.hr.max.)

par(mfrow=c(1,2))
####maps of data
library(maps)
library(LatticeKrig)
##O3 values
quilt.plot(data$Longitude,data$Latitude,data$V1004)
map('state',add=TRUE)
quilt.plot(data$Longitude,data$Latitude,preds)
map('state',add=TRUE)
##CMAQ values
quilt.plot(cmaq$Longitude,cmaq$Latitude,cmaq$CMAQ_O3)
map('state',add=TRUE)

##Distance Matrix
zzz <-gls(Ozone.8.hr.max.~-1+V1004+V1005+V1006+V1007+V1008+V1009+V1010+V1011+V1012+V1013+V1014+V1015+V1016+V1017+V1018+V1019+V1020,correlation = corExp(form = ~Longitude+Latitude,nugget = T),data=data)

coefz <-as.numeric(coef(zzz$modelStruct$corStruct,unconstrained = F))
R <-as.matrix(dist(cbind(data$Longitude,data$Latitude),method = "euclidean"))
w <- coefz[2]
r <- coefz[1]
R <- (-1/r)*R
R <- exp(R)
L <- t(chol(R))
sig2 <- zzz$sigma^2
sigma <- sig2*((1-w)*R+w*diag(800))
X <- as.matrix(data[,1004:1020])
X <- solve(L)%*%X

bhat <- as.matrix(coefficients(zzz))
y <- data$Ozone.8.hr.max.
y <- solve(L)%*%y


coeffss <- solve(t(X)%*%solve(sigma)%*%X)%*%t(X)%*%solve(sigma)%*%y
y.hat <- X%*%coeffss
resids <- y.hat-y
sum(resids^2)/sum((y-mean(y))^2)

qq <- summary(zzz)
1-sum((qq$fitted-data$Ozone.8.hr.max.)^2)/sum((mean(data$Ozone.8.hr.max.)-data$Ozone.8.hr.max.)^2)

summary(zzz)
varr <- mean((y.hat-y)^2)
pdf("spatial_assump.pdf")
par(mfrow=c(1,2))
hist(resids[abs(resids)<=(3.5*sqrt(varr))]/sqrt(varr),breaks=20, xlab = "Standardized Residuals", main="Standardized Residuals")
plot(zzz$fitted,zzz$residuals,ylab="Residuals",xlab="Fitted Values", main="Residuals v. Fitted Values")
dev.off()

corrs <- NULL
for(i in 1004:2000){
corrs[i-1003]<-cor(data[,1004],data[,i])
}

pdf("spatial_ACF.pdf")
par(mfrow=c(1,1))
plot(1:500,corrs[1:500], type="l",ylab="Correlation",xlab="Location Lag",main="Spatial ACF")
dev.off()

X <- as.matrix(data[,1003:1019])
predicc <-X %*% as.matrix(zzz$coefficients)

library(maps)
library(LatticeKrig)
##O3 values
pdf("spatial_predict.pdf")
quilt.plot(data$Longitude,data$Latitude,predicc)
map('state',add=TRUE)
dev.off()



