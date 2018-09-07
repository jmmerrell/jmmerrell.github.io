lst <- ls()
rm(list=lst[-c(which(lst=="data2"))])
setwd("C:/Users/merre/Desktop/jmmerrell.github.io/movie_random_forest")
data <- read.table("movie_data.txt",header = F,sep="|", fill = T,stringsAsFactors = F,quote = "")
data[,1]<- as.Date(data[,1],"%B %d %Y")
for(i in 8:15){
data[,i] <- as.numeric(data[,i])
}

colnames(data) <- c("date","movie","studio","genre","basedon","actionanim","factfict","budget","thrcount","threngag","fisrtweeknd","domgross","infdomgross","intgross","totgross","direct","compose","act1","act2","act3","act4","act5","act6","act7","act8","act9","act10","act11","act12","act13","act14","act15","act16","act17","act18","act19","act20","rating","franch")
###order data by date
data <- data[order(data$date),]
###Drop all movies with no inflation adjusted gross and without actors
data <- data[c(data$infdomgross>0 & data$act1!=""),]
###Create inflation adjusted budget
data <- cbind(data,data$infdomgross/data$domgross*data$budget)
###Get average franchise gross
data <- cbind(data,sapply(1:length(data$date),function(x) crossprod(data$infdomgross[1:x-1],data$franch[1:x-1]==data$franch[x] & data$franch[1:x-1]!="")/crossprod(data$franch[1:x-1]==data$franch[x] & data$franch[1:x-1]!="")))
###replace NAN and NA
data <- replace(data,is.na(data),0)
###Consolidate categorical variables
which.rows <- which(data$genre %in% c("Concert/Performance","Multiple Genres","Reality","Western"))
data$genre[which.rows]<-""
which.rows <- which(data$genre %in% c("Black Comedy"))
data$genre[which.rows]<-"Comedy"
###
which.rows <- which(data$basedon %in% c("Based on Ballet","Based on Musical Group","Based on Musical or Opera","Based on Play","Based on Song","Based on Song"))
data$basedon[which.rows]<-"Based on Theater"
which.rows <- which(data$basedon %in% c("Based on Factual Book/Article","Based on Religious Text"))
data$basedon[which.rows]<-"Based on NonFiction"
which.rows <- which(data$basedon %in% c("Based on Fiction Book/Short Story","Based on Folk Tale/Legend/Fairytale","Based on Game","Based on Movie","Based on Short Film","Based on Theme Park Ride","Based on Toy","Based on Web Series","Based on Spinoff","Compilation"))
data$basedon[which.rows]<-"Based on Fiction"
###
which.rows <- which(data$actionanim %in% c("Animation/Live Action","Digital Animation","Hand Animation","StopMotion Animation","Rotoscoping","Multiple Production Methods"))
data$actionanim[which.rows]<-"Animated"
which.rows <- which(data$actionanim %in% c("Live Action"))
data$actionanim[which.rows]<-"Live Action"

library(plyr)

###Get actor gross
library(doParallel)
library(foreach)
nCores <- 4
registerDoParallel(nCores)
start <- Sys.time()
actor<- matrix(0,length(data$date),2)
actsum <- foreach(j=18:37,.combine = "cbind") %dopar% {
    for (i in 1:length(data$date)){
      if(any(c("","Director","Co-Director")==data[i,j])){
        actor[i,1]<-0
        actor[i,2]<-0
      }
      else{
        ###Current sum
        which.rows <- unlist(sapply(18:37,function(x,y,z) which(data[1:y-1,x]==data[i,j]),y=i,z=j))
        actor[i,1] <- sum(data$infdomgross[which.rows])
        ###Last 10 years sum
        date.rows <- which(data[which.rows,1]>=(data[i,1]-3652))
        actor[i,2] <- sum(data$infdomgross[which.rows[date.rows]])
        
      }
    }
actor
}


data <- cbind(data,actsum)


row.add <- matrix(0,length(data$date),1)
###Add in top grossing actor for each movie
for(i in 1:length(data$date)){
  row.add[i,1]<- sum(sapply(1:1,function(x,y,z) 
    y[rev(order(y))][x],z=i,y=unlist(data[i,c(seq(from=43,to=81, by=2))])))
}
data <- cbind(data,row.add)
###Add in top 3 grossing actor for each movie
for(i in 1:length(data$date)){
  row.add[i,1]<- sum(sapply(1:3,function(x,y,z) 
    y[rev(order(y))][x],z=i,y=unlist(data[i,c(seq(from=43,to=81, by=2))])))
}
data <- cbind(data,row.add)
###Add in top 5 grossing actor for each movie
for(i in 1:length(data$date)){
  row.add[i,1]<- sum(sapply(1:5,function(x,y,z) 
    y[rev(order(y))][x],z=i,y=unlist(data[i,c(seq(from=43,to=81, by=2))])))
}
data <- cbind(data,row.add)


###10 yr and cum director gross

for (i in 1:length(data$date)){
  if(data[i,16]==""){
    actor[i,1]<-0
    actor[i,2]<-0
  }
  else{
  ###Current sum
  which.rows <- unlist(which(data[1:i-1,16]==data[i,16]))
  actor[i,1] <- sum(data$infdomgross[which.rows])
  ###Last 10 years sum
  date.rows <- which(data[which.rows,1]>=(data[i,1]-3652))
  actor[i,2] <- sum(data$infdomgross[which.rows[date.rows]])
  }
}
data <- cbind(data,actor)

###10 yr and cum composer gross

for (i in 1:length(data$date)){
  if(data[i,17]==""){
    actor[i,1]<-0
    actor[i,2]<-0
  }
  else{
  ###Current sum
  which.rows <- unlist(which(data[1:i-1,17]==data[i,17]))
  actor[i,1] <- sum(data$infdomgross[which.rows])
  ###Last 10 years sum
  date.rows <- which(data[which.rows,1]>=(data[i,1]-3652))
  actor[i,2] <- sum(data$infdomgross[which.rows[date.rows]])
  }
}
data <- cbind(data,actor)
###avg studio gross
for (i in 1:length(data$date)){
  if(data[i,3]==""){
    actor[i,1]<-0
    actor[i,2]<-0
  }
  else{
    ###Current sum
    which.rows <- unlist(which(data[1:i-1,3]==data[i,3]))
    actor[i,1] <- sum(data$infdomgross[which.rows])/length(which.rows)
    ###Last 10 years sum
    date.rows <- which(data[which.rows,1]>=(data[i,1]-3652))
    actor[i,2] <- sum(data$infdomgross[which.rows[date.rows]])/length(date.rows)
  }
}
data <- cbind(data,actor)

colnames(data) <- c("date","movie","studio","genre","basedon","actionanim","factfict","budget","thrcount","threngag","firstweeknd","domgross","infdomgross","intgross","totgross","direct","compose","act1","act2","act3","act4","act5","act6","act7","act8","act9","act10","act11","act12","act13","act14","act15","act16","act17","act18","act19","act20","rating","franch","infbudget","avgfranchgross","act1gross","act1grosscur","act2gross","act2grosscur","act3gross","act3grosscur","act4gross","act4grosscur","act5gross","act5grosscur","act6gross","act6grosscur","act7gross","act7grosscur","act8gross","act8grosscur","act9gross","act9grosscur","act10gross","act10grosscur","act11gross","act11grosscur","act12gross","act12grosscur","act13gross","act13grosscur","act14gross","act14grosscur","act15gross","act15grosscur","act16gross","act16grosscur","act17gross","act17grosscur","act18gross","act18grosscur","act19gross","act19grosscur","act20gross","act20grosscur","top1act","top3act","top5act","directgross","directgrosscur","compgross","compgrosscur","avgstudiogross","avgstudiogrosscur")
data2 <- data[c(data$studio!="" & data$genre!="" & data$actionanim!="" & data$avgstudiogrosscur>0 & data$budget>0 & data$infdomgross>1000000 &data$thrcount>0 & data$date>="2000-01-01" & data$direct!=""),]
data2 <- data2[c(is.na(data2$studio)==F),]
for(i in 1:90){
i <-1

if(grepl("gross",names(data2)[i])|grepl("budget",names(data2)[i])){
  
 data2[,i]<-data2[,i]/1000000 
}
}
saveRDS(data2, "movie_data.rds")
######################################
######################################
###Done with data manipultation########
######################################
######################################
######################################
######################################
data2 <- readRDS("movie_data.rds")
##Create table with variable names class and explanation
var.names <- c(colnames(data)[c(1,2,3,4,5,6,7,8,9,13,16,17,38,39)],"actx")
var.class <- c("date","character","factor","factor","factor","factor","factor","integer","integer","integer","character","character","factor","character","character")
var.explain <- c("date of premier","movie name","studio that produced movie","genre of movie","what the story is based on","live action or animated","factual or fictional","cost to make","how many theaters it opened in","inflation adjusted revenue","director","composer","MPAA rating","which franchise","The xth actor listed in the movie")
datatab <- data.frame(var.names,var.class,var.explain)
colnames(datatab)<- c("Name","Class","Explanation")
library(xtable)
xtable(datatab,include.rownames=FALSE)


##Create table with variable names class and explanation for computed variables
var.names <- colnames(data)[c(40,41,82,83,84,85,86,87,88,89,90)]
var.class <- rep("integer",11) 
var.explain <- c("inflation adjusted cost to make movie","average revenue for all previous movies in franchise","cumulative career gross of the top actor(last 10 yrs)","cumulative career gross of the top 3 actors(last 10 yrs)","cumulative career gross of the top 5 actors(last 10 yrs)","cumulative career gross of director","cumulative career gross of director(last 10 yrs)","cumulative career gross of composer","cumulative career gross of composer(last 10 yrs)","cumulative career gross of studio","cumulative career gross of studio(last 10 yrs)")
datatab <- data.frame(var.names,var.class,var.explain)
colnames(datatab)<- c("Name","Class","Explanation")
library(xtable)
xtable(datatab,include.rownames=FALSE)

##Create table with variable names class and explanation for computed variables
var.names <- colnames(data)[c(3:5,7,9,38,40,41,82,83,86,90)]
datatab <- data.frame(var.names)
colnames(datatab)<- c("Variable Name")
library(xtable)
xtable(datatab,include.rownames=FALSE)

# poss.cols <- c("studio","genre","basedon","factfict","thrcount","infdomgross","rating","infbudget","avgfranchgross","top1act","top3act","directgrosscur","compgrosscur","avgstudiogrosscur")
# poss.combs <- sapply(1:14, function(x,y) combn(y,x,simplify = T),y=poss.cols)

#model.matrix[model.matrix[,2]>.7&model.matrix[,4]>.95&model.matrix[,3]<87050,1]
# ###Create a matrix to find the best variable subsets
# model.matrix <- matrix(0,2^14-1,4)
#    for(j in 1:14){
#        for(k in 1:choose(14,j)){
#              modtext <- ""
#           for(m in 1:j){
#                  modtext <- paste(modtext,poss.combs[[j]][m,k],sep="+")
#                }
#           modl <- eval(parse(text=paste("glm(infdomgross~",substring(modtext, 2),",data=data2,family=Gamma(link=log))", sep="")))
#            r2 <- 1-modl$deviance/modl$null.deviance
#            aic.mod <- modl$aic
#            preds <- predict.glm(modl,se.fit=T,type="response")
#            low.pred <- preds$fit -qgamma(.025,log(preds$fit)*preds$residual.scale,preds$residual.scale)*preds$se.fit
#            up.pred <- preds$fit +qgamma(.975,log(preds$fit)*preds$residual.scale,preds$residual.scale)*preds$se.fit
#            pred.perc <- mean(modl$y>=low.pred&modl$y<=up.pred)
#            model.matrix[sum(sapply(1:j,function(x) choose(5,x-1)))+k-1,1]<-modtext
#            model.matrix[sum(sapply(1:j,function(x) choose(5,x-1)))+k-1,2]<-r2
#            model.matrix[sum(sapply(1:j,function(x) choose(5,x-1)))+k-1,3]<-aic.mod
#            model.matrix[sum(sapply(1:j,function(x) choose(5,x-1)))+k-1,4]<-pred.perc
#            
#              }
#      }
# 
library(randomForest)

data3 <- data2[,c(13,4:7,9,40,41,82:91)]
data3[,2:5] <- lapply(data3[,2:5], factor)
train <- sample(1:nrow(data3),nrow(data3)*2/3)
rf <- randomForest(infdomgross ~ ., data=data3, subset=train)

##This shows optimal number of trees
pdf("optim_trees.pdf")
plot(rf)
dev.off()

##Optimal number of variables to select at each node
oob.err=double(17)
test.err=double(17)

#mtry is no of Variables randomly chosen at each split
for(mtry in 1:17) 
{
  rf<-randomForest(infdomgross ~ ., data=data3, subset=train, mtry=mtry, ntree=100)
  oob.err[mtry] = rf$mse[100] #Error of all Trees fitted
  
  pred<-predict(rf,data3[-train,]) #Predictions on Test Set for each Tree
  test.err[mtry]= with(data3[-train,], mean( (infdomgross - pred)^2)) #Mean Squared Test Error
  
  cat(mtry," ") #printing the output to the console
  
}

pdf("mtry.pdf")
matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))
dev.off()

##Function to calculate R2, MSE, Bias, prediction interval coverage, and AIC
library(HH)
out.mat <- matrix(0,1,3)
colnames(out.mat)<-c("r2","bias","rmse")
model.stats <- function(x){
  train <- sample(1:nrow(x),nrow(x)*2/3)
  rf <- randomForest(infdomgross ~ ., data=x, subset=train, mtry=8, ntree=100)
  out.mat[1,1] <- mean(rf$rsq)
  out.mat[1,2] <- mean(rf$predicted/1000000-x$infdomgross[train]/1000000)
  out.mat[1,3] <- sqrt(mean(((rf$predicted/1000000-x$infdomgross[train]/1000000)^2)))
  return(out.mat)
}


######MSE and Bias Simulation Study
library(xtable)
modl <-randomForest(infdomgross ~ ., data=data3,mtry=8, ntree=100)

data.off <- cbind(data3$infdomgross/1000000,modl$predicted/1000000)
data.off1 <- data.off[data.off[,1]>400,]
data.off2 <- data.off[data.off[,1]>80&data.off[,1]<120,]
outs <- matrix(c(mean(data.off1[,1]),mean(data.off1[,2]),mean(data.off2[,1]),mean(data.off2[,2])),2,2,byrow=T)
row.names(outs)<- c("Movies Above $400 million","Movies Between $80 and $120 million")
colnames(outs)<- c("Mean of Observed Values","Mean of Predicted Values")
xtable(outs)

Sys.time()
nCores <- 1
registerDoParallel(nCores)
start <- Sys.time()
mat.rows <- foreach(ii=1:1000,.combine = "rbind") %do% {
model.stats(data3)
}
Sys.time()

library(ggplot2)
library(gridExtra)

plot1 <- ggplot()+
  geom_histogram(aes(x=mat.rows[,1], y=..density..,alpha=.2),  fill="red",bins=50,binwidth=.01) +
  geom_density()+
  xlab("R-Squared")+
  ggtitle("R-Squared Simulations")+
  theme_grey(base_size = 18) +
  theme(legend.position="none")
plot2 <- ggplot()+
  geom_histogram(aes(x=mat.rows[,2], y=..density..,alpha=.2),  fill="red",bins=50,binwidth=.01) +
  geom_density()+
  xlab("Bias")+
  ggtitle("Bias Simulations")+
  theme_grey(base_size = 18) +
  theme(legend.position="none")
plot3 <- ggplot()+
  geom_histogram(aes(x=mat.rows[,3], y=..density..,alpha=.2),  fill="green",bins=50,binwidth=.1) +
  geom_density()+
  xlab("RMSE")+
  ggtitle("RMSE Simulations")+
  theme_grey(base_size=18)+
theme(legend.position="none")

pdf("biasmse.pdf", width = 12, height = 8, onefile=T, paper='A4r') # Open a new pdf file
grid.arrange(plot1, plot2,plot3, ncol=2, nrow=2) # Write the grid.arrange in the file
dev.off()






