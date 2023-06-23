library(ggplot2)
library(dplyr)
library(corrplot)
library(ggpubr)
library(Hmisc)
library(olsrr)
library(jtools)
library(lmtest)
library(car)
library(ggfortify)
library(lmtest)
library(caret)
library(tidyverse)
library(leaps)
library(ISLR)
library(MASS)
library(pls)
library(glmnet)
library(tree)
library(rpart) 
library(rpart.plot)
library(randomForest)
library(rsample)


###PRE-PROCESS DATA###

df21 <- read.csv("final_data2.csv",sep = ';')
names(df21)

#rename columns

df21 <- df21 %>% rename( Happiness_score=Ladder.score,Region = Regional.indicator,
                         GDP=Logged.GDP.per.capita, Social_support=Social.support, Freedom=Freedom.to.make.life.choices,Corruption=Perceptions.of.corruption,Dystopia=Dystopia...residual)

rownames(df21) <- df21$Country
names(df21)

#check for missing values
sapply(df21, function(x)sum(is.na(x)))


###CHECK FOR OUTLIERS###
df21<- subset(df21, Country!='Denmark')
df21<- subset(df21, Country!='Chad')
df21<- df21[which(df21$Suicide_rate<21),]
num_df21 <- subset(df21, select = -c(Country,Region))

df21_standardize <- as.data.frame(scale(num_df21))


par(mar=c(9,6,2,2))

boxplot(num_df21, main = "Happiness report data",las =2)

sapply(num_df21, class)
summary(num_df21)
summary(df21_standardize)

###FINDING CORRELATION BETWEEN PREDICTORS###
library(reshape2)

# creating correlation matrix
corr_mat <- round(cor(num_df21),2)

# reduce the size of correlation matrix
melted_corr_mat <- melt(corr_mat)
head(melted_corr_mat)

# plotting the correlation heatmap
library(ggplot2)
ggplot(data = melted_corr_mat, aes(x=Var1, y=Var2,
                                   fill=value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value),
            color = "black", size = 3)+
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 9, hjust = 1))+
  coord_fixed()

###CHECKING FOR NORMALITY###

summary(num_df21$Happiness_score)
qqnorm(num_df21$Happiness_score)#the values are distributed linearly


shapiro.test(num_df21$Happiness_score)#p-value> 0.05. I accept the null and is normally distributed
ggqqplot(num_df21$Happiness_score)

ggdensity(num_df21, x = "Happiness_score", fill = "lightgray", title = "Happiness distribution") +
  stat_overlay_normal_density(color = "red", linetype = "dashed")

###CHECKING FOR LINEARITY###

library(car)
scatterplotMatrix(num_df21, regLine = TRUE) # scatterplot matrix 

num_df21_col <- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = corr_mat, col = num_df21_col, Colv = NA, symm = TRUE, Rowv = NA) # heatmap

corrplot(corr_mat, type = "upper", tl.col = "black", tl.srt = 45)

library(PerformanceAnalytics)

chart.Correlation(df21_standardize, histogram=TRUE, pch=19)
chart.Correlation(num_df21, histogram=TRUE, pch=19)
### MULTIPLE LINEAR REGRESSION ###

regressor_lm = lm(formula = Happiness_score ~ ., data = num_df21)
summary(regressor_lm)

#check linearity with residuals vs fitted values

autoplot(regressor_lm)

#check fro homoskedasticity with Breusch-Pagan test

bptest(regressor_lm)#the p-value>.05 so we cannot reject the null of homoskedasticity

#check for multicollinearity

vif(regressor_lm) 
sqrt(vif(regressor_lm))#the vif indicates that there is a problem o multicollinearity

#plot diagnostic
#par(mar=c(9,6,2,2))
plot(regressor_lm)

####performing k-fold cross validation

set.seed(150)
train_control <- trainControl(method = "cv",number = 10)
model <- train(Happiness_score ~., data = num_df21, 
               method = "lm",
               trControl = train_control)

print(model)#get model metrics
model$results
#partition data and fit model on train set

new_df <- createDataPartition(y = num_df21$Happiness_score, p = .5, list=FALSE)
train <- num_df21[new_df, ]
test <- num_df21[-new_df, ]

new_df21 <- train (Happiness_score ~., data = train, method="lm", metric = "Rsquared", trControl = train_control)

new_df21$results
new_df21$resample %>% pivot_longer(-Resample) %>%qplot(name, value, geom="boxplot", data=.)
print(new_df21)

####model selection using cp, bic, rss and adjustedr2###

regfit.full = regsubsets(Happiness_score~.,num_df21)
reg.summary<-summary(regfit.full)
summary(regfit.full)
names(reg.summary)
reg.summary$rsq
par(mar=c(1,1,1,1))
par(mar=c(1,1,1,1) + 0.1, mgp=c(5,1,0), oma=c(3,3,3,3))
plot(reg.summary$rss,xlab="Number of variables",ylab="RSS",type="b")
plot(reg.summary$adjr2,xlab="Number of variables",ylab="Adjr2",type="l")

which.max(reg.summary$adjr2)
points(which.max(reg.summary$adjr2),reg.summary$adjr2[which.max(reg.summary$adjr2)],col="blue",cex=2,pch=20)
#plot cp
plot(reg.summary$cp,xlab="Number of variables",ylab="Cp",type="b")
#plot bic
plot(reg.summary$bic,xlab="Number of variables",ylab="BIC",type="l")

###DIMENSION REDUCTION###

##PRINCIPAL COMPONENT REGRESSION##
library(pls)

set.seed(17)
pcr.fit = pcr (Happiness_score~., data=num_df21, scale=TRUE, validation="CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type="MSEP", legendpos="topright", main="PCR")

set.seed(11)
train=sample(seq(96), 48, replace=FALSE)
pcr.fit = pcr(Happiness_score~., data = num_df21, subset = train, scale=TRUE, validation = "CV")
validationplot (pcr.fit, val.type = "MSEP", legendpos = "topright", main ="PCR test")

x=model.matrix(Happiness_score~.-1, data=num_df21)
y=num_df21$Happiness_score
pcr.pred = predict(pcr.fit,x[-train,], ncomp=2)
mean((pcr.pred-y[-train])^2)

pcr.fit=pcr(y~x, scale=TRUE, ncomp=2)
summary(pcr.fit)

### PARTIAL LEAST SQUARES ###
pls.fit = plsr(Happiness_score~., data = num_df21, subset=train, scale=TRUE, validation="CV")
summary(pls.fit)
par(mar=c(1,1,1,1) + 0.1, mgp=c(5,1,0), oma=c(3,3,3,3))
validationplot(pls.fit, val.type="MSEP", legendpos="topright", main="PLSR test")

pls.pred=predict(pls.fit,x[-train,],ncomp=6)
mean((pls.pred-y[-train])^2)

pls.fit=plsr(Happiness_score~., data=num_df21, scale=TRUE, ncomp=6)
summary(pls.fit)

###TREE BASED MODELS###

##Regression Trees##
#visualize the tree on all the data using ctree

library(party)
library(partykit)
num_df21
tree1 <- ctree(Happiness_score~., num_df21)
plot(tree1, cgp = gpar(fontsize=4))

tree2 <- rpart(Happiness_score~., num_df21)
printcp(tree2)
rpart.plot(tree2)
pred <- predict(tree2, newdata = num_df21)
RMSE(pred = pred, obs = num_df21$Happiness_score)

#create trees on data partition 

set.seed(101)
train = sample(1:nrow(num_df21), nrow(num_df21)/2)
tree = tree(Happiness_score~.,num_df21,subset=train)
summary(tree)

plot(tree); text(tree, pretty=0)

cv.df = cv.tree(tree)
par(mar=c(1,1,1,1) + 0.1, mgp=c(5,1,0), oma=c(3,3,3,3))
plot(cv.df$size, cv.df$dev, type='b')

yhat=predict(tree,newdata=num_df21[-train,])
test=num_df21[-train,"happiness_score"]
plot(yhat,test)
abline(0,1)
tree4 <- rpart(Happiness_score~., num_df21, subset=train)
printcp(tree4)

rpart.plot(tree4)
plotcp(tree4, minlin=FALSE, upper=c("splits"))

#create partition in data and visualize tree
set.seed(101)
train = sample(1:nrow(df21_standardize), nrow(df21_standardize)/2)
h.tree = tree(Happiness_score~.,df21_standardize,subset=train)
summary(tree)
plot(tree,); text(tree, pretty=0)

cv.df21_standardize =cv.tree(h.tree ,FUN=prune.tree )
cv.df21_standardize

plot(cv.df21_standardize$size,cv.df21_standardize$dev,type="b", lwd=3,col="blue",
     xlab="Final Nodes", ylab="test error")

#visualize the optimal tree using pruning

prune.df21_standardize=prune.tree(h.tree, best =3)
plot(prune.df21_standardize,lwd=3)
text(prune.df21_standardize ,pretty =0,cex=1.2,col="blue")


pred <- predict(tree, newdata = df21_standardize[-train,])
test=df21_standardize[-train,"Happiness_score"]
RMSE(pred = pred, obs = df21_standardize$Happiness_score)

###BAGGING
#create train and test

set.seed(123)
part <- createDataPartition(num_df21$Happiness_score,p=.7,list=FALSE,times=1)

train_df_t <- num_df21[part,]
test_df_t<-num_df21[-part,]
ctrl <- trainControl(method = "cv",  number = 10) 
bagged_cv <- train(
  Happiness_score ~ .,
  data = train_df_t,
  method = "treebag",
  trControl = ctrl,
  importance = TRUE
)
library(ipred)
# assess 10-50 bagged trees
ntree <- 10:100

# create empty vector to store OOB RMSE values
rmse <- vector(mode = "numeric", length = length(ntree))

for (i in seq_along(ntree)) {
  # reproducibility
  set.seed(123)
  
  # perform bagged model
  model <- bagging(
    formula = Happiness_score ~ .,
    data    = train_df_t,
    coob    = TRUE,
    nbagg   = ntree[i]
  )
  # get OOB error
  rmse[i] <- model$err
}

plot(ntree, rmse, type = 'l', lwd = 2)
abline(v = 49, col = "red", lty = "dashed")
bagged_cv
plot(varImp(bagged_cv), 20) 
pred <- predict(bagged_cv,test_df_t)
RMSE(pred, test_df_t$Happiness_score)


###RANDOM FOREST###

library(randomForest)
library(gbm)

#use the same partition used for bagging
set.seed(123)

# default RF model
m1 <- randomForest(
  formula = Happiness_score ~ .,
  data    = train_df_t
)
# number of trees with lowest MSE
which.min(m1$mse)
# RMSE of this optimal random forest
sqrt(m1$mse[which.min(m1$mse)])
# create training and validation data 
set.seed(123)
valid_split <- initial_split(train_df_t, .8)

# training data
df_train_v2 <- analysis(valid_split)

# validation data
df_valid <- assessment(valid_split)
x_test <- df_valid[setdiff(names(df_valid), "Happiness_score")]
y_test <- df_valid$Happiness_score

rf_oob_comp <- randomForest(
  formula = Happiness_score ~ .,
  data    = df_train_v2,
  xtest   = x_test,
  ytest   = y_test
)

# extract OOB & validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous(labels = scales::dollar) +
  xlab("Number of trees")

# names of features
features <- setdiff(names(train_df_t), "Happiness_score")

set.seed(123)

m2 <- tuneRF(
  x          = train_df_t[features],
  y          = train_df_t$Happiness_score,
  ntreeTry   = 500,
  mtryStart  = 5,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = FALSE      # to not show real-time progress 
)
OOB_RMSE <- vector(mode = "numeric", length = 100)
library(ranger)
optimal_ranger <- ranger(
  formula         = Happiness_score ~ ., 
  data            = train_df_t, 
  num.trees       = 500,
  mtry            = 10,
  min.node.size   = 5,
  sample.fraction = .8,
  importance      = 'impurity'
)

  
optimal_ranger$variable.importance %>% 
  tidy() %>%
  dplyr::arrange(desc(x)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(reorder(names, x), x)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top 25 important variables")
