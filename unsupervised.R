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

df21 <- read.csv(file = 'final_data2.csv', sep = ";")

df21 <- df21 %>% rename( Happiness_score=Ladder.score,Region = Regional.indicator,
                         GDP=Logged.GDP.per.capita, Social_support=Social.support, Freedom=Freedom.to.make.life.choices,
                         Corruption=Perceptions.of.corruption)
rownames(df21) <- df21$Country

# FINDING MISSING VALUES AND SUBSTITUTE MISSING VALUES WITH MEAN
sapply(df21, function(x)sum(is.na(x))) 

sapply(df21, class)
num_df21_stand <- scale(df21[,-c(1:2)])
head(df21)
summary(num_df21_stand)

###CLUSTERING###

library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

distance <- get_dist(num_df21_stand)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))


#k-means clustering

k2 <- kmeans(num_df21_stand, centers = 2, nstart = 25)
str(k2)

fviz_cluster(k2, data = num_df21_stand)

k3 <- kmeans(num_df21_stand, centers = 3, nstart = 25)

k4 <- kmeans(num_df21_stand, centers = 4, nstart = 25)
k5 <- kmeans(num_df21_stand, centers = 5, nstart = 25)

# plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = num_df21_stand) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = num_df21_stand) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = num_df21_stand) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = num_df21_stand) + ggtitle("k = 5")

aggregate(num_df21_stand,by=list(k3$cluster),FUN=mean)

num_df21_stand <- data.frame(num_df21_stand, k3$cluster)

ris3 <- eclust(num_df21_stand, "kmeans", k=3)
fviz_silhouette(ris3)
sil3 <- ris3$silinfo$widths
neg_sil_index3 <- which(sil3[,'sil_width']<0)
sil3[neg_sil_index3,]

library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)

set.seed(123)

fviz_nbclust(num_df21_stand, kmeans, method = "wss")
fviz_nbclust(num_df21_stand, kmeans, method = "silhouette")

set.seed(123)
final <- kmeans(num_df21_stand,3, nstart = 25)
print(final)

fviz_cluster(final, data = num_df21_stand)

#hierarchical clustering

library(dendextend)

# Dissimilarity matrix
d <- dist(num_df21_stand, method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "complete" )

# methods to assess
m <- c( "average", "single", "complete", "ward")
names(m) <- c( "average", "single", "complete", "ward")

# function to compute coefficient
ac <- function(x) {
  agnes(num_df21_stand, method = x)$ac
}

map_dbl(m, ac)

hc3 <- agnes(num_df21_stand, method = "ward")
pltree(hc3, cex = 0.6, hang = -1, main = "Ward Linkage Dendogram") 

# Ward's method
hc5 <- hclust(d, method = "ward.D2" )

# Cut tree into 4 groups
sub_grp <- cutree(hc5, k = 5)

# Number of members in each cluster
table(sub_grp)
plot(hc5, cex = 0.6)
rect.hclust(hc5, k = 5, border = 2:5)

gap_stat1 <- clusGap(num_df21_stand, FUN = hcut, nstart = 25, K.max = 10, B = 50) 
fviz_gap_stat(gap_stat1) 

