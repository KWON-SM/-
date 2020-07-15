data(infert)
str(infert)

# 반응변수 : case
table(infert$case)
# 0   1 
# 165  83 
prop.table(table(infert$case))
# 0         1 
# 0.6653226 0.3346774 

#데이터 요약
summary(infert)

#결측치 확인 ---> 없음
colSums(is.na(infert)) 

# 반응변수(Class)를 Y, 설명변수를 X 라는 데이터프레임으로 분리
library('dplyr')
Y <- infert$case
X <- infert[,c("age", "parity", "induced", "spontaneous")]
head(X)
Y    <- as.factor(Y)
X$age    <- as.integer(X$age)
X$parity    <- as.integer(X$parity)
X$induced    <- as.integer(X$induced)
X$spontaneous    <- as.integer(X$spontaneous)
str(X)

# 설명변수(독립변수) 표준화하기 
X2 <- scale(X)
var(X2)

XY <- data.frame(Y,X2)
str(XY)
# train data 70%
library(caret)
set.seed(123) 
parts <- createDataPartition(XY$Y, p=0.7)
data.train <- XY[parts$Resample1, ]
table(data.train$Y)
# 0   1 
# 108  66 
prop.table(table(data.train$Y))
# test data 30%
data.test <- XY[-parts$Resample1, ]
table(data.test$Y)
# 0  1 
# 57 17 
prop.table(table(data.test$Y))

#### 훈련용 데이터로 나이브 베이즈 모델을 생성하기
nai.fit <- naiveBayes(Y~., data=data.train)

# 테스트 데이터로 예측을 수행하고, 나이브 베이즈 모델의 성능 평가하기
?predict
nai.pred <- predict(nai.fit, data.test, type='class')
nai.tb <- table(nai.pred, data.test$Y)
nai.tb
# nai.pred  0  1
#        0 50 11
#        1  7  6

# accuracy : 0.7567568
mean(data.test$Y == nai.pred)

# error rate : 0.2432432 
(1-sum(diag(nai.tb))/sum(nai.tb))

# ROC곡선 & AUC 
library(ROCR)

nb.pred <- prediction(as.integer(nai.pred), as.integer(data.test$Y))
roc     <- performance(nb.pred, measure = "tpr", x.measure = "fpr")
roc.x = unlist(slot(roc, 'x.values'))
roc.y = unlist(slot(roc, 'y.values'))
plot(x=c(0, 1), y=c(0, 1), type="l", col="red", lwd=2,
     ylab="True Positive Rate", xlab="False Positive Rate")
lines(x=roc.x, y=roc.y, col="orange", lwd=2)

# AUC (The Area Under an ROC Curve)
auc <- performance(nb.pred, measure="auc")
auc <- auc@y.values[[1]]
auc  # 0.6150671

#### 로지스틱 회귀 모델 적합


# test data 70, train data 30 
set.seed(123) 
train <- sample(1:nrow(XY), size=0.7*nrow(XY), replace=F)
test <- (-train)
Y.test <- Y[test]
scales::percent(length(train)/nrow(XY))
# 70%
head(train)

#### train data 로지스틱 회귀 모델 적합
?glm
glm.fit <- glm(Y~., data=XY, family=binomial(link="logit"), subset=train)
summary(glm.fit)
# Call:
#   glm(formula = Y ~ ., family = binomial(link = "logit"), data = XY, 
#       subset = train)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -1.5351  -0.7702  -0.5038   0.8424   2.7075  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  -1.0585     0.1845  -5.738 9.60e-09 ***
#   age           0.2748     0.1775   1.548 0.121719    
# parity       -0.9511     0.2616  -3.636 0.000277 ***
#   induced       0.9196     0.2412   3.812 0.000138 ***
#   spontaneous   1.4148     0.2441   5.795 6.83e-09 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 255.13  on 207  degrees of freedom
# Residual deviance: 210.33  on 203  degrees of freedom
# (332 observations deleted due to missingness)
# AIC: 220.33
# 
# Number of Fisher Scoring iterations: 4

#후진 소거법(backward elimination)
step(glm.fit, direction="backward")
# Start:  AIC=220.33
# Y ~ age + parity + induced + spontaneous
# 
#                Df Deviance    AIC
# <none>             210.33 220.33
# - age          1   212.75 220.75
# - parity       1   226.42 234.42
# - induced      1   226.48 234.48
# - spontaneous  1   254.92 262.92
# 
# Call:  glm(formula = Y ~ age + parity + induced + spontaneous, family = binomial(link = "logit"), 
#            data = XY, subset = train)
# 
# Coefficients:
#   (Intercept)          age       parity      induced  spontaneous  
# -1.0585       0.2748      -0.9511       0.9196       1.4148  
# 
# Degrees of Freedom: 207 Total (i.e. Null);  203 Residual
# (332 observations deleted due to missingness)
# Null Deviance:	    255.1 
# Residual Deviance: 210.3 	AIC: 220.3

# 후진소거법의 모델 적합하고, 모델의 유의성 검정하기
glm.fit2 <- glm(Y ~ age + parity + induced + spontaneous, 
                data=XY, family=binomial(link="logit"), subset=train)

anova(glm.fit2, test="Chisq")

# test data로 모델 성능 평가 
glm.probs <- predict(glm.fit2, XY[test,], type="response")
glm.pred <- ifelse(glm.probs > .5, 1, 0)
table(Y.test, glm.pred)
#       glm.pred
# Y.test  0  1
#     0 36 12
#     1 16 11

# accuracy
mean(Y.test == glm.pred) #0.6266667
# error rate
mean(Y.test != glm.pred) #0.3733333

# ROC : tpr & fpr 
glm.pred <- prediction(glm.probs, Y.test)
roc <- performance(glm.pred, measure = "tpr", x.measure = "fpr")
roc.x = unlist(slot(roc, 'x.values'))
roc.y = unlist(slot(roc, 'y.values'))
plot(x=c(0, 1), y=c(0, 1), type="l", col="red", lwd=2,
     ylab="True Positive Rate", xlab="False Positive Rate")
lines(x=roc.x, y=roc.y, col="orange", lwd=2)

# AUC (The Area Under an ROC Curve)
auc <- performance(glm.pred, measure = "auc")
auc <- auc@y.values[[1]]
auc  # 0.6751543

