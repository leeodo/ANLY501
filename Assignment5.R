#setup libraries
library(rpart)
library(rpart.plot)
library(rattle)
library(ggplot2)

#Read the data
car_eval = read.csv("car_evaluation.csv", header = FALSE)
names(car_eval) = c('buy_price', 'maint_cost', 'num_door',
                    'num_person', 'lug_boot', 'safty', 'decision')

#change data types
car_eval$decision = as.factor(car_eval$decision)
car_eval$buy_price = as.factor(car_eval$buy_price)
car_eval$maint_cost = as.factor(car_eval$maint_cost)
car_eval$num_door = as.factor(car_eval$num_door)
car_eval$num_person = as.factor(car_eval$num_person)
car_eval$lug_boot = as.factor(car_eval$lug_boot)
car_eval$safty = as.factor(car_eval$safty)
str(car_eval)

#Train test split
ind = sample(1728, floor(1728*0.7))
train = car_eval[ind,]
test = car_eval[-ind,]
X_test = test[,1:6]
y_test = test[,7]

#perform Decision Tree
set.seed(2021)
DT <- rpart(decision ~ ., data = train, method="class")
summary(DT)
rpart.plot(DT)

plotcp(DT)

DT2 <- rpart(decision ~ ., data = car_eval, method="class", cp = 0.037)
summary(DT2)
rpart.plot(DT2)

DT2_pred = predict(DT2, X_test, type = 'class')
confusion_matrix = as.data.frame(table(DT2_pred, y_test))

ggplot(data = confusion_matrix,
       mapping = aes(x = DT2_pred,
                     y = y_test)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "blue",
                      high = "red",
                      trans = "log")

DT3 <- rpart(decision ~ ., data = car_eval, method="class",
             parms = list(split="information"))
summary(DT3)
plotcp(DT3)

DT3 <- rpart(decision ~ ., data = car_eval, method="class", cp = 0.04,
             parms = list(split="information"))
summary(DT3)
rpart.plot(DT3)


DT3_pred = predict(DT3, X_test, type = 'class')

confusion_matrix = as.data.frame(table(DT3_pred, y_test))

ggplot(data = confusion_matrix,
       mapping = aes(x = DT3_pred,
                     y = y_test)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "blue",
                      high = "red",
                      trans = "log")

caret::confusionMatrix(DT3_pred, y_test, positive="true")


caret::confusionMatrix(DT2_pred, y_test, positive="true")
