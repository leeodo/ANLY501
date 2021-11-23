#loading the libraries
library(e1071)
library(dplyr)
library(ggplot2)
library(gridExtra)

#specify column classes
classes = c("numeric", "factor", "factor", "numeric", "numeric", "numeric",
            "factor", "numeric", "factor", "numeric", "factor", "numeric")
heart = read.csv("heart.csv", colClasses = classes)

#label and data split
X = heart[,-12]
y = heart[,12]
y = sapply(y, function(x){ifelse(x==1, "heart disease", "no heart disease")})
y = as.factor(y)

#train test split
ind = sample(length(y), floor(0.7*length(y)))

X_train = X[ind,]
y_train = y[ind]

X_test = X[-ind,]
y_test = y[-ind]

#Naive Bayes
NB = naiveBayes(X_train, y_train, laplace = 1)

NB_pred = predict(NB, X_test, type = "class")

sum(NB_pred==y_test)/length(NB_pred)

nb_cm = as.data.frame(table(NB_pred, y_test))

#SVM
set.seed(2021)

X_train_svm = X_train %>% mutate_if(is.factor,as.numeric)
X_test_svm = X_test %>% mutate_if(is.factor,as.numeric)

kernals = c("linear", "polynomial", "radial", "sigmoid")


SVM = function(K){
  model = svm(X_train_svm, y_train, kernel = K)
  pred = predict(model, X_test_svm, type = "class")
  accuracy = sum(pred==y_test)/length(pred)
  return(accuracy)
}

preds = sapply(kernals, SVM)

preds = data.frame(Kernal = as.factor(names(preds)), Accuracy = as.numeric(preds))

preds

#radial kernel
svm_model = svm(X_train_svm, y_train, kernal = "radial")
svm_pred = predict(svm_model, X_test_svm, type = "class")

svm_cm = as.data.frame(table(svm_pred, y_test))

#plots
#svm kernals' accuracy
ggplot(preds, aes(x = Kernal, y = Accuracy, fill = Kernal)) +
         geom_bar(stat = "identity") +
         theme_minimal() +
         scale_fill_brewer(palette="Oranges")+
  geom_text(aes(label = round(Accuracy, 4)),
            vjust = 0, show.legend = FALSE)
ggsave("SVM_Kernals.png")

#svm confusion matrix
ggplot(data = svm_cm,
       mapping = aes(x = svm_pred,
                     y = y_test)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)),
            vjust = 1, show.legend = FALSE) +
  scale_fill_gradient(trans = "log", guide = "none") +
  ggtitle("Confucion Matrix of SVM Model Radial Kernel") +
  xlab("Predicted Lables") +
  ylab("True Labels")
ggsave("SVM_CM.png")

#Naive Bayes
ggplot(data = nb_cm,
       mapping = aes(x = NB_pred,
                     y = y_test)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(trans = "log", guide = "none") +
  ggtitle("Confusion Matrix of Naive Bayes Model") +
  xlab("Predicted Lables") +
  ylab("True Labels")
ggsave("NB_CM.png")

