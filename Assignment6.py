#import packages
import numpy as np
import pandas as pd #For data wragling
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

from sklearn.naive_bayes import MultinomialNB#Multinomial Naive Bayes
from sklearn.svm import LinearSVC#linear kernal SVM
from sklearn.metrics import accuracy_score#accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay#confusion matrix

import matplotlib.pyplot as plt

#%%
#reading data
data = pd.read_csv("SPAM.csv")

#%%
#extract label column
y = data['Category'].to_numpy()

#extract corpus
X = data['Message'].to_numpy()

#%%
#vectorize the text data and remove the stop words
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

#%%
#Using the stemmer
TFs = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
TFMxs = TFs.fit_transform(X)
TFMxs.shape

#%%
#Not using the stemmer
TF = TfidfVectorizer(stop_words='english')
TFMx = TF.fit_transform(X)
TFMx.shape

#%%
#train test split
X_train, X_test, y_train, y_test = train_test_split(TFMx, y, test_size=0.3, random_state=42)

X_trains, X_tests, y_trains, y_tests = train_test_split(TFMxs, y, test_size=0.3, random_state=42)

#%%
############Naive Bayes############
#Fit the model with stemmer
clfrs = MultinomialNB()
clfrs.fit(X_trains, y_trains)

#%%
#Fit the model without stemmer
clfr = MultinomialNB()
clfr.fit(X_train, y_train)

#%%
#make prediction without stemmer
y_pred = clfr.predict(X_test)
#Accuracy score
print(accuracy_score(y_test, y_pred))

#%%
#make prediction with stemmer
y_preds = clfrs.predict(X_tests)
#Accuracy score
print(accuracy_score(y_tests, y_preds))

#%%
#confusion matrix
f, axes = plt.subplots(1, 2, figsize=(18, 8), sharey='row')
#without stemmer
cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clfr.classes_)

#withstemmer
cms = confusion_matrix(y_tests, y_preds, normalize='true')
disps = ConfusionMatrixDisplay(confusion_matrix=cms,
                              display_labels=clfr.classes_)

disp.plot(ax=axes[0])
disps.plot(ax=axes[1])

disp.ax_.set_title('Confusion Matrix Without Stemmer', fontsize=20)
disps.ax_.set_title('Confusion Matrix With Stemmer', fontsize = 20)
disp.ax_.set_xlabel('')
disps.ax_.set_xlabel('')
disp.ax_.set_ylabel('True Lable', fontsize=20)
disps.ax_.set_ylabel('')

f.text(0.43, 0.08, 'Predicted label', fontsize=25)

plt.savefig("NB_CMs_py.png")
plt.show()

#%%
###############SVM#################
#without stemmer
svmclf = LinearSVC()
svmclf.fit(X_train, y_train)
svm_pred = svmclf.predict(X_test)
print(accuracy_score(y_test, svm_pred))

#%%
#with stemmer
svmclfs = LinearSVC()
svmclfs.fit(X_trains, y_trains)
svm_preds = svmclfs.predict(X_tests)
print(accuracy_score(y_tests, svm_preds))

#%%
#confusion matrix
f, axes = plt.subplots(1, 2, figsize=(18, 8), sharey='row')
#without stemmer
cm = confusion_matrix(y_test, svm_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svmclf.classes_)

#withstemmer
cms = confusion_matrix(y_tests, svm_preds, normalize='true')
disps = ConfusionMatrixDisplay(confusion_matrix=cms,
                              display_labels=svmclfs.classes_)

disp.plot(ax=axes[0])
disps.plot(ax=axes[1])

disp.ax_.set_title('Confusion Matrix Without Stemmer', fontsize=20)
disps.ax_.set_title('Confusion Matrix With Stemmer', fontsize = 20)
disp.ax_.set_xlabel('')
disps.ax_.set_xlabel('')
disp.ax_.set_ylabel('True Lable', fontsize=20)
disps.ax_.set_ylabel('')

f.text(0.43, 0.08, 'Predicted label', fontsize=25)
plt.savefig("SVM_CMs_py.png")
plt.show()
