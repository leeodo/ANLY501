
#import packages
import numpy as np
import pandas as pd #For data wragling
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import graphviz

from sklearn.metrics import accuracy_score#accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay#confusion matrix

import matplotlib.pyplot as plt

#%%
#load both train and test data 
data = pd.read_csv("SPAM.csv")
print(data.head())

#%%
#extract label column
y = data['Category'].to_numpy()

#extract corpus
X = data['Message'].to_numpy()

#%%
print(np.unique(y).reshape(-1,1))

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
TF = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
TFMx = TF.fit_transform(X)
TFMx.shape

#%%
#train test split
X_train, X_test, y_train, y_test = train_test_split(TFMx, y, test_size=0.3,
                                                    random_state=42)

#%%
clf = DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

#%%
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

#%%
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

#%%
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

#%%
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.savefig("accu_cp.png")
plt.show()

#%%
#First decision tree
clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.007)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

#%%
Tree_Object = tree.export_graphviz(clf,
                                   out_file=None,
                                   feature_names = TF.get_feature_names_out(), 
                                   class_names=clf.classes_,
                                   filled = True,
                                   rounded = True,
                                   special_characters=True)

graph = graphviz.Source(Tree_Object) 
    
graph.render("MyTree", format='png', view=True) 

#%%
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()

#%%
#Second decision tree
clf = DecisionTreeClassifier(criterion="entropy",random_state=15, ccp_alpha=0.007)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

#%%
Tree_Object = tree.export_graphviz(clf,
                                   out_file=None,
                                   feature_names = TF.get_feature_names_out(), 
                                   class_names=clf.classes_,
                                   filled = True,
                                   rounded = True,
                                   special_characters=True)

graph = graphviz.Source(Tree_Object) 
    
graph.render("MyTree1", format='png', view=True) 

#%%
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()

#%%
#Third decision tree
clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.01)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

#%%
Tree_Object = tree.export_graphviz(clf,
                                   out_file=None,
                                   feature_names = TF.get_feature_names_out(), 
                                   class_names=clf.classes_,
                                   filled = True,
                                   rounded = True,
                                   special_characters=True)

graph = graphviz.Source(Tree_Object) 
    
graph.render("MyTree2", format='png', view=True) 

#%%
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()