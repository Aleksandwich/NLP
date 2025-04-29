from sklearn import datasets
wine = datasets.load_wine()
type(wine)


print(type(wine))

""" print(wine.DESCR)
print(wine.feature_names)

print(wine.target_names) """


import pandas as pd

df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
df["target"] = wine.target
df.head()


import polars as pl

df = pl.DataFrame(
    data=wine.data, schema=wine.feature_names).with_columns(
    target=pl.Series(wine.target)
)
df.head()

X_wine, y_wine = wine.data, wine.target

X_wine.shape

y_wine


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.3)
y_train
import matplotlib.pyplot as plt
print(X_wine.shape)
print(y_wine)
print(X_train.shape)
print(y_train.shape)

plt.hist(y_train, align="right", label="train") 
plt.hist(y_test, align="left", label="test")
plt.legend()
plt.xlabel("Classe")
plt.ylabel("Nombre d'exemples")
plt.title("Répartition des classes") 
plt.show()


from sklearn.svm import LinearSVC
clf = LinearSVC(dual=True) # initialize the classifier, costructeur
clf.fit(X_train, y_train)

clf.predict(X_test)
print(clf.score(X_test, y_test))




from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))



print("\nAperçu du DataFrame (Polars):")
print(df.head())
print(f"\nDimensions du DataFrame: {df.shape}")