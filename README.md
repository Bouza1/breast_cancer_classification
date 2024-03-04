![breast_cancer_title3](https://github.com/Bouza1/breast_cancer_classification/assets/97123953/8b0fea20-fe16-4b09-a8f8-1fa432dc0c40)

# Content
- Dataset Exploration
  - data_exploration.ipynb
- Current Machine Learning Workflow
  - CMLW_evaluation.ipynb
- Proposed Machine Learning Workflow
  - PMLW_evaluation.ipynb

# Overview
The objective of this repository is to conduct an in-depth exploration of the dataset, seen below, and offer a comprehensive assessment of the current machine learning workflow (CMLW). Subsequently, a refined machine learning workflow is proposed (PMLW), tailored to rectify the shortcomings of the CMLW, while aligning with the characteristics of the dataset uncovered during the exploration.

# Dataset
References to the dataset refer to the dataset seen below. A detailed data analysis and exploration of the dataset can be seen in the data_exploration.ipynb. 

![image](https://github.com/Bouza1/breast_cancer_classification/assets/97123953/8bda2e0d-8fb8-4a27-8a4c-f72f892eea65)

The Dataset is an altered version of the more commonly known [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

# Current Machine Learning Workflow (CMLW)
The current machine learning work flow can be seen in its entirety below. With an evaluation found in the CMLW_evaluation.ipynb file.

``` python
df = pd.read_csv('data/breast-cancer.csv')

df = df.dropna()

cat_columns = df.select_dtypes(['object']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.astype('category'))
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

X = df.iloc[:,:len(df.columns)-1]
y = df.iloc[:,len(df.columns)-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=None)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Printing out performance of the model
print("Accuracy: %s" % (metrics.accuracy_score(y_test, y_pred)*100))                                                                         
```

# Proposed Machine Learning Workflow (PMLW)
An overview of the PMLW can be seen below, with a full implementation and evaluation found in the PMLW_evaluation.ipynb file.

![Blank diagram](https://github.com/Bouza1/breast_cancer_classification/assets/97123953/34625376-fdac-4e35-8ea5-bd6d33a791b9)

# References

- Singh, D. and Singh, B. (2020) ‘Investigating the impact of data normalization on classification 
performance’, Applied Soft Computing, 97, p. 105524. doi:10.1016/j.asoc.2019.105524.

- Wu, O. 2023. Rethinking Class Imbalance in Machine Learning. arXiv preprint arXiv:2305.03900.

