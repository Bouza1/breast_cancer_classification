![breast_cancer_title3](https://github.com/Bouza1/breast_cancer_classification/assets/97123953/8b0fea20-fe16-4b09-a8f8-1fa432dc0c40)

# Content
- Dataset Exploration
  - data_exploration.ipynb
- Current Machine Learning Workflow
  - CMLW_evaluation.ipynb
- Proposed Machine Learning Workflow
  - PMLW_evaluation.ipynb

# Dataset
References to the dataset refer to the dataset seen below. A detailed data analysis and exploration of the dataset can be seen in the data_exploration.ipynb. 

![image](https://github.com/Bouza1/breast_cancer_classification/assets/97123953/8bda2e0d-8fb8-4a27-8a4c-f72f892eea65)

The Dataset is an altered version of the more commonly known [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

# Current Machine Learning Workflow (CMLW)
The current machine learning work flow can be seen in its entirity below. With an evaluation found in the CMLW_evaluation.ipynb file.

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

![Blank diagram - Page 1 (3)](https://github.com/Bouza1/breast_cancer_classification/assets/97123953/2cb65a2c-409e-4ea6-9d5a-8773346d359d)
