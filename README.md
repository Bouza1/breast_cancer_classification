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

- Acuña, E. and Rodriguez, C. (2004) ‘The treatment of missing values and its effect on classifier 
accuracy’, Classification, Clustering, and Data Mining Applications, pp. 639–647. 
doi:10.1007/978-3-642-17103-1_60. 

- Anderson, W.F. et al. (2010) ‘Male breast cancer: A population-based comparison with female 
breast cancer’, Journal of Clinical Oncology, 28(2), pp. 232–239. 
doi:10.1200/jco.2009.23.8162. 

- Bin Rafiq, R. et al. (2020) ‘Validation methods to promote real-world applicability of machine 
learning in medicine’, 2020 3rd International Conference on Digital Medicine and Image 
Processing [Preprint]. doi:10.1145/3441369.3441372. 

- Chawla, N.V. et al. (2002) ‘Smote: Synthetic minority over-sampling technique’, Journal of 
Artificial Intelligence Research, 16, pp. 321–357. doi:10.1613/jair.953. 

- Gu, Q. et al. (2008) ‘Data mining on imbalanced data sets’, 2008 International Conference on 
Advanced Computer Theory and Engineering [Preprint]. doi:10.1109/icacte.2008.26. 

- Jadhav, A., Pramod, D. and Ramanathan, K. (2019) ‘Comparison of performance of data imputation 
methods for numeric dataset’, Applied Artificial Intelligence, 33(10), pp. 913–933. 
doi:10.1080/08839514.2019.1637138. 

- Rahman, M.M., and Davis, D.N. (2013) ‘Addressing the class imbalance problem in medical 
datasets’, International Journal of Machine Learning and Computing, pp. 224–228. 
doi:10.7763/ijmlc.2013.v3.307. 

- Singh, D. and Singh, B. (2020) ‘Investigating the impact of data normalization on classification 
performance’, Applied Soft Computing, 97, p. 105524. doi:10.1016/j.asoc.2019.105524. 

- Singh, N. and Singh, P. (2021) ‘Exploring the effect of normalization on Medical Data 
Classification’, 2021 International Conference on Artificial Intelligence and Machine Vision 
(AIMV) [Preprint]. doi:10.1109/aimv53313.2021.9670938. 

- Sultana, J. and Khader Jilani, A. (2018) ‘Predicting breast cancer using logistic regression and multiclass classifiers’, International Journal of Engineering &amp; Technology, 7(4.20), p. 22. 
doi:10.14419/ijet.v7i4.20.22115. 

- Wu, O. 2023. Rethinking Class Imbalance in Machine Learning. arXiv preprint arXiv:2305.03900.
