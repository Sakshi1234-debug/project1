#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from tabulate import tabulate
import warnings                                       # Remove unnecessary warnings for better interpretability 
warnings.filterwarnings('ignore')


# In[2]:


traindata_df = pd.read_excel('train.xlsx')
testdata_df = pd.read_excel('test.xlsx')


# In[3]:


traindata_df.head()


# In[4]:


testdata_df.head()


# In[5]:


traindata_df.info()


# In[6]:


traindata_df = traindata_df.drop(['previous','profit','id'], axis = 1)
traindata_df.head()


# In[7]:


testdata_df = testdata_df.drop(['previous','id'], axis = 1)
testdata_df.head()


# In[8]:


# profit column had 2 rows at the end which had null values for all columns. Removing them
traindata_df = traindata_df.drop(index = traindata_df.index[-2:]) 


# In[9]:


traindata_df.describe()


# In[10]:


traindata_df2 = traindata_df.copy()
x = traindata_df['custAge'].median()
traindata_df2['custAge'].fillna(x, inplace = True)
traindata_df=traindata_df2
traindata_df.info()


# In[11]:


c= traindata_df['schooling'].value_counts(dropna = False)
c


# In[12]:


traindata_df3 = traindata_df.copy()
x = traindata_df['schooling'].mode()[0]
traindata_df3['schooling'].fillna(x, inplace = True)
traindata_df=traindata_df3
traindata_df.head()


# In[13]:


traindata_df4 = traindata_df.copy()
traindata_df4['day_of_week'].fillna("unknown", inplace = True)
traindata_df=traindata_df4
traindata_df.head()


# In[14]:


traindata_df.info()


# In[15]:


testdata_df.describe()


# In[16]:


testdata_df2 = testdata_df.copy()
x = testdata_df['custAge'].median()
testdata_df2['custAge'].fillna(x, inplace = True)
testdata_df=testdata_df2
testdata_df.info()


# In[17]:


c1 = testdata_df['schooling'].value_counts(dropna = False)
c1


# In[18]:


testdata_df3 = testdata_df.copy()
x = testdata_df['schooling'].mode()[0]
testdata_df3['schooling'].fillna(x, inplace = True)
testdata_df=testdata_df3
testdata_df.info()


# In[19]:


testdata_df4 = testdata_df.copy()
testdata_df4['day_of_week'].fillna("unknown", inplace = True)
testdata_df=testdata_df4
testdata_df.info()


# In[20]:


report = sv.analyze(traindata_df, target_feat='responded')


# In[21]:


report.show_notebook()


# In[22]:



# Save report to an HTML file
report.show_html("sweetviz_report.html", open_browser=True)


# In[23]:


traindata_df5 = traindata_df.copy()
traindata_df5['responded'] = np.where(traindata_df5['responded'] == 'yes',1,0)
traindata_df = traindata_df5
traindata_df.info()


# In[24]:


pip install dataprep


# In[25]:



from dataprep.eda import create_report




report = create_report(traindata_df)


# Generate a comprehensive report
create_report(traindata_df).show_browser() # Display report in the browser
report.save('dataprep_report.html')


# In[26]:



from dataprep.eda import create_report
from dataprep.eda import plot


# Create a sample DataFrame for demonstration



# Plot visualizations for specific features against the target variable
# This helps in understanding the relationship between features and the target
plot(traindata_df, 'custAge', 'responded')# Visualize 'age' in relation to 'responded'



# In[27]:


plot(traindata_df, 'loan', 'responded')  # Visualize 'salary' in relation to 'responded'


# In[28]:


plot(traindata_df, 'marital', 'responded')  # Visualize 'gender' in relation to 'responded'


# In[29]:


train_filter = traindata_df[['custAge','campaign','pdays','emp.var.rate','cons.price.idx','cons.conf.idx',
                             'euribor3m','nr.employed','pmonths','pastEmail','responded']]
plt.figure(figsize = (15, 9))
heatmap = sns.heatmap(train_filter.corr(), vmin = -1, vmax = 1, annot = True)
heatmap.set_title('Correlation Heatmap', fontdict = {'fontsize' : 12}, pad = 12)


# In[30]:


traindata_df = traindata_df.drop(['pmonths'], axis = 1)
traindata_df.columns


# In[31]:


traindata_df['pdays'] = traindata_df['pdays'].apply(lambda x: -1 if x == 999 else x)
traindata_df.head()


# In[32]:


report = sv.analyze(traindata_df, target_feat='responded')


# In[33]:


report.show_notebook()


# In[34]:


traindata_df7 = traindata_df.copy()
traindata_df = traindata_df7.drop(['campaign','housing','loan'], axis = 1)
traindata_df = traindata_df.drop(['month','day_of_week'], axis = 1)
traindata_df


# In[35]:


schooling_enc = traindata_df['schooling'].unique()
profession_enc = traindata_df['profession'].unique()
marital_enc = traindata_df['marital'].unique()
default_enc = traindata_df['default'].unique()
contact_enc = traindata_df['contact'].unique()
poutcome_enc = traindata_df['poutcome'].unique()
categories = [profession_enc, marital_enc, schooling_enc, default_enc, contact_enc, poutcome_enc]
categories


# In[36]:


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
        
traindata_df_categorical = traindata_df[['profession','marital','schooling','default','contact','poutcome']]
traindata_df_categorical = onehot_encoder.fit_transform(traindata_df_categorical)
traindata_df_categorical = pd.DataFrame(traindata_df_categorical, columns=onehot_encoder.get_feature_names_out())
traindata_df_categorical


# In[37]:


traindata_df=pd.concat([traindata_df,traindata_df_categorical],axis=1)
traindata_df=traindata_df.drop(['profession','marital','schooling','default','contact','poutcome'],axis=1)


# In[38]:


traindata_df.columns


# In[39]:


testdata_df.columns


# In[40]:


x = traindata_df.drop(['responded'],axis=1).values
y = traindata_df['responded'].values


# In[41]:


x


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 27, 
                                                    shuffle = True, stratify = y)


# In[43]:


x_train


# In[44]:


from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
x_train[:,:9]= SC.fit_transform(x_train[:,:9])
x_test[:,:9]= SC.transform(x_test[:,:9])
x_test


# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
model = linear_model.LogisticRegression(fit_intercept = True, max_iter = 1000)
       
parameters = {'class_weight': [None, 'balanced'], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        
grid_search = GridSearchCV(model, parameters, scoring = "f1")
grid_search.fit(x_train, y_train)


# In[46]:


best_model = grid_search.best_estimator_
        
#best_model        
best_model.fit(x_train, y_train)
      


# In[47]:


y_pred=best_model.predict(x_test)
# y_pred = grid_search.predict(x_test)


# In[48]:


from sklearn.metrics import f1_score
f1_score(y_test,y_pred)


# In[49]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred)


# In[50]:


from sklearn.metrics import accuracy_score
y_pr = best_model.predict(x_test)
accuracy_score(y_test, y_pr)


# In[51]:


# Rearranging colunms to bring target column to the last
traindata_df_dt = traindata_df[['custAge', 'pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
       'euribor3m', 'nr.employed', 'pastEmail',
       'profession_admin.', 'profession_services', 'profession_blue-collar',
       'profession_entrepreneur', 'profession_technician',
       'profession_retired', 'profession_housemaid', 'profession_student',
       'profession_unknown', 'profession_unemployed',
       'profession_self-employed', 'profession_management', 'marital_single',
       'marital_divorced', 'marital_married', 'marital_unknown',
       'schooling_university.degree', 'schooling_high.school',
       'schooling_professional.course', 'schooling_basic.4y',
       'schooling_unknown', 'schooling_basic.9y', 'schooling_basic.6y',
       'schooling_illiterate', 'default_no', 'default_unknown', 'default_yes',
       'contact_cellular', 'contact_telephone', 'poutcome_nonexistent',
       'poutcome_failure', 'poutcome_success','responded']]


# In[52]:


xdt = traindata_df_dt.drop(['responded'], axis = 1)
ydt = pd.DataFrame(traindata_df_dt['responded'])
x_train, x_test, y_train, y_test = train_test_split(xdt, ydt, test_size = 0.2, random_state = 27, shuffle = True, stratify = y)
x_train.iloc[:,:8]= SC.fit_transform(x_train.iloc[:,:8])
x_test.iloc[:,:8]= SC.fit_transform(x_test.iloc[:,:8])


# In[53]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(x_train,y_train)


# In[54]:


y_train['responded']=y_train['responded'].astype("string")
y_train.info()


# In[55]:


from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
plt.figure(figsize = (12,6))
plot_tree(dt, feature_names = x_train.columns,
class_names = y_train['responded'].unique(),filled = True,rounded = True); 


# In[56]:


y_pred = dt.predict_proba(x_train)[:,1]
roc_auc_score(y_train,y_pred)


# In[57]:


y_pred1 = dt.predict_proba(x_test)[:,1]
roc_auc_score(y_test,y_pred1)


# In[58]:


res_pred = dt.predict(x_test)
score = accuracy_score(y_test, res_pred)
score


# In[59]:


f1_score(y_test,res_pred)


# In[60]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10,max_depth = 6,oob_score = True)
rf.fit(x_train,y_train)
rf.oob_score_


# In[61]:


y_pred2 = rf.predict_proba(x_train)[:,1]
roc_auc_score(y_train,y_pred2)


# In[62]:


y_pred3 = rf.predict_proba(x_test)[:,1]
roc_auc_score(y_test,y_pred3)


# In[63]:


y_predict = rf.predict(x_train)
accuracy_score(y_train, y_predict)


# In[64]:


type(y_predict)


# In[65]:


from sklearn.metrics import accuracy_score
import pandas as pd

# Assume y_test is a DataFrame with a single column
# Extract the column to get a Pandas Series
y_test_series = y_test.iloc[:, 0]

# Predicted labels (assume y_predict2 is a NumPy array or similar)
y_predict2 = rf.predict(x_test)

# Ensure both are of the same data type
if y_test_series.dtype != y_predict2.dtype:
    # Convert y_predict2 to the data type of y_test_series
    y_predict2 = y_predict2.astype(y_test_series.dtype)

# Calculate the accuracy
accuracy = accuracy_score(y_test_series, y_predict2)
print("Accuracy:", accuracy)


# In[66]:


f1_score(y_test,y_predict2)


# In[67]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(classification_report(y_test, y_predict2))


# In[68]:


from sklearn.metrics import RocCurveDisplay
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rf, x_train, y_train, ax=ax, alpha=0.8)
rfc_disp1 = RocCurveDisplay.from_estimator(rf, x_test, y_test, ax=ax, alpha=0.8)
plt.plot([0,1],[0,1],'r--')
plt.title('ROC Curve')
plt.show()


# In[69]:


# Check data types for x_test, y_test, and the Random Forest predictions
print("x_test type:", type(x_test))
print("y_test type:", type(y_test))
print("Prediction probabilities type:", type(rf.predict_proba(x_test)))


# In[70]:


# Assuming y_test is a single-column DataFrame
y_test1 = y_test.squeeze()

# Ensure it's now a Pandas Series
print("y_test1 type:", type(y_test1))


# In[71]:


# Convert to a NumPy array
y_test1 = y_test1.values.flatten()

# Ensure it's a NumPy array
print("y_test1 type:", type(y_test1))


# In[72]:


# Test the output of Random Forest predictions
pred_probs = rf.predict_proba(x_test)

# Validate the shape and content of the prediction probabilities
print("Prediction probabilities:", pred_probs)


# In[73]:


x_test1 = x_test.squeeze()

# Ensure it's now a Pandas Series
print("x_test1 type:", type(x_test1))


# In[74]:


# Convert to a NumPy array
x_test1 = x_test1.values.flatten()

# Ensure it's a NumPy array
print("x_test1 type:", type(x_test1))


# In[75]:


(x_test1.shape),(y_test1.shape)


# In[76]:


x_test1_2d = x_test1.reshape(-1, 1)  # Converts 1D to 2D with one feature per row


# In[77]:


y_test1_2d = y_test1.reshape(-1, 1)  # Converts 1D to 2D with one label per row


# In[78]:


import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

# Create the plotting axis
ax = plt.gca()

# Plot ROC curves from the estimator
rfc_disp = RocCurveDisplay.from_estimator(rf, x_train, y_train, ax=ax, alpha=0.8)
rfc_disp1 = RocCurveDisplay.from_estimator(rf, x_test1, y_test1, ax=ax, alpha=0.8)

# Plot reference diagonal and add titles
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve')
plt.show()


# In[79]:


y_test = np.array(y_test)  


# In[80]:


type(y_test)


# In[81]:


rffit = rf.fit(x_train,y_train)
imp_df = pd.DataFrame({
    "Varname": x_train.columns,
    "Imp": rffit.feature_importances_
})

imp_df.sort_values(by="Imp", ascending=False)


# In[82]:


get_ipython().system('pip install xgboost')
import xgboost as xgb


# In[83]:


non_numeric_columns = x_train.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)


# In[84]:


# Check data types in x_train
print(x_train.dtypes)


# In[85]:


# Find missing values
missing_values = x_train.isnull().sum()
print("Missing values:", missing_values)

# Fill or drop missing values if needed
# x_train.fillna(value=0, inplace=True)  # Example of filling with zeros


# In[86]:


# Check if y_train is numeric
if not isinstance(y_train, (pd.Series, np.ndarray)):
    raise ValueError("y_train must be a Pandas Series or NumPy array.")

# Check data types in y_train
print("y_train dtype:", y_train.dtype)

# Convert to numeric if needed
if y_train.dtype == 'object':
    y_train = y_train.astype('int')  # Convert object types to integers


# In[87]:


y_train = np.array(y_train)


# In[88]:


import xgboost as xgb

# Initialize DMatrix with corrected data
train_dm = xgb.DMatrix(x_train, y_train)

# XGBoost parameters
xgb_params = {
    'eta': 0.3,
    'max_depth': 5,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 1,
    'silent': 1
}

# Perform cross-validation
cv_results = xgb.cv(xgb_params, train_dm, num_boost_round=500, nfold=5, metrics={'auc'}, seed=123)
print(cv_results)


# In[89]:


test_dm = xgb.DMatrix(x_test, y_test, feature_names=list(x_test.columns))
watchlist = [(train_dm, 'train')]
xgb_model =  xgb.train(xgb_params,train_dm,
                         num_boost_round = 100, 
                         evals = watchlist, 
                         verbose_eval = 10)
y_pred4=xgb_model.predict(test_dm)


# In[90]:


roc_auc_score(y_test,y_pred4)


# In[91]:


y_predict2 = rf.predict(x_test)
accuracy_score(y_test, y_predict2)


# In[ ]:


print(classification_report(y_test, y_predict2))


# In[ ]:


x_train.columns


# In[ ]:


testdata_df.columns


# In[ ]:


traindata_df.columns


# In[ ]:


testdata_df=testdata_df.drop(['housing','loan','month', 'day_of_week','campaign','pmonths'],axis=1)


# In[92]:


testdata_df.columns


# In[93]:


schooling_enc1 = testdata_df['schooling'].unique()
profession_enc1 = testdata_df['profession'].unique()
marital_enc1 = testdata_df['marital'].unique()
default_enc1 = testdata_df['default'].unique()
contact_enc1 = testdata_df['contact'].unique()
poutcome_enc1 = testdata_df['poutcome'].unique()
categories1 = [profession_enc1, marital_enc1, schooling_enc1, default_enc1, contact_enc1, poutcome_enc1]
categories1


# In[94]:


onehot_encoder = OneHotEncoder(sparse=False, categories=categories1)
        

testdata_df_categorical = testdata_df[['profession','marital','schooling','default','contact','poutcome']]
testdata_df_categorical = onehot_encoder.fit_transform(testdata_df_categorical)
testdata_df_categorical = pd.DataFrame(testdata_df_categorical, columns=onehot_encoder.get_feature_names_out())
testdata_df_categorical


# In[95]:


testdata_df=pd.concat([testdata_df,testdata_df_categorical],axis=1)
testdata_df=testdata_df.drop(['profession','marital','schooling','default','contact','poutcome'],axis=1)


# In[96]:


testdata_df.columns


# In[97]:


testdata_df


# In[98]:


testdata_df= testdata_df[['custAge', 'pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
       'euribor3m', 'nr.employed', 'pastEmail', 'profession_admin.',
       'profession_services', 'profession_blue-collar',
       'profession_entrepreneur', 'profession_technician',
       'profession_retired', 'profession_housemaid', 'profession_student',
       'profession_unknown', 'profession_unemployed',
       'profession_self-employed', 'profession_management', 'marital_single',
       'marital_divorced', 'marital_married', 'marital_unknown',
       'schooling_university.degree', 'schooling_high.school',
       'schooling_professional.course', 'schooling_basic.4y',
       'schooling_unknown', 'schooling_basic.9y', 'schooling_basic.6y',
       'schooling_illiterate', 'default_no', 'default_unknown', 'default_yes',
       'contact_cellular', 'contact_telephone', 'poutcome_nonexistent',
       'poutcome_failure', 'poutcome_success']]
testdata_df


# In[99]:


rffit = rf.fit(x_train,y_train)
testdata_df['propensity'] = rf.predict_proba(testdata_df)[:,1]
print(testdata_df.head())


# In[100]:


testdata_df1 = testdata_df.copy()
testdata_df1['propensity'] = np.where(testdata_df1['propensity'] >= 0.5,1,0)
testdata_df1


# In[101]:


testdata_df1.to_csv("testingCandidate.csv")


# In[104]:


train_ts = pd.read_excel('train.xlsx')
train_ts


# In[105]:


t = pd.read_csv('testingCandidate.csv')


# In[106]:


t.head()


# In[107]:


import pickle
pickle.dump(rf,open('rf.pkl','wb'))
rf=pickle.load(open('rf.pkl','rb'))


# In[ ]:




