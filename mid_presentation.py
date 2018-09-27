
# coding: utf-8

# ## US Personal Credit Loan Data Default Prediction
# 
# ### Course Project: Project in Data Science COMSE6998, Fall 2017

# #### code running under python3.x version
# author:yang

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import StratifiedShuffleSplit,StratifiedKFold,train_test_split
from scipy import stats
from sklearn import decomposition,linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier,Lasso,SGDClassifier,LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,hamming_loss
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords,wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import os
from collections import Counter
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().magic('matplotlib inline')
#using matplotlib’s ggplot style
plt.style.use('ggplot')
import seaborn as sns
sns.set(color_codes=True)


# ## Download Personal Loan Data
# #### [Personal Loan Data](https://www.kaggle.com/wendykan/lending-club-loan-data/data)
# 
# #### Code is running under python 3.x version

# In[2]:


#load dataset
thepath ='/Users/gyang/Desktop/ProjectDataScience/final/'

loandata =pd.read_csv(thepath +'loan.csv')


# In[3]:


#fully load original dataset
pd.set_option('display.max_columns',None)
loandata.head()


# In[4]:


loandata.shape


# ## Data Cleaning

# In[5]:


loandata.dtypes


# In[6]:


loandata2 =loandata.drop(['id','member_id','emp_title','url','zip_code','desc','issue_d',
                         'earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d'], axis=1, inplace=False)


# In[7]:


loandata2.head()


# In[8]:


#reorder grade variable
loandata2['grade'] = loandata2['grade'].astype('category')
loandata2['grade'].cat.reorder_categories(['A','B','C','D','E','F','G'], inplace=True)

#reorder emp_length
loandata2['emp_length'] = loandata2['emp_length'].astype('category')
loandata2['emp_length'].cat.reorder_categories(["< 1 year", "1 year","2 years","3 years","4 years",
                                                "5 years","6 years","7 years","8 years","9 years",
                                                "10+ years","n/a"], inplace=True)


# In[9]:


#target label
loandata2['loan_status'].value_counts()


# In[10]:


ax1 =loandata2.groupby('loan_status').size().plot(kind='bar',color ='orange',rot=45,title="Loan Status Distribution")
ax1.set_xlabel("Count")
ax1.set_ylabel("Loan Status")
ax1.grid(True)


# In[11]:


def loanStat(x):
    if x in ['Current','Fully Paid','In Grace Period']:
        return 0
    else:
        return 1    


# In[12]:


loandata2['default'] =loandata2['loan_status'].apply(lambda x :loanStat(x))
loandata2['default'] = loandata2['default'].astype('category')


# In[13]:


loandata2['default'].value_counts()


# In[14]:


#check missing value
#categorical features
miss_category = loandata2.select_dtypes(include=['object','category'])
miss_cate_rate = miss_category.isnull().sum(axis =0)/float(len(miss_category))
miss_cate_rate.sort_values(ascending=False)


# In[15]:


loandata3 =loandata2.drop(miss_cate_rate[miss_cate_rate >0.5].index, axis=1, inplace=False) 


# In[16]:


# check numerical features
miss_numerical = loandata3.select_dtypes(exclude=['object','category'])
miss_num_rate = miss_numerical.isnull().sum(axis =0)/float(len(miss_numerical))
miss_num_rate.sort_values(ascending=False)


# In[17]:


loandata4 =loandata3.drop(miss_num_rate[miss_num_rate >0.5].index, axis=1, inplace=False) 


# In[18]:


#lower case
loandata4['title'] =loandata4['title'].str.lower()


# In[19]:


def loantitle(x):
    if x in ['business']: #business
        return 1
    elif x in ['other']: #other
        return 3
    else:
        return 2 #personal


# In[20]:


loandata4['loan_title'] =loandata4['title'].apply(lambda x :loantitle(x))
loandata4['loan_title'] = loandata4['loan_title'].astype('category')


# In[21]:


#drop title feature
loandata5 =loandata4.drop(['title','loan_status'],1)


# In[22]:


loandata5.head()


# In[25]:


#export to csv
loandata5.to_csv(thepath +'loandata5.csv',index=False)


# ## Data Visualization

# In[106]:


sns.countplot(x="default", data=loandata5)


# In[24]:


loandata2['emp_length'].value_counts()


# In[25]:


sns.countplot(x="emp_length", data=loandata5,order=["< 1 year", "1 year","2 years","3 years","4 years",
                                                   "5 years","6 years","7 years","8 years","9 years"
                                                   "10+ years","n/a"])
plt.xticks(rotation=45)


# In[26]:


sns.set()
_ =plt.hist(loandata5['loan_amnt'],edgecolor='black',bins =30)
_ =plt.xlabel('Loan Amount')
_ =plt.ylabel('Counts')
_ =plt.title('Loan Amount Histogram Distribution')
plt.show()


# In[27]:


sns.set()
_ =plt.hist(loandata5['int_rate'],edgecolor='black',bins =25)
_ =plt.xlabel('Interest Rate')
_ =plt.ylabel('Counts')
_ =plt.title('Interest Rate Histogram Distribution')
plt.show()


# In[28]:


sns.countplot(x="grade", data=loandata5)


# In[29]:


sns.boxplot(x="grade", y="loan_amnt", data=loandata5)


# In[30]:


#grade vs interest rate
sns.boxplot(x="grade", y="int_rate",hue="default", data=loandata5)


# In[31]:


#scatterplot loan amount VS annual income
sns.pairplot(x_vars=["loan_amnt"], y_vars=["annual_inc"], data=loandata5, hue="default", size=7)


# In[32]:


#scatterplot loan amount VS annual income
#grid labels
kws = dict(s=50, linewidth=0.5,edgecolor="orange")
g = sns.FacetGrid(loandata5, col="default", palette="Set1",size =7)
g = (g.map(plt.scatter, "loan_amnt", "annual_inc", **kws).add_legend())


# In[33]:


sns.pairplot(x_vars=["loan_amnt"], y_vars=["annual_inc"], data=loandata5, hue="grade", size=7)


# In[34]:


#grade VS Numbers of Open accounts
sns.violinplot(x="grade", y="open_acc", hue="default", data=loandata5, split=True,size =10)


# In[35]:


#Loan Purpose VS Loan Amount
sns.violinplot(x="loan_title", y="loan_amnt", hue="default", data=loandata5, split=True,size =15)
plt.title("Loan Amount by Purpose")
plt.xlabel("Loan Purpose")
plt.ylabel("Loan Amount")


# In[36]:


#density plot
#heatmap
#loan amount vs interest rate
#blue color - default =1
#red color - default =0

d0 = loandata5.loc[loandata5.default == 0]
d1 = loandata5.loc[loandata5.default == 1]
ax = sns.kdeplot(d0.loan_amnt, d0.int_rate,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(d1.loan_amnt, d1.int_rate,
                 cmap="Blues", shade=True, shade_lowest=False)


# In[37]:


#wordcloud Description Variable
text = loandata['desc'].dropna().str.lower()
token = [re.sub(r'[^a-zA-Z]+', ' ',token) for token in text]


# In[38]:


def filter_list(list_name):
    #remove stopwords
    stops = set(stopwords.words("english"))
    
    #convert to single word within list
    word_list= []
    for words in list_name:
        for word in words[:-4].split():
            word_list.append(word)
            
    filtered_words = [word for word in word_list if word not in stops]
    
    wnl = WordNetLemmatizer()
    
    filtered_words_1 =[]
    for r in filtered_words:
        filtered_words_1.append(wnl.lemmatize(r)) #change plurals nouns.
        
    filtered_words_2 =[]
    for r in filtered_words_1:
        filtered_words_2.append(wnl.lemmatize(r,'v')) #change to original tense(verb.)
    
    return filtered_words_2   


# In[39]:


new_token =filter_list(token)
new_str =','.join(new_token) #convert into string


# In[40]:


def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = int(360.0 * 21.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)

wordcloud = WordCloud(background_color='white',width=1200,
                      height=1000,max_words=100,color_func=random_color_func).generate(new_str)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[41]:


df = pd.DataFrame({'Total_LoanAmount' : loandata.groupby(['addr_state'])['loan_amnt'].sum(),
                  'Avg_Income':loandata.groupby(['addr_state'])['annual_inc'].mean(),
                  'Avg_InterestRate':loandata.groupby(['addr_state'])['int_rate'].mean(),
                  'Total_Payment':loandata.groupby(['addr_state'])['total_pymnt'].sum(),
                  'Avg_OpenAcc':loandata.groupby(['addr_state'])['total_acc'].mean()}).reset_index()


# In[42]:


df.head()


# In[43]:


sns.set()
# Top States on Total Loan Amount VS Average Income
df.sort_values(['Total_LoanAmount'], ascending=False, inplace=True)

f, ax = plt.subplots(nrows=1, ncols=2,figsize=(14, 15))

#add col 1 -- total loan
sns.set_color_codes("muted")
sns.barplot(x='Total_LoanAmount', y='addr_state', data=df,color="b",ax=ax[0])
#ax.set_xlabel('Total Loan Amount')
#ax.set_ylabel('States (Abv.)')
#ax.legend(ncol=2, loc="lower right", frameon=True)
ax[0].set(ylabel="",xlabel="Total Loan Amount (USD)")
sns.despine(left=True, bottom=True)

#add col 2 -- average income 
df.sort_values(['Avg_Income'], ascending=False, inplace=True)
sns.set_color_codes("muted")
sns.barplot(x='Avg_Income', y='addr_state', data=df,color="b",ax=ax[1])
ax[1].set(ylabel="",xlabel="Average Annual Income (USD)")
sns.despine(left=True, bottom=True)


# In[44]:


# Total Loan Amount -USD (Involved Current Payment Amount)

sns.set()
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(8, 15))

df.sort_values(['Total_LoanAmount'], ascending=False, inplace=True)

# Plot the total loan amount
sns.set_color_codes("pastel")
sns.barplot(x="Total_LoanAmount", y="addr_state", data=df,
            label="Total Loan Amount", color="b")

# Plot current total payment amount involved
sns.set_color_codes("muted")
sns.barplot(x="Total_Payment", y="addr_state", data=df,
            label="Current Total Payment Amount", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="",
       xlabel="Total Loan Amount -USD (Involved Current Payment Amount)")
sns.despine(left=True, bottom=True)


# In[45]:


import plotly
# register Poltly free account first,then check out your API keys
plotly.tools.set_credentials_file(username='yg2499', api_key='el7rt983crGV1cp4vHvr')


# In[46]:


for col in df.columns:
    df[col] = df[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],        [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

df['text'] = df['addr_state'] + '<br>'+            'Average Annual Income (USD): '+df['Avg_Income'] +'<br>'+            'Average Interest Rate (Percentage): '+df['Avg_InterestRate']+'<br>'+            'Total Payment Amount (USD): '+df['Total_Payment'] +'<br>'+            'Average Number of Open Account: '+df['Avg_OpenAcc']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df['addr_state'],
        z = df['Total_LoanAmount'].astype(float),
        locationmode = 'USA-states',
        text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Billions USD")
        ) ]

layout = dict(
        title = '2007 -2015 United States Personal Credit Overview by State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# ## Split Dataset
# 
# ### Part 1: Label -- Default Variable (Loan Status -0/1 Binary Type)

# In[182]:


# training data - 70% ,testing data -30%
sess = StratifiedShuffleSplit(loandata5['default'].values,test_size = 0.3)
for train_index,test_index in sess:
    trainData = loandata5.iloc[train_index]
    testData = loandata5.iloc[test_index]
    
X_train_1,y_train = trainData.drop(['default'],1) ,trainData['default']
X_test_1,y_test = testData.drop(['default'],1) ,testData['default']


# In[108]:


def cleandata(dt):
    #categorical features
    #one-hot-encode
    CONVERT_COLUMNS =list(dt.select_dtypes(include=['object','category']).columns)
    data_2 = pd.get_dummies(dt, columns =CONVERT_COLUMNS)
    
    #convert all NaN to Zero
    data_3 = data_2.replace(np.nan, 0)
    
    #numerical feature scaling
    scaler = MinMaxScaler()
    header_list = list(data_3.iloc[:,:30].columns)
    data_3[header_list] = scaler.fit_transform(data_3[header_list])
    
    return data_3


# In[139]:


X_train =cleandata(X_train_1)
X_test =cleandata(X_test_1)


# In[110]:


X_train.rename(columns={'emp_length_< 1 year': 'emp_length_less_1year', 
                        'emp_length_10+ years': 'emp_length_more_10years',
                       'emp_length_n/a': 'emp_length_na'}, inplace=True)
X_test.rename(columns={'emp_length_< 1 year': 'emp_length_less_1year', 
                        'emp_length_10+ years': 'emp_length_more_10years',
                       'emp_length_n/a': 'emp_length_na'}, inplace=True)


# In[111]:


X_train.head()


# In[112]:


X_train.shape


# In[113]:


X_test.shape


# In[114]:


loandata5['pymnt_plan'].value_counts()


# In[115]:


X_test['pymnt_plan_n'].value_counts()


# In[116]:


X_train['pymnt_plan_y'].value_counts()


# In[117]:


# Get missing columns in the training test
missing_cols = set(X_train.columns) - set(X_test.columns)
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X_train.columns]


# In[118]:


X_test.shape


# ## Dataset is ready to train ...

# In[119]:


# build models

lr = LogisticRegression() #logistic regression
rfc = RandomForestClassifier() #random forest
dtc = DecisionTreeClassifier(max_depth=30) #decision tree
abdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),learning_rate=1,algorithm="SAMME",n_estimators=300) #ABDT
xg =XGBClassifier() #XGBoost
nb =GaussianNB() #naive bayes


# In[120]:


nb.fit(X_train,y_train)


# In[121]:


xg.fit(X_train,y_train)


# In[122]:


lr.fit(X_train,y_train)
rfc.fit(X_train,y_train)
dtc.fit(X_train,y_train)
abdt.fit(X_train,y_train)


# In[123]:


# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


# In[124]:


# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
params['metric'] = 'auc'


# In[125]:


gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=40,)


# In[126]:


pred_gbm = gbm.predict(X_test,num_iteration=gbm.best_iteration)


# In[127]:


pred_lr = lr.predict_proba(X_test)
pred_rfc = rfc.predict_proba(X_test)
pred_dtc = dtc.predict_proba(X_test)
pred_abdt = abdt.predict_proba(X_test)
pred_xg = xg.predict_proba(X_test)
pred_nb =nb.predict_proba(X_test)


# In[128]:


p_lr, r_lr, t_lr = precision_recall_curve(y_test,pred_lr[:,1])
p_rfc, r_rfc, t_rfc = precision_recall_curve(y_test,pred_rfc[:,1])
p_dtc, r_dtc, t_dtc = precision_recall_curve(y_test,pred_dtc[:,1])
p_abdt, r_abdt, t_abdt = precision_recall_curve(y_test,pred_abdt[:,1])
p_xgb, r_xgb, t_xgb = precision_recall_curve(y_test,pred_xg[:,1])
p_gbm, r_gbm, t_gbm = precision_recall_curve(y_test,pred_gbm)
p_nb, r_nb, t_nb = precision_recall_curve(y_test,pred_nb[:,1])

plt.plot(r_lr,p_lr,label='LR')
plt.plot(r_rfc,p_rfc,label='Random Forest')
plt.plot(r_dtc,p_dtc,label='Decision Tree')
plt.plot(r_abdt,p_abdt,label='AdaBoost DT')
plt.plot(r_xgb,p_xgb,label='XGBoost')
plt.plot(r_gbm,p_gbm,label='LightGBM')
plt.plot(r_nb,p_nb,label='Naive Bayes')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

print("Logistic Regression AUC:{0} ".format(roc_auc_score(y_test,pred_lr[:,1])))
print("RandomForest AUC:{0} ".format(roc_auc_score(y_test,pred_rfc[:,1])))
print("Decision Tree AUC:{0} ".format(roc_auc_score(y_test,pred_dtc[:,1])))
print("Ada Boost Decision Tree AUC:{0} ".format(roc_auc_score(y_test,pred_abdt[:,1])))
print("XGBoost AUC:{0} ".format(roc_auc_score(y_test,pred_xg[:,1])))
print("LightGBM AUC:{0} ".format(roc_auc_score(y_test,pred_gbm)))
print("Naive Bayes AUC:{0} ".format(roc_auc_score(y_test,pred_nb[:,1])))


# In[129]:


# Determine the false positive and true positive rates
fpr_lr, tpr_lr, _lr = roc_curve(y_test,pred_lr[:,1])
fpr_rfc, tpr_rfc, _rfc = roc_curve(y_test,pred_rfc[:,1])
fpr_dtc, tpr_dtc, _dtc = roc_curve(y_test,pred_dtc[:,1])
fpr_abdt, tpr_abdt, _abdt = roc_curve(y_test,pred_abdt[:,1])
fpr_xgb, tpr_xgb, _xgb = roc_curve(y_test,pred_xg[:,1])
fpr_gbm, tpr_gbm, _gbm = roc_curve(y_test,pred_gbm)
fpr_nb, tpr_nb, _nb = roc_curve(y_test,pred_nb[:,1])


# In[130]:


# Calculate the AUC
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_rfc = auc(fpr_rfc, tpr_rfc)
roc_auc_dtc = auc(fpr_dtc, tpr_dtc)
roc_auc_abdt = auc(fpr_abdt, tpr_abdt)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
roc_auc_gbm = auc(fpr_gbm, tpr_gbm)
roc_auc_nb = auc(fpr_nb, tpr_nb)

print('Logistic Regression ROC AUC: %0.4f' % roc_auc_lr)
print('Random Forest ROC AUC: %0.4f' % roc_auc_rfc) 
print('Decision Tree ROC AUC: %0.4f' % roc_auc_dtc) 
print('AdaBoost Decision Tree ROC AUC: %0.4f' % roc_auc_abdt) 
print('XGBoost ROC AUC: %0.4f' % roc_auc_xgb) 
print('LightGBM ROC AUC: %0.4f' % roc_auc_gbm)
print('Naive Bayes ROC AUC: %0.4f' % roc_auc_nb)


# In[131]:


# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr_lr, tpr_lr, label='LR ROC curve (area = %0.4f)' % roc_auc_lr)
plt.plot(fpr_rfc, tpr_rfc, label='Random Forest ROC curve (area = %0.4f)' % roc_auc_rfc)
plt.plot(fpr_dtc, tpr_dtc, label='Decision Tree ROC curve (area = %0.4f)' % roc_auc_dtc)
plt.plot(fpr_abdt, tpr_abdt, label='AdaBoost DT ROC curve (area = %0.4f)' % roc_auc_abdt)
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost ROC curve (area = %0.4f)' % roc_auc_xgb)
plt.plot(fpr_gbm, tpr_gbm, label='LightGBM ROC curve (area = %0.4f)' % roc_auc_gbm)
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes ROC curve (area = %0.4f)' % roc_auc_nb)

#plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Compare with Different Models')
plt.legend(loc="lower right")
plt.show()


# In[133]:


print('Logistic Regression -- The RMSE of prediction is:', mean_squared_error(y_test, pred_lr[:,1]) ** 0.5)
print('Random Forest -- The RMSE of prediction is:', mean_squared_error(y_test, pred_rfc[:,1]) ** 0.5)
print('Decision Tree -- The RMSE of prediction is:', mean_squared_error(y_test, pred_dtc[:,1]) ** 0.5)
print('AdBoost Decision Tree -- The RMSE of prediction is:', mean_squared_error(y_test, pred_abdt[:,1]) ** 0.5)
print('XGBoost -- The RMSE of prediction is:', mean_squared_error(y_test, pred_xg[:,1]) ** 0.5)
print('LightGBM -- The RMSE of prediction is:', mean_squared_error(y_test, pred_gbm) ** 0.5)
print('Naive Bayes -- The RMSE of prediction is:', mean_squared_error(y_test, pred_nb[:,1]) ** 0.5)


# In[134]:


pred_lr2 = lr.predict(X_test)
pred_rfc2 = rfc.predict(X_test)
pred_dtc2 = dtc.predict(X_test)
pred_abdt2 = abdt.predict(X_test)
pred_xg2 = xg.predict(X_test)
pred_gbm = gbm.predict(X_test,num_iteration=gbm.best_iteration)
pred_nb2 = nb.predict(X_test)


# In[135]:


print("Logistic Regression -- Accuracy: " + str(accuracy_score(y_test, pred_lr2)))
print("Random Forest -- Accuracy: " + str(accuracy_score(y_test, pred_rfc2)))
print("Decision Tree -- Accuracy: " + str(accuracy_score(y_test, pred_dtc2)))
print("AdaBoost Decision Tree -- Accuracy: " + str(accuracy_score(y_test, pred_abdt2)))
print("XGBoost -- Accuracy: " + str(accuracy_score(y_test, pred_xg2)))
print("LightGBM -- Accuracy: " + str(accuracy_score(y_test, (pred_gbm >0.5).astype(int))))
print("Naive Bayes -- Accuracy: " + str(accuracy_score(y_test, pred_nb2)))


# In[215]:


#feature importance
#Random Forest model -top 50 important features

names = X_train.columns.values
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]
top_k = 50
new_indices = indices[:top_k]

print("Features sorted by their score: ")

for f in range(top_k):
    print("%d. feature %d - %s (%0.2f%%)" % (f + 1, indices[f], names[f], importances[new_indices[f]]*100))           


# In[138]:


loandata5.shape

