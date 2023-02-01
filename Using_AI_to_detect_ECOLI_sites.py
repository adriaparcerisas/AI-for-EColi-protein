#!/usr/bin/env python
# coding: utf-8

# In[159]:


import streamlit as st
import pandas as pd
import numpy as np
from shroomdk import ShroomDK
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.ticker as ticker
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import favicon
import math

st.set_option('deprecation.showPyplotGlobalUse', False)
html_code = """
<link rel="shortcut icon" type="image/x-icon" href="favicon.ico">
"""

st.markdown(html_code, unsafe_allow_html=True)

millnames = ['',' k',' M',' B',' T']

def (n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


st.markdown(""" <style> div.css-12w0qpk.e1tzin5v2{
 backg-color: #f5f5f5;
 border: 2px solid;
 padding: 10px 5px 5px 5px;
 border-radius: 10px;
 color: #ffc300;
 box-shadow: 10px;
}
div.css-1r6slb0.e1tzin5v2{
 backg-color: #f5f5f5;
 border: 2px solid; /* #900c3f */
 border-radius: 10px;
 padding: 10px 5px 5px 5px;
 color: green;
}
div.css-50ug3q.e16fv1kl3{
 font-weight: 900;
} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> div.css-ocqkz7.e1tzin5v4{
 backg-color: #f5f5f5;
 border: 2px solid;
 padding: 10px 5px 5px 5px;
 border-radius: 10px;
 color: #ffc300;
 box-shadow: 10px;
}
div.css-keje6w.ce1tzin5v2{
 backg-color: #f5f5f5;
 border: 2px solid; /* #900c3f */
 border-radius: 10px;
 padding: 10px 5px 5px 5px;
 color: orange;
}
div.css-12ukr4l.e1tzin5v0{
 font-weight: 900;
} 
</style> """, unsafe_allow_html=True)

st.markdown('_Created on Wednesday Feb 02 13:36:37 2023_')
st.markdown('_Author: Adrià Parcerisas Albés_')


# In[25]:


st.title('Visualization,  Logistic Regression and SVM: Protein Localization Sites')


# In[28]:


st.subheader('1. Data Description')


# In[27]:


st.markdown("""
1. Title: Protein Localization Sites


2. Creator and Maintainer:
	     Kenta Nakai
             Institue of Molecular and Cellular Biology
	     Osaka, University
	     1-3 Yamada-oka, Suita 565 Japan
	     nakai@imcb.osaka-u.ac.jp
             http://www.imcb.osaka-u.ac.jp/nakai/psort.html
   Donor: Paul Horton (paulh@cs.berkeley.edu)
   Date:  September, 1996
   See also: yeast database


3. The references below describe a predecessor to this dataset and its 
development. They also give results (not cross-validated) for classification 
by a rule-based expert system with that version of the dataset.

Reference: "Expert Sytem for Predicting Protein Localization Sites in 
           Gram-Negative Bacteria", Kenta Nakai & Minoru Kanehisa,  
           PROTEINS: Structure, Function, and Genetics 11:95-110, 1991.

Reference: "A Knowledge Base for Predicting Protein Localization Sites in
	   Eukaryotic Cells", Kenta Nakai & Minoru Kanehisa, 
	   Genomics 14:897-911, 1992.


5. Number of Instances:  336 for the E.coli dataset and 


6. Number of Attributes.
         for E.coli dataset:  8 ( 7 predictive, 1 name )

	     
7. Attribute Information.

  1.  Sequence Name: Accession number for the SWISS-PROT database
  2.  mcg: McGeoch's method for signal sequence recognition.
  3.  gvh: von Heijne's method for signal sequence recognition.
  4.  lip: von Heijne's Signal Peptidase II consensus sequence score.
           Binary attribute.
  5.  chg: Presence of charge on N-terminus of predicted lipoproteins.
	   Binary attribute.
  6.  aac: score of discriminant analysis of the amino acid content of
	   outer membrane and periplasmic proteins.
  7. alm1: score of the ALOM membrane spanning region prediction program.
  8. alm2: score of ALOM program after excluding putative cleavable signal
	   regions from the sequence.



8. Missing Attribute Values: None.


9. Class Distribution. The class is the localization site. Please see Nakai &
		       Kanehisa referenced above for more details.

  cp  (cytoplasm)                                    143
  im  (inner membrane without signal sequence)        77               
  pp  (perisplasm)                                    52
  imU (inner membrane, uncleavable signal sequence)   35
  om  (outer membrane)                                20
  omL (outer membrane lipoprotein)                     5
  imL (inner membrane lipoprotein)                     2
  imS (inner membrane, cleavable signal sequence)      2
  """
  )


# In[29]:


st.subheader('2. Data Analysis')


# In[30]:


# Carreguem les dades i indiquem les columnes i classes
data = pd.read_csv('ecoli.csv')
data.columns = ["Sequence Name", "mcg", "gvh", "lip","chg","aac","alm1","alm2","Outcome"]
data_names = ["mcg", "gvh", "lip","chg","aac","alm1","alm2"]
data_class = ["cp","im","pp","imU","om","omL","imL","imS"]


# In[31]:


st.dataframe(data)


# In[32]:


st.subheader('3. Attributes histogram')


# In[49]:

fig1s = px.histogram(data, x="mcg", nbins=20)
fig1s.update_layout(
    title='McGeochs method for signal sequence recognition',
    xaxis_tickfont_size=14,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

fig2s = px.histogram(data, x="gvh")
fig2s.update_layout(
    title='Von Heijnes for signal sequence recognition',
    xaxis_tickfont_size=14,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

tab1, tab2 = st.tabs(["McGeochs method", "Von Heijnes"])
with tab1:
   st.plotly_chart(fig1s, theme="streamlit", use_container_width=True)
with tab2:
   st.plotly_chart(fig2s, theme="streamlit", use_container_width=True)



fig3 = px.histogram(data, x="lip", nbins=5)
fig3.update_layout(
    title='Von Heijnes Signal Peptidase II consensus sequence score',
    xaxis_tickfont_size=14,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

st.plotly_chart(fig3, theme="streamlit", use_container_width=True)


fig4 = px.histogram(data, x="chg")
fig4.update_layout(
    title='Presence of charge on N-terminus of predicted lipoproteins',
    xaxis_tickfont_size=14,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
st.plotly_chart(fig4, theme="streamlit", use_container_width=True)




fig5 = px.histogram(data, x="aac", nbins=5)
fig5.update_layout(
    title='Score of discriminant analysis of the amino acid content of outer membrane and periplasmic proteins',
    xaxis_tickfont_size=14,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

fig6 = px.histogram(data, x="alm1")
fig6.update_layout(
    title='Score of the ALOM membrane spanning region prediction program',
    xaxis_tickfont_size=14,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)




fig7 = px.histogram(data, x="alm2", nbins=5)
fig7.update_layout(
    title='Score of ALOM program after excluding putative cleavable signal regions from the sequence',
    xaxis_tickfont_size=14,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

tab5, tab6, tab7 = st.tabs(["Score of aac of outer membrane and periplasmic proteins", "Score of the ALOM membrane spanning region", "Score of ALOM program after excluding putative cleavable signal regions"])

with tab5:
   st.plotly_chart(fig5, theme="streamlit", use_container_width=True)
with tab6:
   st.plotly_chart(fig6, theme="streamlit", use_container_width=True)
with tab7:
   st.plotly_chart(fig7, theme="streamlit", use_container_width=True)


fig8 = px.histogram(data, x="Outcome")
fig8.update_layout(
    title='Protein Localization Sites',
    xaxis_tickfont_size=14,
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
st.plotly_chart(fig8, theme="streamlit", use_container_width=True)

# In[51]:


st.markdown("""
####  Binary or categorical variables?

The variables can be categorical or numerical. Categorical variables can be binary (with two unique responses) or with more than two possible responses.
As we can see in the previous histograms, they have two binary categorical variables (Lip and Chg) and the result would also be multiclass, so there are possible answers.
The other variables are numerical (Mcg, Gvh, Aac, Alm1, Alm2)
""")


# In[52]:


st.subheader ('4. Principal Component Analysis (PCA)')
st.write('The PCA does an unsupervised dimensionality reduction, while the logistic regression does the prediction.')


# In[60]:


from sklearn.decomposition import PCA


# In[64]:


x_int=data.iloc[:,1:8]
x_int=pd.DataFrame(x_int)
y=pd.DataFrame(data.iloc[:,8])

pca=PCA() #en aquest cas, no li hem assignat cap valor de components, per tant, pren el nombre per defecte
pca.fit(x_int)


# In[65]:


st.write('How many factors are required to explain at least 80% of data variance?')


# In[70]:


pca_components=pca.components_ # Mirem els components principals
pca_values=pca.explained_variance_ratio_ # Calculem l'explained ratio per veure quines són les variables importants
cum_var_exp = np.cumsum(pca_values)# i fem la cumsum per veure quan arribem a explicar el 80%
dic={'PC-1': pca_components[0],'PC-2': pca_components[1],'PC-3': pca_components[2],'PC-4': pca_components[3],'aac': pca_components[4],'PC-5': pca_components[5],'PC-6': pca_components[6]}
st.write("### Principal Component Analysis (PCA)")
st.write(dic)
dic={'1': pca_values[0],'2': pca_values[1],'3': pca_values[2],'4': pca_values[3],'5': pca_values[4],'6': pca_values[5],'7': pca_values[6]}
st.write('')
col1,col2=st.columns(2)
with col1:
	st.write("### PCA values")
	st.write(dic)
col2.write("### PCA Cumsum Explained Variance values")
col2.write(cum_var_exp)


# In[69]:


st.write('')
st.markdown('Looking at the results obtained, at least 3 variables are necessary to explain more than 80% of the variance on the results **(84.42%)**. With the two most relevant variables it can be explained more than 75%!') 


# In[71]:


#Plot the PCA spectrum, per veure la representació gràfica dels resultats
plt.figure(11)
plt.clf()
#plt.axes([.2, .2, .7, .7])
plt.bar(range(7),pca_values, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(7),cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.axis('tight')
plt.title('PCA spectrum')
st.write("### PCA spectrum")
st.pyplot()



# In[72]:


st.markdown("""
#### Which variables are more related to 'outcomes' in our dataset?
""")


# In[74]:


st.write('In the following table, we can observe the values of each variable in relation with each component. So, the variable with the highest result will be those related with the outcome.')


# In[73]:


st.dataframe(pd.DataFrame(pca.components_, columns=['PC-1', 'PC-2','PC-3', 'PC-4','PC-5', 'PC-6', 'PC-7'], index=x_int.columns))


# In[75]:


st.write('')


# In[76]:


st.subheader('5. Logistic Regression Model')


# In[77]:


# Estudiem les dades
st.write('First of all, we are gonna see the descriptive statistics of our data.')
st.write("### Descriptive Statistics")
st.write(data.describe())


# In[78]:


data['intercept'] = 1.0


# In[122]:


train_cols = data.columns[1:8]
outcome_cols = y.columns

outcomess=[]
for i in range (len (y[outcome_cols])):
    if data['Outcome'].iloc[i] not in outcomess:
        outcomess.append(data['Outcome'].iloc[i])
        
final_outcomes=[]
for i in range (len(y[outcome_cols])):
    if data['Outcome'].iloc[i]=='cp':
        final_outcomes.append(0)
    elif data['Outcome'].iloc[i]=='im':
        final_outcomes.append(1)
    elif data['Outcome'].iloc[i]=='imS':
        final_outcomes.append(2)
    elif data['Outcome'].iloc[i]=='imL':
        final_outcomes.append(3)
    elif data['Outcome'].iloc[i]=='imU':
        final_outcomes.append(4)
    elif data['Outcome'].iloc[i]=='om':
        final_outcomes.append(5)
    elif data['Outcome'].iloc[i]=='omL':
        final_outcomes.append(6)
    else:
        final_outcomes.append(7)
targets=pd.DataFrame(final_outcomes)
targets.columns=["Outcome"]

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, classification_report

x_train, x_test, y_train, y_test = train_test_split(data[train_cols],targets, test_size=.4)




# In[130]:


from sklearn.linear_model import LogisticRegression
import io

# MODEL
#---------------------------------------------------------------------------------------------
result_logit2 = LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(x_train,np.ravel(y_train))

# PREDICCIÓ
#---------------------------------------------------------------------------------------------
predicion2 = result_logit2.predict(x_test)

# Resumem el fitting del model
#---------------------------------------------------------------------------------------------
st.write('')
st.markdown('Fitting model sum-up')
st.markdown("""To continue our investigation, we will need to pass our outcomes from categorical to numerical variables to proceed.
To do that, we will pass the following outcomes to numbers:
- cp-->0
- im-->1
- imS-->2
- imL-->3
- imU-->4
- om-->5
- omL-->6
- pp-->7
         """)
st.write('')
col1,col2,col3=st.columns(3)
with col1:
	st.metric('Explained Variance Score of LogisticRegression multiclass Model: ',(explained_variance_score(y_test,predicion2)))
col2.metric('Logistic Regression Score:',(result_logit2.score(x_test, y_test)))
report=(classification_report(y_test,predicion2))
df = pd.read_csv(io.StringIO(report))
col3.write("### Classification Report")
col3.dataframe(df)




st.write('')
plt.figure(12)
plt.scatter(range(len(y_test)),y_test,label="Real",color="b")
plt.scatter(range(len(y_test)),predicion2,label="Prediction",color="g")
plt.legend(scatterpoints=1)
plt.title('Logistic model vs Real value')
st.pyplot()

# Plot outputs
plt.figure(13)
plt.scatter(range(len(y_test)), y_test, label="Real", color='black')
plt.plot(range(len(y_test)), predicion2, label="Prediction", color='blue',
         linewidth=3)
plt.legend(scatterpoints=1)
plt.xticks(())
plt.yticks(())
plt.title('Component-based logistic model vs Real value')
st.pyplot()


# In[131]:


st.markdown("""
#### Improving the model:
""")
st.write('To improve the model we are gonna use several new metrics that combined with the model can make it more accurable.')
st.write('The main feature to be added is the GridSearchCV, which will help us to improve the predictive results.')
st.write('')


# In[132]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline(steps=[('pca', pca), ('logistic', result_logit2)])


#Prediction
n_components = [3, 5, 7]
Cs = np.logspace(-4, 4, 3)
    
estimator = GridSearchCV(pipe,dict(pca__n_components=n_components,
                              logistic__C=Cs))

RES=estimator.fit(x_train, np.ravel(y_train))
predicion4 = RES.predict(x_test)

col1,col2,col3=st.columns(3)
with col1:
	st.metric('Explained Variance Score of Logistic Model 2:',(explained_variance_score(y_test,predicion4)))
col2.metric('Logistic Regression Score for Model 2:',(RES.score(x_test, y_test)))
report2=(classification_report(y_test,predicion4))
df2 = pd.read_csv(io.StringIO(report2))
col3.write("### Classification Report for Model 2")
col3.dataframe(df2)
st.write('')

#print (np.exp(RES.params))

plt.figure(15)
plt.scatter(range(len(y_test)),y_test,label="Real",color="b")
plt.scatter(range(len(y_test)),predicion4,label="Prediction",color="g")
plt.legend(scatterpoints=1)
plt.title('Logistic model 2 vs real value')
st.pyplot()


# Plot outputs
plt.figure(16)
plt.scatter(range(len(y_test)), y_test, label="Real", color='black')
plt.plot(range(len(y_test)), predicion4, label="Prediction", color='blue',
         linewidth=3)
plt.legend(scatterpoints=1)
plt.xticks(())
plt.yticks(())
plt.title('Component-based logistic model 2 vs real value')
st.pyplot()


# In[144]:


st.subheader('6. Applying Support Vector Machines')
st.markdown("""
ANN-based approaches have been the most used in recent years, but other machine learning methodologies such as random forest (RF) or support vector machines (SVM) are being adopted with increasing frequency.
SVMs estimate the optimal decision function (in the form of a hyperplane) that can separate the two classes together with the goal of finding the hyperplane that maximizes the distance between the closest points. The optimal kernel for the data set has been the RBF; and has been optimized through the scoring function. The variables C and ɣ have also been optimized.
For this reason, the following criteria has been used and applied to our model: SVM model using c=1.0 and lamb=auto as a parameters. In this step, all of the possible kernels have been proved.
""")


# In[153]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

# Primer testegem la diferèmcia entre els pesos balancejats i no balancejats.
result_svm1 = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(x_train,np.ravel(y_train))
# Ens quedem amb el balancejat i apliquem el mateix a tots els kernels

result_svm2 = SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(x_train,np.ravel(y_train))

result_svm3 = SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(x_train,np.ravel(y_train))

result_svm4 = SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovo', degree=2, gamma='auto', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(x_train,np.ravel(y_train))

result_svm5 = SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(x_train,np.ravel(y_train))


# In[154]:


#SVM1
#---------------------------------------------------------------------------------------------
st.subheader('Prediction results for SVM Linear')
# get support vectors
col1,col2,col3=st.columns(3)
with col1:
    # convert the array to a DataFrame
    arr=result_svm2.support_vectors_
    arr2 = pd.DataFrame(arr)

    st.write("""### Support vectors of the SVM Linear""")
    st.write(arr2)
# get indices of support vectors
# convert the array to a DataFrame
arr=result_svm2.support_
arr2 = pd.DataFrame(arr)

col2.write("""### Indices of the SVM Linear""")
col2.write(arr2)

 # get number of support vectors for each class
arr=result_svm2.n_support_
arr2 = pd.DataFrame(arr)

col3.write("""### Number for each class in SVM Linear""")
col3.write(arr2)

st.write('')
st.subheader('Evaluation results:')

#predicció
prediction_2 = result_svm2.predict(x_test)

#Avaluació
col1,col2,col3=st.columns(3)
with col1:
	st.metric('Explained Variance Score of SVM Linear Kernel: ',(explained_variance_score(y_test,prediction_2)))
col2.metric('Accuracy of Balanced SVM Linear Kernel Model: ',(accuracy_score(y_test,prediction_2)))
col3.metric('Score of Linear Kernel: ',(result_svm2.score(x_test, y_test)))

col3,col4=st.columns(2)
report3=(classification_report(y_test,prediction_2))
df3 = pd.read_csv(io.StringIO(report3))
with col3:
	st.write("### Classification Report for SVM Linear Kernel Model")
	st.dataframe(df3)
report4=(confusion_matrix(y_test,prediction_2))
col4.write("### Confusion Matrix for SVM Linear Kernel Model")
col4.write(report4)


# In[155]:


#SVM2
#---------------------------------------------------------------------------------------------
st.write('')
st.subheader('Prediction results for SVM RBF:')
# get support vectors
col1,col2,col3=st.columns(3)
with col1:
    # convert the array to a DataFrame
    arr=result_svm3.support_vectors_
    arr2 = pd.DataFrame(arr)

    st.write("""### Support vectors of the SVM RBF""")
    st.write(arr2)
# get indices of support vectors
# convert the array to a DataFrame
arr=result_svm3.support_
arr2 = pd.DataFrame(arr)

col2.write("""### Indices of the SVM RBF""")
col2.write(arr2)

 # get number of support vectors for each class
arr=result_svm3.n_support_
arr2 = pd.DataFrame(arr)

col3.write("""### Number for each class in SVM RBF""")
col3.write(arr2)

st.write('')
st.subheader('Evaluation results:')

#predicció
prediction_3 = result_svm3.predict(x_test)

#Avaluació
col1,col2,col3=st.columns(3)
with col1:
	st.metric('Explained Variance Score of SVM RBF Kernel: ',(explained_variance_score(y_test,prediction_3)))
col2.metric('Accuracy of Balanced SVM Linear Kernel Model: ',(accuracy_score(y_test,prediction_3)))
col3.metric('Score of RBF Kernel: ',(result_svm3.score(x_test, y_test)))

col3,col4=st.columns(2)
report3=(classification_report(y_test,prediction_3))
df3 = pd.read_csv(io.StringIO(report3))
with col3:
	st.write("### Classification Report for SVM RBF Kernel Model")
	st.dataframe(df3)
report4=(confusion_matrix(y_test,prediction_3))
col4.write("### Confusion Matrix for SVM RBF Kernel Model")
col4.write(report4)


# In[156]:


#SVM2
#---------------------------------------------------------------------------------------------
st.write('')
st.subheader('Prediction results for SVM poly:')
# get support vectors
col1,col2,col3=st.columns(3)
with col1:
    # convert the array to a DataFrame
    arr=result_svm4.support_vectors_
    arr2 = pd.DataFrame(arr)

    st.write("""### Support vectors of the SVM poly""")
    st.write(arr2)
# get indices of support vectors
# convert the array to a DataFrame
arr=result_svm4.support_
arr2 = pd.DataFrame(arr)

col2.write("""### Indices of the SVM poly""")
col2.write(arr2)

 # get number of support vectors for each class
arr=result_svm4.n_support_
arr2 = pd.DataFrame(arr)

col3.write("""### Number for each class in SVM poly""")
col3.write(arr2)

st.write('')
st.subheader('Evaluation results:')

#predicció
prediction_4 = result_svm4.predict(x_test)

#Avaluació
col1,col2,col3=st.columns(3)
with col1:
	st.metric('Explained Variance Score of SVM poly Kernel: ',(explained_variance_score(y_test,prediction_4)))
col2.metric('Accuracy of Balanced SVM poly Kernel Model: ',(accuracy_score(y_test,prediction_4)))
col3.metric('Score of poly Kernel: ',(result_svm4.score(x_test, y_test)))

col3,col4=st.columns(2)
report3=(classification_report(y_test,prediction_4))
df3 = pd.read_csv(io.StringIO(report3))
with col3:
	st.write("### Classification Report for SVM poly Kernel Model")
	st.dataframe(df3)
report4=(confusion_matrix(y_test,prediction_4))
col4.write("### Confusion Matrix for SVM poly Kernel Model")
col4.write(report4)


# In[157]:


#SVM2
#---------------------------------------------------------------------------------------------
st.write('')
st.subheader('Prediction results for SVM sigmoid:')
# get support vectors
col1,col2,col3=st.columns(3)
with col1:
    # convert the array to a DataFrame
    arr=result_svm5.support_vectors_
    arr2 = pd.DataFrame(arr)

    st.write("""### Support vectors of the SVM sigmoid""")
    st.write(arr2)
# get indices of support vectors
# convert the array to a DataFrame
arr=result_svm5.support_
arr2 = pd.DataFrame(arr)

col2.write("""### Indices of the SVM sigmoid""")
col2.write(arr2)

 # get number of support vectors for each class
arr=result_svm5.n_support_
arr2 = pd.DataFrame(arr)

col3.write("""### Number for each class in SVM sigmoid""")
col3.write(arr2)

st.write('')
st.subheader('Evaluation results:')

#predicció
prediction_5 = result_svm5.predict(x_test)

#Avaluació
col1,col2,col3=st.columns(3)
with col1:
	st.metric('Explained Variance Score of SVM sigmoid Kernel: ',(explained_variance_score(y_test,prediction_5)))
col2.metric('Accuracy of Balanced SVM sigmoid Kernel Model: ',(accuracy_score(y_test,prediction_5)))
col3.metric('Score of sigmoid Kernel: ',(result_svm5.score(x_test, y_test)))

col3,col4=st.columns(2)
report3=(classification_report(y_test,prediction_5))
df3 = pd.read_csv(io.StringIO(report3))
with col3:
	st.write("### Classification Report for SVM sigmoid Kernel Model")
	st.dataframe(df3)
report4=(confusion_matrix(y_test,prediction_5))
col4.write("### Confusion Matrix for SVM sigmoid Kernel Model")
col4.write(report4)


# In[148]:


st.write('')


# In[150]:


st.markdown('#### Which of these kernels is the most optimal one for the problem?')


# In[158]:


plt.figure(14)

import plotly.subplots as sp
import plotly.graph_objs as go

y_test = [1, 2, 3, 4, 5]
prediction_2 = [2, 3, 4, 5, 6]

# Create the subplot with 2 rows and 2 columns, and select the first subplot (1,1)
fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("SVM classifier: Linear Balanced",))

# Create the scatter plot for the real values
scatter_real = go.Scatter(x=range(len(y_test)), y=y_test, mode='markers', name='Real', marker=dict(color='blue'))

# Create the scatter plot for the predictions
scatter_pred = go.Scatter(x=range(len(y_test)), y=prediction_2, mode='markers', name='Prediction', marker=dict(color='green'))

# Add the scatter plots to the first subplot
fig.add_trace(scatter_real)
fig.add_trace(scatter_pred)

# Update the layout of the subplot
fig.update_layout(title='SVM classifier: Linear Balanced', xaxis_title='Sample index', yaxis_title='Value')

# Display the subplot in Streamlit
st.plotly_chart(fig)

import plotly.express as px

fig = px.scatter(x=range(len(y_test)), y=y_test, color='Real', labels={'y_test': 'Real'})
fig.add_scatter(x=range(len(y_test)), y=prediction_3, color='Prediction', labels={'prediction_3': 'Prediction'})
fig.update_layout(title='SVM classifier: Rbf')

st.plotly_chart(fig)

fig = px.scatter(x=range(len(y_test)), y=y_test, color='Real', labels={'y_test': 'Real'})
fig.add_scatter(x=range(len(y_test)), y=prediction_4, color='Prediction', labels={'prediction_4': 'Prediction'})
fig.update_layout(title='SVM classifier: Poly')

st.plotly_chart(fig)

fig = px.scatter(x=range(len(y_test)), y=y_test, color='Real', labels={'y_test': 'Real'})
fig.add_scatter(x=range(len(y_test)), y=prediction_5, color='Prediction', labels={'prediction_5': 'Prediction'})
fig.update_layout(title='SVM classifier: Sigmoid')

st.plotly_chart(fig)


# In[160]:


st.write('')
col1,col2=st.columns(2)
with col1:
    st.metric('Score of Linear Kernel: ',result_svm2.score(x_test, y_test))
col2.metric('Score of Rbf Kernel: ',result_svm3.score(x_test, y_test))

col3,col4=st.columns(2)
with col3:
    st.metric('Score of Poly Kernel: ',result_svm4.score(x_test, y_test))
col4.metric('Score of Sigmoid Kernel: ',result_svm5.score(x_test, y_test))


# In[161]:


st.write('')
st.subheader('7. Final conclusion')
st.write('In this analysis we have evaluated a manner to predict the Cellular  Localization Sites of Proteins designing a Probablistic Classification System for Predicting. Based on the results and final scores, it can be clearly said that the optimal model is the SVM Linear Kernel.')


# In[162]:


st.write('Comparing it with the results of the other kernels through our first regression model, the optimal one has been the SVM with a lineal kernel, and the parameter C bein 1.0 and gamma  at 3. Taking into account that the E-Coli outcomes are in ulticlasses, the result obtained is so interesting.')


# In[163]:


st.markdown('This app has been done by **_Adrià Parcerisas_**, a PhD Biomedical Engineer related to Machine Learning and Artificial intelligence technical projects for data analysis and research, as well as dive deep on-chain data analysis about cryptocurrency projects. You can find me on [Twitter](https://twitter.com/adriaparcerisas)')
st.write('')
st.markdown('The full sources used to develop this app can be found to the following link: [Github link](https://github.com/adriaparcerisas/AI-for-EColi-protein)')
st.markdown('_Powered by [Flipside Crypto](https://flipsidecrypto.xyz) and [MetricsDAO](https://metricsdao.notion.site)_')


# In[ ]:




