
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import csv
import json
import collections
import scholarly
from itertools import product
import nltk
import spacy
import matplotlib.pyplot as plt
nlp = spacy.load('en_core_web_md')


# # Parse our data into CSV file

# In[2]:


def generate_citation_count(citation_dict, df):
    citation_list = zerolistmaker(df.shape[0])
    for key in citation_dict:
        citation_list[int(key)] = citation_dict[key]
    return citation_list


# In[3]:


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


# In[4]:


def parse_data_into_dataframe(filename):
    row = []
    df = pd.DataFrame()
    refCount = 0
    title = None
    authors = None
    year = None
    publication_venue = None
    index = None
    abstract = None

    citation_dict = {}

    with open(filename) as file:
        content = {}
        for line in file:
            if(len(line) > 2):
                #* --- paperTitle
                #@ --- Authors
                #t ---- Year
                #c  --- publication venue
                #index 00---- index id of this paper
                #% ---- the id of references of this paper (there are multiple lines, with each indicating a reference)
                #! --- Abstract
                value = line[2:-1]
                if(line[1] =='*'):
                    title = value
                elif(line[1] =='@'):
                    authors = value 
                elif(line[1] == 't'):
                    year = value
                elif(line[1] =='c'):
                    publication_venue = value
                elif(line[1] =='i'):
                    index = value[4:] 
                elif(line[1] == '%'):
                    if(value in citation_dict):
                        citation_dict[value] = citation_dict[value] + 1
                    else:
                        citation_dict[value] = 1
                elif(line[1] == '!'):
                    abstract = value
            if(line == '\n'):
                content = {'paperTitle': title, 'Authors' : authors, 'Year': year, 'publication venue': publication_venue, 'index id' : index, 'Abstract' : abstract}
                row.append(content)
                title = None
                authors = None
                year = None
                publication_venue = None
                index = None
                abstract = None
        
    df = pd.DataFrame(row)  
    
    df['citation_count'] = generate_citation_count(citation_dict, df)
    return df


# # Concat our data from multiple data sets and write it into a CSV file.

# In[5]:


def concat_data():
    df_1 = parse_data_into_dataframe('outputacm.txt')
    df_2 = parse_data_into_dataframe('citation-network2.txt')
    df_1 = df_1.drop(['index id'], axis=1)
    df_2 = df_2.drop(['index id'], axis=1)
    df_m = pd.concat([df_1, df_2], ignore_index=True)
    df_3 = parse_data_into_dataframe('V3.txt')
    df_3 = df_3.drop(['index id'], axis=1)
    df_m1 = pd.concat([df_m, df_3], ignore_index=True)
    return df_m1
    
train_df = parse_data_into_dataframe('citation-network2.txt')
train_df


# In[6]:


train_df.to_csv('train_new.csv', index=False)


# In[7]:


columns = ['Abstract', 'Authors', 'Year', 'index id', 'paperTitle', 'publication venue', 'citation_count']


# In[8]:


train_df = pd.read_csv('train_new.csv', nrows=10000)


# In[9]:


train_df


# # Get test data from Arxiv

# In[10]:


#Parse JSON data and create a df
def parse_json_data_into_dataframe(filename, nrows):
    data = []
    count = 0
    with open(filename) as f:
        for line in f:
            if(count >= nrows):
                break
            data.append(json.loads(line))
        
    df = pd.DataFrame(data);
    return df;


# In[11]:


with open ('arxivData.json', 'rb') as f:
    data = json.load(f)
#author = data['author']
new_df = pd.DataFrame(data)
author_list = []
authors = new_df['author']
authors
new_df


# In[12]:


final_list = []
citation_count_list = []
for entry in authors:
    a = entry
    all_authors = ""
    a = a[1:-1]
    a_list = a.split(",")
    for author in a_list:
        author_name = author[1:-1].split(":")
        if(len(author_name) > 1):
            author_name = author_name[1].replace("'","")
            all_authors += author_name + ","
    all_authors = all_authors[:-1]
    final_list.append(all_authors)
    citation_count_list.append(0)


# In[13]:


test1 = pd.DataFrame()
test1["authors"] = final_list
new_df["author"] = test1["authors"]
new_df = new_df.drop("link", axis = 1)
new_df = new_df.drop("tag", axis = 1)
new_df = new_df.drop("day", axis = 1)
new_df = new_df.drop("month", axis = 1)
new_df['Authors'] = new_df['summary']
new_df = new_df.drop("summary", axis = 1)
new_df.columns = ['Authors', 'index id', 'paperTitle', 'Year', 'Abstract']
test_df = new_df


# In[14]:


test_df['citation_count'] = citation_count_list
test_df.to_csv('test_new.csv', index=False)
test_df = pd.read_csv('test_new.csv', nrows=100)


# In[15]:


test_df


# In[16]:


# drop venue column and fill NAN values with N/A
train_df = train_df.drop(['publication venue'], axis=1)
train_df['Authors'].fillna('N/A', inplace=True)
train_df['Abstract'].fillna('N/A', inplace=True)
train_df['Year'].fillna(0.0, inplace=True)
train_df['Year'] = train_df['Year'].astype(np.int64)

test_df['Authors'].fillna('N/A', inplace=True)
test_df['Abstract'].fillna('N/A', inplace=True)
test_df['Year'].fillna(0.0, inplace=True)
test_df['Year'] = test_df['Year'].astype(np.int64)


# In[17]:


train_df.head(12)


# In[18]:


test_df


# # Get Domain of each paper using Scumpy based on NER

# In[19]:


domain_list = ["Biomedical Research", "Chemistry", "Biology", "Economics", "Earth Science","Physics", "Neuroscience",
                "Political", "Economics", "Literature", "Computer Science", "Software", "Microprocessor", 
               "Physics", "History", "Art", "Statistics", "Business", "Health", "Geometry", "Maths"]


# In[20]:


def get_domain(string):
    valid_domain = 'N/A'
    max_score = 0
    word1 = nlp(string)
    for domain in domain_list:
        word2 = nlp(domain)
        score = word1.similarity(word2)
        if max_score < score:
            max_score = score
            valid_domain = domain

    return valid_domain


# In[21]:


def fetch_domain_for_each_paper(df):
    domain_list1 = []
    for index,row in df.iterrows():
        author_domain = get_domain(row["paperTitle"])
        domain_list1.append(author_domain)
    df["domain"] = domain_list1
    return df


# In[22]:


train_df = fetch_domain_for_each_paper(train_df)


# In[23]:


test_df = fetch_domain_for_each_paper(test_df)


# In[24]:


test_df


# In[25]:


paper_domain_dict = {}
for index, rows in train_df.iterrows():
    domain = rows["domain"]
    if domain in paper_domain_dict:
        paper_domain_dict[domain] += 1
    else:
        paper_domain_dict[domain] = 1


# # Generate Author Table

# In[26]:


author_papers = {}
author_citations = {}
author_citations_list = {}
author_domain_list = {}

for index,row in train_df.iterrows():
    authors = row["Authors"]
    authors_list = authors.split(',')
    for author in authors_list:
        if author in author_papers:
            author_papers[author] += 1
        else:
            author_papers[author] = 1
        if author in author_citations:
            author_citations[author] += row["citation_count"]
        else:
            author_citations[author] = row["citation_count"]
        if author in author_citations_list:
            author_citations_list[author].append(row["citation_count"])
        else:
            author_citations_list[author] = [row["citation_count"]]
        if author not in author_domain_list:
            author_domain_list[author] = row["domain"]


# In[27]:


def compute_h_index(citations):
    hIndex = 0
    citations.sort(reverse=True)
    for i in range(len(citations)):
        if citations[i] >= i + 1:
            hIndex = i+1
    return hIndex


# In[28]:


def compute_i10_index(citations):
    count = 0;
    for i in range(len(citations)):
        if citations[i] >= 10:
            count = count + 1
    return count;


# In[29]:


def compute_g_index(citations, avg_citations):
    count = 0;
    for i in range(len(citations)):
        if citations[i] >= avg_citations:
            count = count + 1
    return count;


# In[30]:


def generate_author_table(): 
    authors_list = []
    citation_count_list = []
    paper_count_list = []
    h_index_list = []
    g_index_list = []
    i10_index_list = []
    domain_list = []
    for key in author_citations:
        authors_list.append(key)
        citation_count_list.append(author_citations[key])
        h_index_list.append(compute_h_index(author_citations_list[key]))
        i10_index_list.append(compute_i10_index(author_citations_list[key]))
        g_index_list.append(compute_g_index(author_citations_list[key], author_citations[key]/author_papers[key]))
        paper_count_list.append(author_papers[key]) 
        domain_list.append(author_domain_list[key])

    df_new = pd.DataFrame()
    df_new["author_name"] = authors_list
    df_new["citation_count"] = citation_count_list
    df_new["paper_count"] = paper_count_list
    df_new["h_index"] = h_index_list
    df_new["i10_index"] = i10_index_list
    df_new["g_index"] = g_index_list
    df_new["domain"] = domain_list
    return df_new
    
df_n = generate_author_table()


# In[31]:


df_n = df_n.sort_values(by='h_index', ascending=False)
df_n["average_citations"] = df_n["citation_count"]/df_n["paper_count"]


# In[32]:


score_list = []
for index, rows in df_n.iterrows():
    count = rows['paper_count']
    domain = rows["domain"]
    total_count = paper_domain_dict[domain]
    score_list.append(count / total_count)
df_n['domain_Score'] = score_list
df_n


# # Calculate Author Metric

# In[33]:


df_n["SVS - index"] = 0.3*df_n["average_citations"] + 0.3*df_n["paper_count"] + 0.5*df_n["citation_count"]  + 0.25*df_n["i10_index"] + 0.25*df_n["g_index"] + 0.25*df_n["domain_Score"]


# In[34]:


df_n = df_n.sort_values(by='SVS - index', ascending=False)


# In[35]:


df_n


# # Ques 1 : Top 100 ranked researchers from multiple disciplines

# In[36]:


# You must identify top 100 ranked researchers from multiple disciplines based on your ranking metric 
# and see how it stacks up against their respective h-index.

df_n.head(500)


# # Ques 2

# In[37]:


# Using a citation network graph, devise a “reach” function which can identify the degree of a paper’s influence, 
# which can either be localised in a domain or have a more global inter-disciplinary effect.


# In[38]:


df_new = train_df.copy()


# In[39]:


df_new = df_new.groupby('Year')['domain'].value_counts().unstack().fillna(0)


# In[40]:


df_new = df_new.unstack()


# In[41]:


df_new = df_new.reset_index()


# In[42]:


df_new.columns = ['domain', 'Year', 'Count']


# In[43]:


score_list = []
for index, rows in df_new.iterrows():
    domain = rows['domain']
    count = rows['Count']
    total_count = paper_domain_dict[domain]
    score_list.append(count / total_count)
df_new['domain_Score'] = score_list
df_new


# # Reach function

# In[44]:


# Get citations count of paper present in test set
def generate_citation_count_from_google_scholar(df):
    citation_list = []
    for index,row in df.iterrows():
        print(index)
        citations = 0
        title = row['paperTitle']
        search_query = scholarly.search_pubs_query(title)
        val = next(search_query, "N/A")
        if val != 'N/A' and 'title' in val.bib.keys():
            title1 = val.bib['title']
            if title1[0:3] == title[0:3]:
                if hasattr(val, 'citedby'):
                    citations = val.citedby
        citation_list.append(citations)
    df["citation_count"] = citation_list
    return df

test_df = generate_citation_count_from_google_scholar(test_df)


# In[46]:


test_df.to_csv('test_new_arxiv.csv', index=False)


# In[48]:


test_df


# In[63]:


domain_paper_list = []
for index, rows in test_df.iterrows():
    authors = rows["Authors"]
    citations = rows["citation_count"]
    domain = rows["domain"]
    year = rows['Year']
    domain_list = df_new[df_new['domain'] == domain]
    print(domain_list)
    print(year)
    print(domain)
    domain_score = 0
    if not domain_list.empty:
        if year is domain_list['Year'] == year:
            domain_score = domain_list.loc[domain_list['Year'] == year, 'domain_Score'].iloc[0]
    domain_paper_list.append(domain_score)
test_df['domain_score'] = domain_paper_list


# In[65]:


test_df


# In[101]:


# paper influence by giving wieghts to each parameter:
# 1) author popularity : 0.2, 
# 2) domain-populaarity : 0.2, more popular domain, less influence paper is
# 3) citations_count : 0.8
metric_paper_list = []
domain_paper_list = []
for index, rows in train_df.iterrows():
    authors = rows["Authors"]
    citations = rows["citation_count"]
    domain = rows["domain"]
    year = rows['Year']
    authors_list = authors.split(',')
    count = 1
    curr_sum = 0
    for author in authors_list:
        auth_score = df_n.loc[df_n['author_name'] == author, 'SVS - index'].iloc[0]
        curr_sum = curr_sum + auth_score
        curr_sum = curr_sum / count
        count = count + 1
    domain_list = df_new[df_new['domain'] == domain]
    domain_score = 0
    if not domain_list.empty:
        domain_score = domain_list.loc[domain_list['Year'] == year, 'domain_Score'].iloc[0]
    paper_score = 0.5*curr_sum + 0.8*citations - domain_score + 2 * citations/(2018 - year)
    
    metric_paper_list.append(paper_score)
    domain_paper_list.append(domain_score)
train_df['reach_score'] = metric_paper_list
train_df['domain_score'] = domain_paper_list


# In[102]:


train_df


# # Time Analysis of Paper citation count

# In[69]:


# Graph 1: Graph which will tell the domain and the total number of papers in the domain


# In[70]:


df_domain_count = paper_domain_dict


# In[71]:


df_domain_count


# In[72]:


df1 = pd.DataFrame(list(paper_domain_dict.items()), columns=['Domain', 'Count'])


# In[73]:


df1 = df1.sort_values(by='Count', ascending=False)
df1 = df1.head(10)


# In[74]:


import matplotlib.pyplot as plt_1

x = df1['Domain']
y = df1['Count']
plt_1.bar(x,y, width = 0.8, color = ['red', 'green'])
plt_1.xlabel('Domain')
plt_1.ylabel('Count')
plt_1.xticks(rotation=90);


# In[75]:


mydf = df_n.head(10)

x = mydf['author_name']
y = mydf['citation_count']
plt_1.bar(x,y, width = 0.8, color = ['blue', 'orange'])
plt_1.xlabel('Authors')
plt_1.ylabel('Citation Count')
plt_1.xticks(rotation=90);


# In[76]:


mydf = df_n.head(10)

x = mydf['author_name']
y = mydf['SVS - index']
plt_1.bar(x,y, width = 0.8, color = ['magenta', 'cyan'])
plt_1.xlabel('Authors')
plt_1.ylabel('SVS-Index')
plt_1.xticks(rotation=90);


# In[77]:


import warnings
warnings.filterwarnings("ignore")
temp_df = df_new.ix[(df_new['domain'] == 'Computer Science')]
temp_df1 = df_new.ix[(df_new['domain'] == 'Geometry')]
temp_df2 = df_new.ix[(df_new['domain'] == 'Business')]


# In[78]:


temp_df.plot(x='Year', y=['domain', 'Count'], figsize=(10,5), grid=True, color="green", label=['Computer Science','Computer Science'])
temp_df1.plot(x='Year', y=['domain', 'Count'], figsize=(10,5), grid=True, color="red", label=['Geometry','Geometry'])
temp_df2.plot(x='Year', y=['domain', 'Count'], figsize=(10,5), grid=True, color="magenta", label=['Business','Business'])


# In[79]:


x = temp_df['Year']
y1 = temp_df['Count']
y2 = temp_df1['Count']
plt.plot(x, y1, x, y2)
plt.legend(['Computer Science', 'Geometry'])


# # Baseline Model

# In[80]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


# In[81]:


features = ['Abstract', 'paperTitle', 'domain', 'Authors', 'Year', 'citation_count']


# In[82]:


cat_cols = ["Authors", "paperTitle", "domain", "Abstract"]

for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))


# In[83]:


X = train_df[features].values
Y = train_df['reach_score'].values


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)


# In[85]:


linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
print (linear_regressor.coef_)
y_test_pred = linear_regressor.predict(X_test)


# In[86]:


root_mean_square_error = np.sqrt(mean_squared_error(y_test,y_test_pred))
root_mean_square_error


# In[87]:


linear_regressor.fit(X, Y)


# In[88]:


test_predictions = linear_regressor.predict(test_df[features])


# In[89]:


test_predictions


# In[90]:


test_df


# In[91]:


#Make a csv of final predicted results
submission = pd.DataFrame(
    {'key': test_df.paperTitle, 'reach_score': test_predictions},
    columns = ['key', 'reach_score'])
submission.to_csv('submission.csv', index = False)


# In[92]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(linear_regressor, random_state=1).fit(X, Y)
eli5.show_weights(perm, feature_names = features)


# In[93]:


#Random Forest Model
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(n_estimators=24, max_features=None, max_depth=27, min_samples_split=4,
                              min_samples_leaf=3, random_state=0)
random_forest.fit(X_train, y_train)
y_test_pred_1 = random_forest.predict(X_test)


# In[94]:


root_mean_square_error_rf = np.sqrt(mean_squared_error(y_test,y_test_pred_1))
root_mean_square_error_rf


# In[95]:


random_forest.fit(X, Y)
test_predictions_rf = random_forest.predict(test_df[features])

#Make a csv of final predicted results
submission = pd.DataFrame(
    {'key': test_df.paperTitle, 'Reach-Score': test_predictions_rf},
    columns = ['key', 'Reach-Score'])
submission.to_csv('submission_rf1.csv', index = False)


# In[96]:


test_predictions_rf


# In[97]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(random_forest, random_state=1).fit(X, Y)
eli5.show_weights(perm, feature_names = features)


# # Advanced Model

# In[98]:


test_X = test_df[features]
import lightgbm as lgb
#Train and run the lgb model
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression", "metric" : "rmse", "num_leaves" : 30,
        "min_child_samples" : 100, "learning_rate" : 0.1, "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5, "bagging_frequency" : 5, "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 2000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

X_train, X_test, y_train, y_test
pred_test, model, pred_val = run_lgb(X_train, y_train, X_test, y_test, test_X)


# In[99]:


#Make a csv of final predicted results
submission = pd.DataFrame(
    {'key': test_df.paperTitle, 'Reach-Score': pred_test},
    columns = ['key', 'Reach-Score'])
submission.to_csv('advanced_lgb.csv', index = False)


# In[103]:


test_df = test_df.sort_values(by='citation_count', ascending=False)
test_df.head(10)

