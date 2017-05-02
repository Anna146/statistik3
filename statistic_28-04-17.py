
# coding: utf-8

# # Statistic 12-04-2017

# In[1]:

import pickle       # (De-) Serialization
import pylab as plt # plot
import pandas
import numpy as np
import json


# ## Used TestFiles

# In[2]:

testfiles = [f.split('/')[-1][2:] for f in pickle.load(open('./test_files.pkl','rb'))]


# In[3]:

print(len(testfiles))


# ## Data

# In[4]:

predicted_alt = pickle.load(open('./predicted.pkl', 'rb'))
predicted = json.load(open('my_pred.json', 'r'))


# In[5]:
throws = [0, 1, 10, 11, 12, 13, 16, 18, 19, 20]
real = pickle.load(open('./real.pkl', 'rb'))

real = [[w if w not in throws else 8 for w in d] for d in real]
predicted = [[w if w not in throws else 8 for w in d] for d in predicted]
predicted_alt = [[w if w not in throws else 8 for w in d] for d in predicted_alt]
real = [p[1][:len(p[0])] for p in zip(predicted,real)]
predicted = [p[0][:len(p[1])] for p in zip(predicted,real)]
predicted_alt = [p[0][:len(p[1])] for p in zip(predicted_alt,real)]
# ## Scoring per File

# In[6]:

db = pickle.load(open('./databuilder.pkl', 'rb'))

db.labels = [x[1] for x in enumerate(db.labels) if x[0] not in throws]
print(db.labels)

# In[7]:

from sklearn import metrics


# ## Score About ALL Test Files

# In[22]:

predicted_all_test_files = np.concatenate(predicted)
real_all_test_files = np.concatenate(real)
predicted_alt_all_test_files = np.concatenate(predicted_alt)


# ### <font style="color:red">Precision for all test Files<font>

# In[23]:

print('precision recall f1 accuracy')
precision_per_label = metrics.precision_score(y_true=real_all_test_files, y_pred=predicted_all_test_files, average=None)
print(np.mean(precision_per_label))


# ### <font style="color:red">Recall for all test Files<font>

# In[24]:

recall_per_label = metrics.recall_score(y_true=real_all_test_files, y_pred=predicted_all_test_files, average=None)
print(np.mean(recall_per_label))


# ### <font style="color:red">F1 Score for all test Files<font>

# In[25]:

f1_per_label = metrics.f1_score(y_true=real_all_test_files, y_pred=predicted_all_test_files, average=None)
print(np.mean(f1_per_label))

acc_per_label = metrics.accuracy_score(y_true=real_all_test_files, y_pred=predicted_all_test_files)
print(np.mean(acc_per_label))


print('precision recall f1 accuracy')
precision_alt_per_label = metrics.precision_score(y_true=real_all_test_files, y_pred=predicted_alt_all_test_files, average=None)
print(np.mean(precision_alt_per_label))


# ### <font style="color:red">Recall for all test Files<font>

# In[24]:

recall_alt_per_label = metrics.recall_score(y_true=real_all_test_files, y_pred=predicted_alt_all_test_files, average=None)
print(np.mean(recall_alt_per_label))


# ### <font style="color:red">F1 Score for all test Files<font>

# In[25]:

f1_alt_per_label = metrics.f1_score(y_true=real_all_test_files, y_pred=predicted_alt_all_test_files, average=None)
print(np.mean(f1_alt_per_label))

acc_alt_per_label = metrics.accuracy_score(y_true=real_all_test_files, y_pred=predicted_alt_all_test_files)
print(np.mean(acc_alt_per_label))


# In[27]:

'''
plt.figure(figsize=(15,5))
plt.bar(range(len(precision_per_label)), precision_per_label)
plt.ylim(0,1)
plt.xticks(np.arange(len(db.labels)), db.labels, rotation=90)
plt.title('Precision Per Label')


# In[28]:

pandas.DataFrame(np.transpose([db.labels, precision_per_label]), columns=['Label', 'Precision'])
plt.show()



# In[29]:
plt.figure(figsize=(15,5))
plt.bar(range(len(recall_per_label)), recall_per_label)
plt.ylim(0,1)
plt.xticks(np.arange(len(db.labels)), db.labels, rotation=90)
plt.title('Recall Per Label')
plt.show()

# In[30]:

pandas.DataFrame(np.transpose([db.labels, recall_per_label]), columns=['Label', 'Recall'])
plt.show()

# In[31]:

plt.figure(figsize=(15,5))
plt.bar(range(len(f1_per_label)), f1_per_label)
plt.ylim(0,1)
plt.xticks(np.arange(len(db.labels)), db.labels, rotation=90)
plt.title('F1 Per Label')


# In[32]:

pandas.DataFrame(np.transpose([db.labels, f1_per_label]), columns=['Label', 'F1'])
plt.show()
'''