import json
import numpy as np
import os
import csv
import pandas as pd
import petl as etl
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from joblib import dump, load
def manhattan(fp1, fp2):
    return np.abs((np.array(fp1) - np.array(fp2)))


sos = []
column = []
columns = []
cdata = []
data = []
path = "task1_data/"
with open(os.path.join(path, "task1_fingerprints.json")) as f:
    traj = json.load(f)
    for i in range(103458, 114661):
        for k in traj[str(i)]:
            cdata.append(k)

for i in cdata:
    if i not in column:
        column.append(i)
print(len(column))


#df = np.zeros((114661, 29422))
#df = pd.DataFrame(columns=column)
for i in range(len(traj)):
    for k in traj[str(i)]:
        if k in column:
            data.append([i, k, traj[str(i)][k]])
# print(data)
#with open(os.path.join(path, "task1_train.csv")) as f:
# with open(os.path.join(path, "task1_train.csv")) as f:
#     train_data = []
#     train_h = csv.DictReader(f)
#     for pair in tqdm(train_h):
#         train_data.append([pair['id1'], pair['id2'], float(pair['displacement'])])


# with open(os.path.join(path, "task1_test.csv")) as f:
#     test_data = []
#     test_h = csv.DictReader(f)
#     for pair in tqdm(test_h):
#         test_data.append([pair['id1'], pair['id2']])



# for i in traj:
#     for k in column:
#         if traj[i][k]

df = pd.DataFrame(data)

#df = etl.fromjson("task1_data/task1_fingerprints.json")
unique_rows = df[0].unique()
df_column = pd.DataFrame(np.zeros((5000,7344)), columns=column)
df_column.index =[unique_rows[0:5000]]

for i in unique_rows[0:5000]:
    for k in traj[str(i)]:
        if k in column:
            df_column.loc[[i], [k]] = traj[str(i)][k]
print(df_column)


with open(os.path.join(path, "task1_train.csv")) as f:
    train_data = []
    train_h = csv.DictReader(f)
    for pair in tqdm(train_h):
        train_data.append([pair['id1'], pair['id2'], float(pair['displacement'])])
train = []
test = []
for i in train_data:
    id1 = i[0]
    id2 = i[1]

    if int(id1) in unique_rows[0:5000] and int(id2) in unique_rows[0:5000]:
        dif = np.array(np.array(df_column.loc[int(id1), :]) - np.array(df_column.loc[int(id2), :])).reshape(1, 7344)
        train.append(dif)
        test.append(i[2])

#     sos.append(dist)
# #df = pd.read_csv("ctsten0269_input_1/task1_data/task1_train.csv")
# sos = np.array(sos).reshape(-1,1)
# print(sos)
X_train = np.array(train).reshape(np.array(train).shape[0], 7344)
Y_train = np.array(test)
print(X_train.shape)
print(type(Y_train))
print(type(Y_train[1]))
print(Y_train)

model = RandomForestRegressor(n_estimators= 10, random_state=0)
model.fit(X_train, Y_train)
filename = "Trained_model.sav"
dump(model, filename)

