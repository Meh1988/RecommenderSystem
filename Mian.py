import csv
import math
from datetime import datetime
import random
from statistics import mean
import statistics
from math import sqrt

#The parameters for setting the proposed method
max_r = 5
min_r = 1
num_users = 10000
num_items = 50000
landa = 2.5 # for calculating similaity
teta = 0.6 #for calculate neighbors
alpha = 0.7 #for calculate need rate 
beta = 0.6 #for calculating new neighbors based on confidence
L = 10 #recommendation list

class User:
    items = list()
    ratings = list()
    train_items = list()
    test_items = list()
    train_ratings = list()
    test_ratings = list()
    time_stampts = list()
    similar_users = list()
    similarity = list()
    predictions = list()
    recommendations = list()
    rec_ratings = list()
    prob = list()
    HU = 0
    H_rel = list()
    S_rel = list()
    V_rel = list()
    mean_rates = 0
    
users_list = []
for i in range (num_users):
    a = User()
    a.items = list()
    a.ratings = list()
    a.train_items = list()
    a.test_items = list()
    a.train_ratings = list()
    a.test_ratings = list()
    a.time_stampts = list()
    a.similar_users = list()
    a.similarity = list()
    a.predictions = list()
    a.recommendations = list()
    a.rec_ratings = list()
    a.prob = list()
    a.HU = 0
    a.H_rel = list()
    a.S_rel = list()
    a.V_rel = list()
    a.mean_rates = 0
    a.train_time = list()
    users_list.append(a)
    
all_times = list()

with open('dataset.csv') as csvfile:
    readCSV_ratings = csv.reader(csvfile, delimiter=',')
    for row in readCSV_ratings:
        u = int(row[0])-1
        i = int(row[1])-1
        r = float(row[2])
        t = int(row[3])
        users_list[u].items.append(i)
        users_list[u].ratings.append(r)
        users_list[u].time_stampts.append(t)
        all_times.append(t)
        
min_time = min(all_times) 
max_time = max(all_times)      
        
for i in range(num_users):
    num_train = int(0.8 * len(users_list[i].ratings))
    for j in range(num_train):
        users_list[i].train_items.append(users_list[i].items[j])
        users_list[i].train_ratings.append(users_list[i].ratings[j])
        t = math.ceil((users_list[i].time_stampts[j] - min_time) / 2592000)
        users_list[i].train_time.append(t)
        users_list[i].mean_rates = users_list[i].mean_rates + users_list[i].ratings[j]
        
    if num_train != 0:
        users_list[i].mean_rates = users_list[i].mean_rates / num_train
    
    for j in range(num_train, len(users_list[i].items)):
        users_list[i].test_items.append(users_list[i].items[j])
        users_list[i].test_ratings.append(users_list[i].ratings[j])
 
for u in range (num_users):
    for v in range (num_users):
        if u != v:
            sim = 0
            sim1 = 0
            sim2 = 0
            sim3 = 0
            for i in range (len(users_list[u].train_items)):
                item = users_list[u].train_items[i]
                if item in users_list[v].train_items:
                    index = users_list[v].train_items.index(item)
                    tw = math.sqrt(math.exp(-landa * (max_time - users_list[u].train_time[i])) *
                                   math.exp(-landa * (max_time - users_list[v].train_time[index])))
                    sim1 += ((users_list[u].train_ratings[i] - users_list[u].mean_rates) * 
                             (users_list[v].train_ratings[index] - users_list[v].mean_rates) * tw)
                    sim2 += (math.pow(users_list[u].train_ratings[i] - users_list[u].mean_rates, 2) * tw)
                    sim3 += (math.pow(users_list[v].train_ratings[index] - users_list[v].mean_rates, 2) * tw)
                    
            sim2 = math.sqrt(sim2)
            sim3 = math.sqrt(sim3)
            if (sim2 * sim3) != 0:
                sim = sim1 / (sim2 * sim3)
            if sim >= teta:
                users_list[u].similar_users.append(v)
                users_list[u].similarity.append(sim)
                

HF = list()
for i in range(num_items):
    f = random.randint(0, 7)
    HF.append(f)
                 
for u in range(num_users):
    hu = 0
    for i in range(len(users_list[u].train_items)):
        hu += HF[users_list[u].train_items[i]]
    if len(users_list[u].train_items) != 0:
        hu /= len(users_list[u].train_items)
        users_list[u].HU = hu

H_all = list()
S_all = list()
V_all = list()

for u in range(num_users):
    for i in range(len(users_list[u].test_items)):
        h = 0
        s = 0
        v = 0
        item = users_list[u].test_items[i]
        p = users_list[u].mean_rates
        s1 = 0 
        s2 = 0
        for v in range(len(users_list[u].similar_users)):
            if item in users_list[users_list[u].similar_users[v]].train_items:
                index = users_list[users_list[u].similar_users[v]].train_items.index(item)
                s1 += users_list[u].similarity[v] * (users_list[users_list[u].similar_users[v]].train_ratings[index] 
                 - users_list[users_list[u].similar_users[v]].mean_rates)
                s2 += users_list[u].similarity[v]
                h += users_list[users_list[u].similar_users[v]].HU
                s += users_list[u].similarity[v]                
        if s2 != 0:
            p += s1 / s2
        users_list[u].predictions.append(p)
        users_list[u].H_rel.append(h)
        users_list[u].S_rel.append(s)
        for v in range(len(users_list[u].similar_users)):
            if item in users_list[users_list[u].similar_users[v]].train_items:
                index = users_list[users_list[u].similar_users[v]].train_items.index(item)
                v += math.pow(users_list[u].similarity[v] * (users_list[users_list[u].similar_users[v]].train_ratings[index]
                                                    - users_list[users_list[u].similar_users[v]].mean_rates
                                                    - p + users_list[u].mean_rates), 2)
        if s != 0:
            v /= s
        else:
            v = 0
        users_list[u].V_rel.append(v)
        H_all.append(h)
        S_all.append(s)
        V_all.append(v)
            
H_med = statistics.median(H_all)
S_med = statistics.median(S_all)
V_med = statistics.median(V_all)
    
for u in range(num_users):
    for i in range(len(users_list[u].test_items)):
        fh = 1 - (H_med / (H_med + users_list[u].H_rel[i]))
        fs = 1 - (S_med / (S_med + users_list[u].S_rel[i]))
        a = math.log(0.5 , math.e)
        if V_med < 4:
            b = math.log(((max_r - min_r - V_med)/(max_r - min_r)), math.e)
        else:
            b = 0
        if(b != 0):
            gama = a / b
        else:
            gama = 0
            
        fv = math.pow((max_r - min_r - V_med) / (max_r - min_r), gama)
        rel = math.pow(fh * fs * math.pow(fv, fs), 1 / (2 + fs))
        if rel < alpha:
            new_k = list()
            new_sim = list()
            for v in range(len(users_list[u].similar_users)):
                conf = 0
                sim1 = 0
                for j in range (len(users_list[u].train_items)):
                    item = users_list[u].train_items[j]
                    if item in users_list[users_list[u].similar_users[v]].train_items:
                        index = users_list[users_list[u].similar_users[v]].train_items.index(item)
                        sim1 += ((users_list[u].train_ratings[j] - users_list[u].mean_rates) * 
                                 (users_list[users_list[u].similar_users[v]].train_ratings[index] - users_list[users_list[u].similar_users[v]].mean_rates) * rel)
                        sim2 += (math.pow(users_list[u].train_ratings[j] - users_list[u].mean_rates, 2) * rel)
                        sim3 += (math.pow(users_list[users_list[u].similar_users[v]].train_ratings[index] - users_list[users_list[u].similar_users[v]].mean_rates, 2) * rel)
                sim2 = math.sqrt(sim2)
                sim3 = math.sqrt(sim3)
                if (sim2 * sim3) != 0:
                    conf = sim1 / (sim2 * sim3)
                if conf > beta:
                    new_k.append(users_list[u].similar_users[v])
                    new_sim.append(users_list[u].similarity[v])
            
            item = users_list[u].test_items[i]
            p = users_list[u].mean_rates
            s1 = 0 
            s2 = 0
            for v in range(len(new_k)):
                if item in users_list[new_k[v]].train_items:
                    index = users_list[new_k[v]].train_items.index(item)
                    s1 += new_sim[v] * (users_list[new_k[v]].train_ratings[index] 
                     - users_list[new_k[v]].mean_rates)
                    s2 += new_sim[v]              
            if s2 != 0:
                p += s1 / s2
                users_list[u].predictions[i] = p
                
                
for u in range (num_users):
    temp_items = list()
    temp_ratings = list()
    for i in range(len(users_list[u].predictions)):
        temp_ratings.append(users_list[u].predictions[i])
        temp_items.append(users_list[u].test_items[i])
    if len(users_list[u].predictions) <= L:
        for j in range(len(users_list[u].predictions)):
            users_list[u].recommendations.append(users_list[u].test_items[j])
            users_list[u].rec_ratings.append(users_list[u].predictions[j])
    else:
        for j in range(L):
            ma = temp_ratings.index(max(temp_ratings)) 
            users_list[u].recommendations.append(temp_items[ma])
            users_list[u].rec_ratings.append(temp_ratings[ma])
            temp_items.pop(ma)
            temp_ratings.pop(ma)
    

for u in range (num_users):
    if len(users_list[u].test_items) >= 1:
        len_prob = int(len(users_list[u].test_items)/2)
        for i in range(len_prob):
            users_list[u].prob.append(users_list[u].test_items[i])

precision = 0
recall = 0
F1 = 0
NDCG = 0
num = 0

for u in range (num_users):
    le = len(users_list[u].prob)
    if le >= 1:
        num = num + 1
        a = 0
        h = 0
        for i in range (len(users_list[u].recommendations)):
            item = users_list[u].recommendations[i]
            h = h + HF[item]
            if item in users_list[u].prob:
                a = a + 1
        precision = precision + (a / L)
        recall = recall + (a / le)


if num >= 1:
    precision = precision / num
    recall = recall / num
    F1 = (2 * precision * recall) / (precision + recall)

        
num = 0
for u in range (num_users):
    le = len(users_list[u].prob)
    if le >= 1:
        num = num + 1
        dcg = 0
        dcg_max = 1
        item = users_list[u].recommendations[0]
        if item in users_list[u].prob:
            dcg = dcg + 1
        for i in range (1, len(users_list[u].recommendations)):
            dcg_max = dcg_max + (1 / (math.log((i+1),2)))
            item = users_list[u].recommendations[i]
            if item in users_list[u].prob:
                dcg = dcg + (1 / (math.log((i+1),2)))
        NDCG = NDCG + (dcg / dcg_max)
        
if num >= 1:
    NDCG = NDCG / num 

print("Precision:" + str(precision))  
print("Recall:" + str(recall)) 
print("F1:" + str(F1)) 
print("NDCG:" + str(NDCG))     
   
    
    
    
