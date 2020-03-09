import json
from itertools import chain
import requests
import numpy as np
import pandas as pd
import time
from random import randint
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LinearRegression

def get_response(method_name, args):
    query = 'https://codeforces.com/api/' + method_name + '?'
    for key, value in args.items():
        query += key + '=' + str(value) + '&'
    query = query[0: len(query) - 1]
    resp = requests.get(query)
    assert resp.status_code == 200
    return resp.json()['result']

def get_data():
    users_data = get_response('user.ratedList', {'activeOnly': 'true'})
    cnt_users = len(users_data)
    df = pd.DataFrame(users_data, columns=['handle', 'rating'])

    # adjust
    middle_place = 2300
    first_part_step = 12
    second_part_step = 120

    #middle_place = 500
    #first_part_step = 50000 
    #second_part_step = 50000 
    dfs = []
    for rng in [range(0, middle_place, first_part_step), range(middle_place, cnt_users, second_part_step)]:
        dfs.append(df.iloc[rng])
    df = pd.concat(dfs)

    min_cnt_contests = 40
    good_ids = []
    df = df.reset_index()
    for idx in range(0, df.shape[0]):
        handle = df['handle'].values[idx]
        resp = get_response('user.rating', {'handle': handle})
        if len(resp) >= min_cnt_contests:
            good_ids.append(idx)
        if idx % 10 == 9:
            print ('Users select done: ' + str(idx + 1) + "/" + str(df.shape[0]))

    return df[df.index.isin(good_ids)]

def calc_user_stats(handle, user_rating):
    df_subms = pd.DataFrame(get_response('user.status', {'handle': handle, 'from': '1'}), 
                            columns=['verdict', 'problem'])
    df_subms['tags'] = pd.DataFrame.from_dict(df_subms['problem'].array)['tags']
    df_subms['rating'] = pd.DataFrame.from_dict(df_subms['problem'].array)['rating']
    df_subms['name'] = pd.DataFrame.from_dict(df_subms['problem'].array)['name']
    df_subms = df_subms.drop('problem', axis=1)
    df_subms = df_subms.query('verdict == "OK"').query('rating != "NaN"').drop_duplicates('name')
    
    points = {
        'strings': 0, 
        'math': 0, 
        'implementation': 0, 
        'data structures': 0,
        'dp': 0,
        'constructive algorithms': 0,
        'graphs': 0, 
        'geometry': 0
    }
    total_rating = 0
    for index, row in df_subms.iterrows():
        tags = row['tags']
        prob_rating = row['rating']
        for tag in tags:
            if tag in points.keys():
                # adjust
                arg = (prob_rating - user_rating) / 300
                points[tag] += np.exp(arg)
                total_rating += np.exp(arg)
    total_rating = max(total_rating, 0.1)
    for key in points.keys():
        points[key] /= total_rating
    return points


def solve():
    users = get_data()
    stats_by_user = {} 
    users_by_contest_id = {}
    rating_by_user = {}

    cnt_processed_users = 0
    for index, row in users.iterrows():
        handle = row['handle']
        rating = row['rating']
        rating_by_user[handle] = rating
        stats_by_user[handle] = calc_user_stats(handle, rating)
        rating_list = get_response('user.rating', {'handle': handle})
        for contest in rating_list:
            cur_val = users_by_contest_id.get(contest['contestId'], [])
            cur_val.append(handle)
            users_by_contest_id[contest['contestId']] = cur_val

        cnt_processed_users += 1
        print ('calced user stats: ' + str(cnt_processed_users) + ' out of ' + str(users.shape[0]))

    dataset = []

    cnt_processed_contests = 0
   
    for contest_id in users_by_contest_id.keys():
        handles = ';'.join(users_by_contest_id[contest_id])
        resp = get_response('contest.standings', {'contestId': contest_id, 'handles': handles, 'showUnofficial': 'false'})
        if resp['contest']['type'] not in ['CF', 'ICPC']:
            continue
        if resp['contest']['startTimeSeconds'] + 365*24*60*60 // 2 < time.time():
            continue
        rows = resp['rows']
        problems = resp['problems']
        problems_info = []
        for p in problems:
            problems_info.append({
                'contest_id': contest_id,
                'problem_index': p['index'],
                'problem_name': p['name'],
                'rating': p['rating'],
                'is_strings': 0, 
                'is_math': 0, 
                'is_implementation': 0, 
                'is_data structures': 0,
                'is_dp': 0,
                'is_constructive algorithms': 0,
                'is_graphs': 0, 
                'is_geometry': 0
            })

            for tag in p['tags']:
                if (('is_' + tag) in problems_info[-1].keys()):
                    problems_info[-1]['is_' + tag] = 1

        for row in rows:
            party = row['party']
            member = ''
            if len(party['members']) != 1:
                continue
            else:
                member = party['members'][0]['handle']

            problemResults = row['problemResults']
            for i in range(len(problems)):
                # empty dict for a row
                dataset.append({'handle': member, 'user_rating': rating_by_user[member]})
                for key, value in stats_by_user[member].items():
                    dataset[-1]['skill_' + key] = value
                for key, value in problems_info[i].items():
                    dataset[-1][key] = value
                if float(problemResults[i]['points']) > 0.1:
                    dataset[-1]['is_solved'] = 1
                else:
                    dataset[-1]['is_solved'] = 0
        cnt_processed_contests += 1
        print ('contests processed: ' + str(cnt_processed_contests) + ' out of ' + str(len(users_by_contest_id)))

    df = pd.DataFrame(dataset)
    print (df)

    fout = open('resp.out', 'w')
    fout.write(str(dataset))
    df.to_csv('dataset.csv')

def prepare_data():
    df = pd.read_csv('dataset.csv')
    df = df.drop(df.columns[0], axis=1)
    df = df.drop('handle', axis=1)
    df = df.drop('problem_index', axis=1)
    df = df.drop('problem_name', axis=1)
    df = df.drop('contest_id', axis=1)
    df['user_rating'] = df['user_rating'] / 4000
    df['rating'] = df['rating'] / 4000
    df.to_csv('normalized_dataset.csv', index=False)

#solve()
#prepare_data()

def solve():
    df = pd.read_csv('normalized_dataset.csv')
    y = df['is_solved']
    x = df
    x = x.drop('is_solved', axis=1)

    x = pd.DataFrame(df['user_rating'])
    x['rating'] = df['rating']


    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=0)

    #lr = LogisticRegressionCV()
    #lr.fit(train_x, train_y)
    #pred_y = lr.predict(test_x)
    #print("Logistic regression accuracy = {:.12f}".format(lr.score(test_x, test_y)))
    
    #tmpdf = pd.DataFrame(test_y)
    #tmpdf['predicted'] = pred_y
    #print(tmpdf)

    #linreg = LinearRegression().fit(train_x, train_y)
    #pred_y = linreg.predict(test_x)
    #print("Linear regression accuracy = {:.12f}".format(linreg.score(test_x, test_y)))

    #tmpdf = pd.DataFrame(test_y)
    #tmpdf['predicted'] = pred_y
    #print(tmpdf)

    model = Sequential([
        Dense(64, input_shape=(x.shape[1],)),
        Activation('relu'),
        Dense(12),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid'),
    ]) 

    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(train_x, train_y, verbose=0, epochs=12)
    loss, accuracy=model.evaluate(test_x, test_y, verbose=0)
    print("keras accuracy = {:.12f}".format(accuracy))

    tmpdf = pd.DataFrame(test_y)
    tmpdf['predicted'] = model.predict(test_x)
    tmpdf['predicted'] = tmpdf['predicted'].round()
    tmpdf['diff'] = tmpdf['is_solved'].sub(tmpdf['predicted'], axis=0)
    tmpdf['diff'] = tmpdf['diff'].abs()
    print(tmpdf)
    print('sum: ' + str(tmpdf['diff'].sum()) + ', total: ' + str(tmpdf.shape[0]))

    #inputs = []
    #for i in range(18):
    #    inputs.append([0] * 18)
    #    inputs[-1][i] = 1
    #print(model.predict(pd.DataFrame(inputs)))
 

solve()

