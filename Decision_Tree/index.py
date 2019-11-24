import numpy as np
import random
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

#declare parameter
# attribute/feature
ATT = ['face','rich','height','personality','warm','social','exercise']
DATA_NUM = [100, 500, 1000, 2500, 5000, 10000]

#generator
def GenerateOuptputFrom(i):
        if i[0] == 1:
            if i[2] == 1:
                return 0
            else:
                if i[1] == 1:
                    return 0
                else:
                    if i[3] == 1:
                        return 0
                    else:
                        return 1
        else:
            if i[3] == 0:
                return 1
            else:
                if i[1] == 1:
                    return 0
                else:
                    if i[2] == 1:
                        if i[5] == 1:
                            return 0
                        else:
                            if i[4] == 1:
                                return 0
                            else:
                                if[6] == 1:
                                    return 0
                                else:
                                    return 1
                    else:
                        if i[5] == 1:
                            if i[4] == 1:
                                if[6] == 1:
                                    return 0
                                else:
                                    return 1
                            else:
                                return 0
                        else:
                            return 1
                        

def data_generator(num_of_data):
    data = []
    for _ in range(num_of_data):
        tmp = []
        for i in range(len(ATT)):
            boolean = random.choice([1, 0])
            tmp.append(boolean)
        output = GenerateOuptputFrom(tmp)
        tmp.append(output)
        #last item is its outcome
        data.append(tmp)
    return data

def build_decision_tree_model(data, depth=None):
    data = np.array(data)
    Y = data[:, -1]
    X = data[:, :-1]
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3)
    clf1 = tree.DecisionTreeClassifier( criterion='gini', max_depth=depth, random_state=0)
    clf2 = tree.DecisionTreeClassifier( criterion='entropy', max_depth=depth, random_state=0)
    Clf_model1 = clf1.fit(train_x,train_y)
    Clf_model2 = clf2.fit(train_x,train_y)
    return Clf_model1, Clf_model2, test_x, test_y

def evaluation(model1, model2, test_x, test_y):
    return model1.score(test_x, test_y), model2.score(test_x, test_y)

# main process 
table1 = []
table2 = []
model = []

if __name__=="__main__":
    # test different depth from 3 ~ 6 of tree
    for i in range(3, 8):
        row1 = []
        row2 = []

        # test different input number
        tmp_model = ''
        for num in DATA_NUM:
            #print(num)
            data = data_generator(num)
            if i != 7:
                m1, m2, x, y = build_decision_tree_model(data, i)
                tmp_model = m1
            else:
                m1, m2, x, y = build_decision_tree_model(data)
                tmp_model = m1
            accuracy1, accuracy2 = evaluation(m1, m2, x, y)
            row1.append(accuracy1)
            row2.append(accuracy2)
        model.append(tmp_model)
        table1.append(row1)
        table2.append(row2)

    #gini talbe
    result_table_1 = np.array(table1)
    output1 = pd.DataFrame(data=result_table_1, index=[3, 4, 5, 6, 'no limit'], columns=DATA_NUM)
    print(output1.head())

    #entropy table
    result_table_2 = np.array(table2)
    output2 = pd.DataFrame(data=result_table_2, index=[3, 4, 5, 6, 'no limit'], columns=DATA_NUM)
    print(output2.head())

    # store model in file
    for i in range(len(model)):
        with open('./models/decision_tree_depth'+str(i+3)+'.dot','w') as f:
            f = tree.export_graphviz(model[i],feature_names=ATT,out_file=f )