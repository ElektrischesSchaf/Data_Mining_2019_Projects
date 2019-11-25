import numpy as np
import random
import matplotlib as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

#declare parameter
# attribute/feature
ATT = ['attri_0','attri_1','attri_2','attri_3','attri_4',
'attri_5','attri_6', 'attri_7', 'attri_8', 'attri_9', 
'attri_10', 'attri_11', 'attri_12', 'attri_13', 'attri_14',
'attri_15', 'attri_16', 'attri_17', 'attri_18', 'attri_19', 'attri_20']
DATA_NUM = [100, 500, 1000, 2500, 5000, 10000, 20000]

#generator
def GenerateOuptputFrom(i):
        if i[0] == True:
            if i[2] == True:
                if i[11]==1:
                    return True
                else:
                    if i[12] == True:
                        return True
                    else:
                        return False
            else:
                if i[1] == True:
                    if i[13]==1:
                        return True
                    else:
                        if i[14]== True:
                            return True
                        else:
                            return False
                else:
                    if i[3] == True:
                        return True
                    else:
                        return False
        else:
            if i[3] == True:
                if i[20] == True:
                    return True
                else:
                    if i[2] == True:
                        if i[6] == True:
                            return True
                        else:
                            if i[7] == True:
                                return True
                            else:
                                if[8] == True:
                                    return True
                                else:
                                    return False
                    else:
                        if i[5] == True:
                            if i[9] == True:
                                if[10] == True:
                                    return True
                                else:
                                    return False
                            else:
                                if i[18]== True:
                                    if i[19]== True:
                                        return True
                                    else:
                                        return False
                                else:
                                    return False
                        else:
                            return False
            else:
                if i[15]== True:
                    if i[16]== True:
                        return True
                    else:
                        return False
                else:
                    return False                        

def data_generator(num_of_data):
    data = []
    for _ in range(num_of_data):
        tmp = []
        for i in range(len(ATT)):
            boolean = random.choice([True, False])
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
    test_depth_start=3
    test_depth_end=6
    for i in range(test_depth_start, test_depth_end+2):
        row1 = []
        row2 = []

        # test different input number
        tmp_model = ''
        for num in DATA_NUM:
            #print(num)
            data = data_generator(num)
            if i != test_depth_end+1:
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
            #f = tree.export_graphviz(model[i], feature_names=ATT, out_file=f )
            tree.export_graphviz(model[i], feature_names=ATT, out_file=f )
            #graph=graphviz.Source(f)
            #graph.render('./models/decision_tree_depth'+str(i+3))