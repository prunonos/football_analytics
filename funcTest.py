import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.model_selection import GridSearchCV
# funcion para normalizar/estandarizar DB

def normalDB(db,scaler=MinMaxScaler(), col_res='res',**conf):
    
    db = db.dropna()
        
    res  = list(db[col_res])  # tiene que ser list() porque sino se guardan los indices también
    data = db.drop(col_res,axis=1)
    cols = data.columns
    
    assert len(db) == len(res) == len(data)
    
    scaler.fit(data)
    data = scaler.transform(data)
    
    data = pd.DataFrame(data,columns=cols)
    data[col_res] = res
    
    return data

def preprocessDB(db,frac=0.85,col_res='res',seed=1):
    db = db.sample(frac=1,axis=0,random_state=seed)
    index = int(len(db)*frac)
    y = db[col_res]
    X = db.drop(col_res,axis=1)
    trainX = X[:index]
    testX  = X[index:]
    trainY = y[:index]
    testY  = y[index:]
    
    trainDB = db[:index]

    assert len(trainX) == len(trainY)
    assert len(testX) == len(testY)
    #assert (len(trainX) + len(testX)) == len(trainDB)
    
    return trainDB,trainX,trainY,testX,testY

# visualizar el árbol
def tree_png(c,c_names,ft_names):
    from sklearn.tree import export_graphviz
    from six import StringIO  
    from IPython.display import Image  
    import pydotplus

    dot_data = StringIO()
    export_graphviz(c, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = ft_names,class_names=c_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('diabetes.png')
    return Image(graph.create_png())

def calcRES(db):
    res = []
    for h,a in zip(db['score_home'],db['score_away']):
        if h>a:
            res.append(-1)
        elif a>h:
            res.append(1)
        else:
            res.append(0) 
    
    assert len(res) == len(db)
    
    return res

def medir_tiempo(f):
    def funcion_medida(*args, **kwargs):
        inicio = time.time()
        c = f(*args, **kwargs)
        print("{:.3f}".format(time.time() - inicio))
        return c
    return funcion_medida

@medir_tiempo
def train_wGridSearch(clf,trainX,trainY,param,cv=5,return_train_score=True):
    """
    Función para entrenar con hiperparámetros óptimos a partir de una busqueda por rejilla
    """
    clf_grid = GridSearchCV(estimator=clf, param_grid=param, cv=cv, return_train_score=return_train_score)
    if cv == -1:
        score = clf_grid.fit(trainX,trainY)
    else:
        score = cross_validate(clf_grid,X=trainX,y=trainY, cv=5, return_estimator=True)    
    return score

@medir_tiempo
def train_wCrossVal(clf,trainX,trainY,cv=5):
    score = cross_validate(clf,X=trainX,y=trainY, cv=5, return_estimator=True)    
    return score