from sklearn.ensemble import GradientBoostingRegressor as GB
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor as DTR
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor as KNN

from scipy.stats.mstats import normaltest
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model as LM
from sklearn.decomposition import PCA


class PCAAnalysis():
    def __init__(self, data):
        # @FORMAT: data = df(X1, X2, X3, etc, index=dates)
        self.data = data
        self.X = self.data.values

    def getPCA(self,n_components):

        data = self.data
        X = data.values
        pca = PCA(n_components)
        pca.fit(X)
        newdata = pca.fit_transform(X)
        newdata = pd.DataFrame(newdata)
        self.pca_components = pca.components_

        self.pca_explained_var = pca.explained_variance_ratio_
        print('VARIANCE EXPLAINED', pca.explained_variance_ratio_)

        #covar_o = np.cov(np.transpose(X.values))
        #eigval_o, eigvec_o = np.linalg.eig(covar_o)
        #print(eigvec_o)
        #print(eigval_o)
        #covar = np.cov(np.transpose(newdata.values))
        #eigval, eigvec = np.linalg.eig(covar)
        #print(eigvec)
        #print(eigval)

    def getComponents(self):
        columns =  self.data.columns.values
        return pd.DataFrame(self.pca_components, columns=columns)

    def createLinearComponent(self):
        #Need to run getPCA() first
        a = self.X
        b = np.transpose(self.pca_components)
        c = np.dot(a,b)

        #Returns the linear component based on the loadings of each component generated by getPCA
        self.linearComponents = c
        print(c)
        return c

    #Creates a LM model and returns the prediction vector for the X_Pred
    def createLM(self,X,Y,X_Pred):
        lm = LM.LinearRegression(fit_intercept=True,copy_X=True)
        lm.fit(X,Y)
        Y_Pred = lm.predict(X_Pred)
        return Y_Pred

def write(datadf, path, tab="Sheet1"):
    WRITE_PATH = path
    writer = pd.ExcelWriter(WRITE_PATH, engine='xlsxwriter')
    datadf.to_excel(writer, sheet_name=tab)
    writer.save()

def getBlendedSignal(data,ml_model, gap=60,**kwargs):
    #@FORMAT: data = df(Y,X1,X2...,index=dates), dates goes from earliest to latest
    dates = data.index.values
    Y = data.iloc[:,0].values
    X = data.drop(data.columns[[0]],axis=1).values
    out = []

    for i in range(gap,X.shape[0],1):
        X_ = X[(i-gap):i]
        Y_ = Y[(i-gap):i]
        X_test = X[i]
        X_test = X_test.reshape(1,-1)
        Y_test = Y[i]

        #model = ml_model(**kwargs)
        model = ml_model()
        model.fit(X_, Y_)

        pred = model.predict(X_test)
        out.append([dates[i],Y_test,pred[0]])

    #@RETURNS: [date, Y, Y_pred]
    return out

def getGrid(data, target, model, gridCV):
    Y = data[target]
    X = data.drop(target,axis=1)
    clf = GridSearchCV(model, gridCV)
    clf.fit(X,Y)
    return clf
def getSVCLinear(data, target, kern='linear'):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = svm.SVC(kernel=kern)
    model.fit(X,Y)
    return model
def getSVCRbf(data, target, kern='rbf'):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = svm.SVC(kernel=kern)
    model.fit(X,Y)
    return model

#Because of Greedy algorithm, favors short trees over complex tall trees.
#Overfitting - if there exists another hypothesis that performs worse in training but better throught out the dataset as a whole and beyond.  You will see divigence of accuracy as the tree grows in length.
def getDTC(data, target, depth):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = DTR(max_depth=depth)
    return model


#Good for a complex problem with a collection of simpler solutions.
#Nearly all computations happen during classfication time, not on the training - could increase runtime.
#Curse of dimensionality - if a lot of attributes are non-useful, it will throw off the algorithm which use ALL attributes to determine 'closeness'

def getKNN(data,target, N=27):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = KNN(n_neighbors=N)
    model.fit(X,Y)
    return model

def loopKNNonN(data, target, max_n, N_FOLDS):
    output = []
    for i in range(1,max_n+1):
        cur_model = getKNN(data, target,i)
        X = data.drop(target,axis=1)
        Y = data[target]
        cur_train_score = cross_val_score(cur_model,X, Y, cv=N_FOLDS)
        out = [cur_train_score.mean(),cur_train_score.std()]
        output.append(out)
    output = pd.DataFrame(output)
    return output
#loopN = loopKNNonN(train, targetName, 40, N_FOLDS)
#print(loopN)






bmi = pd.ExcelFile("AAPL.xlsx")
data = bmi.parse("Python_Input")

data = data.dropna()

input = data.drop(['Open'], axis=1)
print(input)

'''
#DECISION TREES
fcsts = getBlendedSignal(input, DTR, gap=20)
fcsts = pd.DataFrame(fcsts, columns=['date', 'Y', 'Y_Pred'])
#print('fcsts', pd.DataFrame(fcsts))

write(fcsts, 'DTR_Results_20.xlsx','sheet1')

#GRADIENT BOOSTING
fcsts = getBlendedSignal(input, GB, gap=20)
fcsts = pd.DataFrame(fcsts, columns=['date', 'Y', 'Y_Pred'])
#print('fcsts', pd.DataFrame(fcsts))

write(fcsts, 'GB_Results_20.xlsx','sheet1')
'''

#GET PCA COMPONENTS
input =  data.drop(['Open','Return_1D'], axis=1)
pca = PCAAnalysis(input)
pca.getPCA(n_components=3)
comp = pca.getComponents()
lin_comp = pca.createLinearComponent()
lin_comp['1d_Returns'] = data['Return_1D']
comp_out = pd.DataFrame(lin_comp, columns=['pc1', 'pc2', 'pc3'], index=input.index.values)
print('comp',comp_out)
write(comp_out, 'pca_components.xlsx','sheet1')



params_svc_lin = [{'kernel':['linear'],'C':[1,10,50,100]}]
params_svc_rbf =  [{'kernel':['rbf'],'C':[1,10,50,100],'gamma':[0.001,.005,'auto',0.2,0.5,1,5]}]
params_boosting =  [{'kernel':['rbf'],'C':[1,10,50,100]}]
params_knn =  [{'kernel':['rbf'],'C':[1,10,50,100]}]

'''
tmp_model = svm.SVC(kernel='linear') #CHANGE
grid =  getGrid(train, targetName,tmp_model, params_svc_lin) #CHANGE
cur_model=grid
print('Grid Search Results')
print(grid.best_params_)
print(grid.best_score_)
'''



'''
model = svm.SVC(kernel='rbf',C=100, gamma=1)
model.fit(X,Y)
pred=model.predict(test.drop(targetName,axis=1))


model = KNN(n_neighbors=18,weights='distance',p=2)
model.fit(X,Y)
pred=model.predict(test.drop(targetName,axis=1))

'''
