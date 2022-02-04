import numpy as np
# two line wx1+b=-1 and wx2+b=+1 we want to find maxium distance between this
# distance (wx2+b=+1) - (wx2+b=+1)  = x2-x1 = 2 / ||w|| maximize this leads to maximing distance or minimize ||w||/2 because we derivate it and get gradient descent 
#loss functio=(lamda*||w||^2) +    1/n(sum(max(0,1-yi(w*xi+b)))       => hinge loss (sum(max(0,1-yi(w*xi+b)) 0 for correct output 1 for wroung output, note yi(w*xi+b) gives positive for correct output
#                   |-> margin between two classes should be large  and   |-> classify datapoints at good seperation
#if yi(w*xi+b)>=1  (lamda*||w||^2)
#else (lamda*||w||^2)+1-yi(w*xi+b)
                                 
class SVM:
    def __init__(self,LearningRate=0.001,c=0.01,niters=2000):
        self.lr=LearningRate
        self.ntrs=niters
        self.w=None
        self.b=None
        self.lamb_par=c                               #c = How many errors can my model can consider means mis_classification
        
    def fit(self,X,y):
        y_=np.where(y<=0,-1,1)
        n_smpls,n_features=X.shape
        self.w=np.zeros(n_features)
        self.b=0
        for i in range(self.ntrs):
            for index,x_val in enumerate(X):
                con=y_[index]*((np.dot(x_val,self.w)+self.b))>=1
                if con:
                    self.w -=self.lr*(2*self.lamb_par*self.w)        #here hinge loss is 0 (sum(max(0,1-yi(w*xi+b))) predicted correctly so loss is 0
                else:
                    self.w -=self.lr*(2*self.lamb_par*self.w - np.dot(x_val,y_[index]))
                    self.b -=self.lr*(-1*(y_[index]))



    def predict(self,X):
        y_pred=np.dot(X,self.w)+self.b
        return np.sign(y_pred)
    


























































































































"""        for i in range(self.ntrs):
            for index,x_val in enumerate(X):
                con=y_[index]*((np.dot(x_val,self.w)-self.b))>=1
                if con:
                    self.w -=self.lr*(2*self.lamb_par*self.w)
                else:
                    self.w -=self.lr*(2*self.lamb_par*self.w - np.dot(x_val,y_[index]))
                    self.b -=self.lr*y_[index]
"""
