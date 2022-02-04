#p(x/y) = p(x∩y)/p(y)
#p(y/x) = p(x∩y)/p(x)
#p(x∩y)=p(x∩y)
#p(x/y).p(y)=p(y/x).p(x)
#we want to predict probability y by given x features, So
#p(y/x)=p(x/y).p(y)/p(x)
#we can negilite p(x) because it is contant for all y predicting probability


# finally Naive Bayes formula is p(y/X)=p(X/y).p(y) where X is set of features ,so
# p(y/X)=p(x1/y).p(x2/y).p(x3/y).p(y) for 3 features         ---------------------------- 3
# p(y/X)-> Positerior probability , p(X/y)-> Likehood or conditional propability , p(y)-> Prior

# Example
# use Naive Bayes formula to find each p(y='yes'/X) , p(y='no'/X) ... for all y values
# Highest probability among above mentioned is consider as predicted output
# from ----3 we can see that multipling many probablities which (exits only from 0 to 1(0.067)) is very small numbers leds to very very small number after multiplication
# so we are going to take the log which coverts the multiplication into addtion
# from ----3 p(y/X)=p(x1/y).p(x2/y).p(x3/y).p(y) after log p(y/X)=log(p(x1/y))+log(p(x2/y))+log(p(x3/y))+log(p(y))



#code
#p(y/X)=log(p(x1/y))+log(p(x2/y))+log(p(x3/y))+log(p(y))
#p(y='yes'/X) , p(y='no'/X) ...





import numpy as np


class NaiveBayes:

    def fit(self,Xx,Yy):
        n_sam,n_feas=Xx.shape
        self.y_classes=np.unique(Yy)
        self.yclss=[i for i in self.y_classes]
        n_clss=len(self.y_classes)
        self.X_unic=[]
        y_=[0 for i in self.y_classes]
        self.all_u=[]
        for i in Xx:
            xu=np.unique(i)
            for j in xu:
                if not j in self.X_unic:
                    self.X_unic.append(j)
        X=[]
        for i in Xx:
            for j in i:
                X.append(j)
        y=[]
        for i in range(n_sam):
            for j in Yy:
                y.append(j)
        
        yu=0
        for i in self.X_unic:
                un=[]
                for t in self.y_classes:
                    ty=0
                    for k in range(len(y)):
                        if X[k]==i and y[k]==t:
                            ty+=1
                    y_[self.yclss.index(t)]=ty
                vg=[g for g in y_]
                self.all_u.append(vg)
       
       
        self.X_unic_prob=[]
        self.y_unic_prob=[]     #priors p(y=yes) & p(y=no) ....
        tot_cls=[]

        for i in self.yclss:
            vvv=sum([j==i for j in y])
            self.y_unic_prob.append(vvv/len(y))
            tot_cls.append(vvv)
        
        for j in self.all_u:
            self.X_unic_prob.append([round(j[t]/tot_cls[t],3) for t in range(len(tot_cls))])
        
                        
        
        
    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return y_pred


    def _predict(self,x):
        Positerior_probability=[]

        for i,clss in enumerate(self.y_classes):
            prior=np.log(self.y_unic_prob[i])
            conditional_propability=np.sum(np.log(self._pdf(i,x))) #sum(p(x=1,2,../y)) , p(y/X)=log(p(x1/y))+log(p(x2/y))+log(p(x3/y))+log(p(y))
            posterior=conditional_propability+prior
            Positerior_probability.append(posterior)
        hgh=[[self.y_classes[i],Positerior_probability[i]]for i in range(len(Positerior_probability))]

        return self.y_classes[np.argmax(Positerior_probability)],hgh  #argmax means argument(position value) of max value


    def _pdf(self,y_value,xx):
        px=[self.X_unic_prob[self.X_unic.index(i)][y_value] for i in xx]
        return px
        
        
        


   
            
            
            
        
        





