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
# we guessian formula to find Likehood p(x1/y) , p(x2/y) , p(x3/y)



#code
#p(y/X)=log(p(x1/y))+log(p(x2/y))+log(p(x3/y))+log(p(y))
#p(y='yes'/X) , p(y='no'/X) ...
#p(x=1,2,../y)=1/sqrt(2*pi*SIGMAy) . e^-((xi-MEANy))/(2*SIGMAy)




import numpy as np


class NaiveBayes:

    def fit(self,X,y):
        n_sam,n_feas=X.shape
        self.y_classes=np.unique(y)
        self.yclss=[i for i in self.y_classes]
        n_clss=len(self.y_classes)     

        #init mean, var, priors
        self.mean=np.zeros((n_clss,n_feas),dtype=np.float64) #mean of all features corresponding to y values
        self.var=np.zeros((n_clss,n_feas),dtype=np.float64)
        self.priors=np.zeros(n_clss,dtype=np.float64)    #frequency_probability of y values p(y) i.e [p(yes),p(no)]

        vly=0
        for clss in self.y_classes:
            c=X[clss==y]
            self.mean[vly,:]=c.mean(axis=0)                 #sum(samples(X=x1,x2../y=yes))/len(p(X/y=yes)) and sum(samples(X=x1,x2../y=no))/len(p(X/y=no))
            self.var[vly,:]=c.var(axis=0)                   #var(samples(X=x1,x2../y=yes)) or var(samples(X=x1,x2../y=no))
            self.priors[vly]=c.shape[0]/float(n_sam)        #p(y)=p(y=yes)/total_samples,p(y=no)/total_samples,...
            vly+=1








        self.X_unic=[]
        y_=[0 for i in self.y_classes]
        if len(self.y_classes)!=2:
            for i in X:
                xu=np.unique(i)
                for j in xu:
                    if not j in self.X_unic:
                        self.X_unic.append(j)   
        self.all_u=[]
        self.X_unic_prob=[]
        self.y_unic_prob=[]     #priors p(y=yes) & p(y=no) ....
        tot_cls=[]
        alx=[]                  #[x1,x2,x3]
        aly=[]                       #[all y values for above one]
        if len(self.y_classes)!=2:
            for i in self.yclss:
                vvv=sum([j==i for j in y])
                self.y_unic_prob.append(vvv/len(y))
                tot_cls.append(vvv)
            Xxx=X.transpose()
            for i in Xxx:
                for j in i:
                    alx.append(j)
            for i in range(n_feas):
                for j in y:
                    aly.append(j)
            for i in self.X_unic:
                un=[]
                for t in range(len(self.y_classes)):
                    ty=0
                    for k in range(len(aly)):
                        if alx[k]==i and aly[k]==self.y_classes[t]:
                            ty+=1
                    y_[t]=ty                                               
                vg=[g for g in y_]
                self.all_u.append(vg)
            for j in self.all_u:
                self.X_unic_prob.append([round(j[t]/tot_cls[t],3) for t in range(len(tot_cls))])









        
        
    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return y_pred


    def _predict(self,x):
        Positerior_probability=[]

        for i,clss in enumerate(self.y_classes):
            if len(self.y_classes)==2:
                prior=np.log(self.priors[i])
            else:
                prior=np.log(self.y_unic_prob[i])
            conditional_propability=np.sum(np.log(self._pdf(i,x))) #sum(p(x=1,2,../y)) , p(y/X)=log(p(x1/y))+log(p(x2/y))+log(p(x3/y))+log(p(y))
            posterior=conditional_propability+prior
            Positerior_probability.append(posterior)
            
        hgh=[[self.y_classes[i],Positerior_probability[i]]for i in range(len(Positerior_probability))]
        
        return self.y_classes[np.argmax(Positerior_probability)],hgh  #argmax means argument(position value) of max value


    def _pdf(self,clss_i,xx):
        if len(self.y_classes)==2:
            mean_=self.mean[clss_i]     #mean of X's for y=yes and mean of X for y=no
            var_=self.var[clss_i]       #var of X's for y=yes and var of X for y=no
            num=np.exp(-(xx-mean_)**2/(2*var_))
            denum=np.sqrt(2*np.pi*var_)                #p(x=1,2,../y)=1/sqrt(2*pi*SIGMAy) . e^-((xi-MEANy)^2)/(2*SIGMAy)
            return num/denum                           #returns p(x=1,2../y)
        else:
            px=[self.X_unic_prob[self.X_unic.index(i)][clss_i] for i in xx]
            return px
            
            
            
        
        





