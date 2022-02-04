import numpy as np
from copy import deepcopy
from matplotlib import cm,pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from svm import SVM
from sklearn.svm import SVC
from Naive_bayes_string import NaiveBayes
from Naive_Bayes import NaiveBayes as NB
from fuzzyClustering import fuzzyClustering
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from kmeans import KMeans
from PCA import PCA
from sklearn.decomposition import KernelPCA
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,roc_auc_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from PIL import Image
import cv2
import base64
import io

def linear(x,Y,pre):
        try:
                
                output=""
                x1=list(map(float,x.split(',')))
                y=list(map(float,Y.split(',')))
                if pre!="":
                        pred=[[i] for i in list(map(float,pre.split(',')))]

                        
                X=np.array([x1])
                X=X.transpose()
                y=np.array(y)

                model=LinearRegression()

                model.fit(X,y)

                y_pred=model.predict(X).round(2)

                
                output=output+"predicted y value of input x:\t"+str(y_pred.tolist())+"\n\n"
                if pre!="":
                        up=model.predict(pred).round(2)
                        output=output+"predicted y value of User's input x:\t"+str(up.tolist())+"\n\n"
                test=0
                for i in range(len(x1)):
                        test+=(y[i]-y_pred[i])**2
                output=output+"Least square test value:"+str(round(test,3))+"\n\n"

                e=0
                for i in range(len(x1)):
                        if round(y_pred[i])==y[i]:
                            e+=1
                output=output+"Predicted"+" "+str(e)+" "+"correctly out of"+" "+str(len(y))+"\n\n"
                output=output+"Intercept:"+str(model.intercept_.round(3))+"\n\n"
                output=output+"Coeffients:"+str(model.coef_.round(3))+"\n"
                fig=plt.figure(facecolor='lightgreen')
                ax=plt.axes()
                ax.set_facecolor('lightgreen')
                ay=fig.add_subplot(1,1,1)
                plt.title("Linear regression")
                ay.scatter(x1,y)
                ay.plot(x1,y_pred)
                fig.canvas.draw()
                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                pil_im=Image.fromarray(img)
                buff=io.BytesIO()
                pil_im.save(buff,format="PNG")
                img_str=base64.b64encode(buff.getvalue())
                return output,""+str(img_str,'utf-8')
        except:
                output="Something went wrong... Please give correct input"
                return output,"error"



def mullinear(Xs,Y,pre):
        try:
                output=""
                b=Xs.split(';')
                x1=[list(map(float,i.split(','))) for i in b]
                y=list(map(float,Y.split(',')))
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]

                 
                X=np.array(x1)
                X=X.transpose()
                y=np.array(y)

                model=LinearRegression()

                model.fit(X,y)

                y_pred=model.predict(X).round(2)

                
                output=output+"predicted y value of input x:\t"+str(y_pred.tolist())+"\n\n"
                if pre!="":
                        up=model.predict(pred).round(2)
                        output=output+"predicted y value of User's input x:\t"+str(up.tolist())+"\n\n"
                test=0
                for i in range(len(x1)):
                        test+=(y[i]-y_pred[i])**2
                output=output+"Least square test value:"+str(round(test,3))+"\n\n"

                e=0
                for i in range(len(x1)):
                        if round(y_pred[i])==y[i]:
                            e+=1
                output=output+"Predicted"+" "+str(e)+" "+"correctly out of"+" "+str(len(y))+"\n\n"
                output=output+"Intercept:"+str(model.intercept_.round(3))+"\n\n"
                output=output+"Coeffients:"+str(model.coef_.round(3))+"\n"
                
                return output,"summa"
        except:
                output="Something went wrong... Please give correct input"
                return output,"error"


def polylinear(X,Y,dox,pre):
        try:    
                output=""
                xx=list(map(float,X.split(',')))
                yy=list(map(float,Y.split(',')))
                degree=int(dox)
                if pre!="":
                        predic=list(map(float,pre.split(',')))
                x=[]
                y=[]
                for i in xx:x.append([i])
                for i in yy:y.append(i)
                x_test=[]
                y_test=[]
                for i in xx:x_test.append([i])
                for i in yy:y_test.append(i)
                if pre!="":
                        for i in predic:
                                x_test.append([i])
                poly=PolynomialFeatures(degree=degree)
                x_poly=poly.fit_transform(x)


                model=LinearRegression()
                model.fit(x_poly,y)


                
                u=model.predict(poly.fit_transform(x_test))
                output=output+"predicted y value of input x:\t"+str(u[0:len(y)].round(2).tolist())+"\n\n"
                if pre!="":
                        output=output+"predicted y value of User's input x:\t"+str(u[len(y):].round(2).tolist())+"\n\n"
                test=0

                for i in range(len(y_test)):
                        test+=(y_test[i]-round(u[i]))**2
                output=output+"Least square test value:"+str(round(test,3))+"\n\n"


                e=0
                for i in range(len(y_test)):
                        if round(u[i])==y_test[i]:
                            e+=1
                output=output+"Predicted"+" "+str(e)+" "+"correctly out of"+" "+str(len(y))+"\n\n"



                output=output+"Intercept:"+str(model.intercept_.round(3))+"\n\n"
                output=output+"Coeffients:"+str(model.coef_.round(3))+"\n"
                fig=plt.figure(facecolor='lightgreen')
                ax=plt.axes()
                ax.set_facecolor('lightgreen')
                ay=fig.add_subplot(1,1,1)
                plt.title("Polynomial regression")
                plt.scatter(x,y,color='red')
                ay.plot(x,model.predict(x_poly),color='blue')
                fig.canvas.draw()
                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                pil_im=Image.fromarray(img)
                buff=io.BytesIO()
                pil_im.save(buff,format="PNG")
                img_str=base64.b64encode(buff.getvalue())
                return output , ""+str(img_str,'utf-8')
        except:
                output="Something went wrong... Please give correct input"
                return output,"error"



def logreg(Xs,Y,tt,stt,ts,rs,ccv,pre,thres):
        try:
                
                b=Xs.split(';')
                x1=[list(map(float,i.split(','))) for i in b]
                yy=list(map(float,Y.split(',')))
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]
                
                if thres!="":
                        thershold=float(thres)
                else:
                        thershold=0.5
                train_test_split_validation=tt
                if ccv!="":
                        cv=int(ccv)
                else:
                        cv=0
                if stt==1:
                        stratifyy=''
                else:
                        stratifyy='y'
                if ts!="":
                        test_size=int(ts)
                else:
                        test_size=2
                if rs!="":
                        random_state=int(rs)
                else:
                        random_state=0
                


                X=np.array(x1)
                y=np.array(yy)
                if len(np.unique(y))>cv or cv>len(y) or cv==1 or cv==0:
                    cv=len(np.unique(y))
                X=X.transpose()
                if test_size>len(y) or test_size==0 or len(np.unique(y))>test_size :
                        test_size=len(np.unique(y))
                if stratifyy=='y':
                    stratify=y
                else:
                    stratifyy=''
                    
                model=LogisticRegression()
                

                Accuracy=[]
                output=""
                if train_test_split_validation==1:
                    output=output+"train_test_split validation\n\n"
                    if stratifyy!='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify)
                    if stratifyy!='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify)
                    if stratifyy=='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,test_size=test_size)
                    if stratifyy=='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
                    if stratifyy=='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state)
                    if stratifyy=='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y)
                    model.fit(X_train,y_train)
                    Y_predicted=model.predict_proba(X_test)[:,1]>thershold
                    output=output+"Predicted "+str(sum((y_test==Y_predicted)))+" Correctly Out of "+str(y_test.shape[0])+"\n\n"
                    output=output+"Predicted "+str(round((sum((y_test==Y_predicted))/y_test.shape[0])*100))+"% Correctly\n\n"
                    cm=confusion_matrix(Y_predicted,y_test,labels=[1,0])
                    if cm[0,0]+cm[0,1]!=cm[0,0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that"+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                    output=output+"So Precision is "+str(cm[0,0]/(cm[0,0]+cm[0,1]))+"\n\n"
                    if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data"+"\n\n"
                    output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"
                    output=output+"Given features and threshold are supporting "+str(round(roc_auc_score(y_test,Y_predicted)*100))+"%"+" to predicte the correct output value"+"\n\n"

                elif train_test_split_validation==2:
                    output=output+"Comes under K-fold cross validation"+"\n\n"
                    kf=KFold(n_splits=cv,shuffle=True)
                    for train_index,test_index in kf.split(X):
                        if yy.count(np.unique(y)[0])>=2 and yy.count(np.unique(y)[1])>=2:
                                X_train,X_test=X[train_index],X[test_index]
                                y_train,y_test=y[train_index],y[test_index]
                                model=LogisticRegression()
                                model.fit(X_train,y_train)
                                Accuracy.append(model.score(X_test,y_test))
                    if yy.count(np.unique(y)[0])>=2 and yy.count(np.unique(y)[1])>=2:
                            output=output+"Accuracy :\t"+str(np.mean(Accuracy))+"\n\n"
                    model=LogisticRegression()
                    model.fit(X,y)
                    Y_predicted=model.predict_proba(X)[:,1]>thershold
                    output=output+"Predicted "+str(sum(y==Y_predicted))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                    cm=confusion_matrix(Y_predicted,y,labels=[1,0])
                    if cm[0,0]+cm[0,1]!=cm[0,0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that"+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                    output=output+"So Precision is "+str(cm[0,0]/(cm[0,0]+cm[0,1]))+"\n\n"
                    if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data"+"\n\n"
                    output=output+"So recall is "+str(cm[0,0]/(cm[0,0]+cm[1,0]))+"\n\n"
                    output=output+"Given features and threshold are supporting "+str(round(roc_auc_score(y,Y_predicted)*100))+"%"+" to predicte the correct output value\n\n"
                    


                elif train_test_split_validation==0:
                    output=output+"Normal validation\n\n"
                    model=LogisticRegression()
                    model.fit(X,y)
                    Y_predicted=model.predict_proba(X)[:,1]>thershold
                    output=output+"Predicted "+str(sum((y==Y_predicted)))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                    output=output+"Predicted "+str(round((sum((y==Y_predicted))/y.shape[0])*100))+"% Correctly\n\n"
                    cm=confusion_matrix(Y_predicted,y,labels=[1,0])
                    if cm[0,0]+cm[0,1]!=cm[0,0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                    output=output+"So Precision is "+str(cm[0,0]/(cm[0,0]+cm[0,1]))+"\n\n"
                    if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                    output=output+"So recall is "+str(cm[0,0]/(cm[0,0]+cm[1,0]))+"\n\n"
                    output=output+"Given features and threshold are supporting "+str(round(roc_auc_score(y,Y_predicted)*100))+"%"+" to predicte the correct output value"+"\n\n"
                   
                fig = plt.figure(facecolor='lightgreen')
                plt.imshow(cm)
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.ylabel('True Label')
                plt.xlabel('Predicated Label')
                model=LogisticRegression()
                model.fit(X,y)
                output=output+"cofficeints:"+str(model.coef_)+"\n\n"
                output=output+"Intercept:"+str(model.intercept_)+"\n\n"
                output=output+"predicted y value of input x:\t"+str(model.predict(X))+"\n\n"
                if pre!="":output=output+"predicted y value of user's input x:\t"+str(model.predict(pred))+"\n\n"
                output=output+"Confusion matrix :"+"\n"
                output=output+"\t\t\t\t\t\t\tpredicted label"+"\n"
                tl=["True","label"]
                k=0
                for i in cm:
                        if k<2:
                                output=output+tl[k]+"\t\t"+str(i)+"\n"
                        else:
                                output=output+"\t\t"+str(i)+"\n"
                        k+=1
                fig.canvas.draw()
                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                pil_im=Image.fromarray(img)
                buff=io.BytesIO()
                pil_im.save(buff,format="PNG")
                img_str=base64.b64encode(buff.getvalue())
                return output , ""+str(img_str,'utf-8')
        except:
                output="Something went wrong... Please give correct input"
                return output,"error"

def knn(Xs,Y,n_niegh,dis,tt,stt,ts,rs,ccv,pre):
        try:
                output=""
                b=Xs.split(';')
                x1=[list(map(float,i.split(','))) for i in b]
                yy=list(map(float,Y.split(',')))
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]
                
                if n_niegh!="":
                        n_neighbors=list(map(int,n_niegh.split(',')))
                else:
                        n_neighbors=2
                train_test_split_validation=tt
                if ccv!="":
                        cv=int(ccv)
                else:
                        cv=0
                if stt==1:
                        stratifyy=''
                else:
                        stratifyy='y'
                if ts!="":
                        test_size=int(ts)
                else:
                        test_size=2
                if rs!="":
                        random_state=int(rs)
                else:
                        random_state=0
                
                train_test_split_validation=tt
                dd=dis



                X=np.array(x1)
                y=np.array(yy)
                X=X.transpose()
                distance_metrics=['euclidean','manhattan','chebyshev','minkowski']
                if len(np.unique(y))<cv or cv>len(yy) or cv==1 or cv==0:
                    cv=2
                if test_size>len(y) or test_size==0 or len(np.unique(y))>test_size :
                    test_size=len(np.unique(y))
                if n_neighbors!=[]:
                    model=KNeighborsClassifier(metric=distance_metrics[dd])
                    btp={'n_neighbors':n_neighbors}
                    gs=GridSearchCV(model,btp,cv=cv)
                    gs.fit(X,y)
                if n_neighbors==[]:
                    n_neighbors=round(len(y)**0.5) if len(y)%2!=0 else round(len(y)**0.5)-1
                else:
                    n_neighbors=gs.best_params_['n_neighbors']
                if stratifyy=='y':
                    stratify=y
                else:
                    stratifyy=''

                    
                model=KNeighborsClassifier(n_neighbors=n_neighbors,metric=distance_metrics[dd])



                Accuracy=[]

                if train_test_split_validation==1:
                    output=output+"Comes under train_test_split validation"+"\n\n"
                    output=output+"best n_neighbors parameter is "+str(gs.best_params_['n_neighbors'])+'\n\n'
                    if stratifyy!='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify)
                    if stratifyy!='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify)
                    if stratifyy=='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,test_size=test_size)
                    if stratifyy=='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
                    if stratifyy=='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state)
                    if stratifyy=='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y)
                    model.fit(X_train,y_train)
                    Y_predicted=model.predict(X_test)
                    output=output+"Predicted "+str(sum((y_test==Y_predicted)))+" Correctly Out of "+str(y_test.shape[0])+"\n\n"
                    output=output+"Predicted "+str(round((sum((y_test==Y_predicted))/y_test.shape[0])*100))+"% Correctly"+"\n\n"
                    cm=confusion_matrix(y_test,Y_predicted)

                elif train_test_split_validation==2:
                    output=output+"Comes under K-fold cross validation"+"\n\n"
                    output=output+"best n_neighbors parameter is "+str(gs.best_params_['n_neighbors'])+'\n\n'
                    kf=KFold(n_splits=cv,shuffle=True)
                    for train_index,test_index in kf.split(X):
                        X_train,X_test=X[train_index],X[test_index]
                        y_train,y_test=y[train_index],y[test_index]
                        model=KNeighborsClassifier(n_neighbors=n_neighbors,metric=distance_metrics[dd])
                        model.fit(X_train,y_train)
                        Accuracy.append(model.score(X_test,y_test))
                    model=KNeighborsClassifier(n_neighbors=n_neighbors)
                    model.fit(X,y)
                    Y_predicted=model.predict(X)
                    output=output+"Predicted "+str(sum((y==Y_predicted)))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                    output=output+"Mean Accuracy :"+str(np.mean(Accuracy))+"\n\n"
                    cm=confusion_matrix(y,Y_predicted)


                elif train_test_split_validation==0:
                    output=output+"Comes under Normal validation\n\n"
                    output=output+"best n_neighbors parameter is "+str(gs.best_params_['n_neighbors'])+'\n\n'
                    model=KNeighborsClassifier(n_neighbors=n_neighbors)
                    model.fit(X,y)
                    Y_predicted=model.predict(X)
                    output=output+"Predicted "+str(sum(y==Y_predicted))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                    output=output+"Predicted "+str(round((sum((y==Y_predicted))/y.shape[0])*100))+"% Correctly"+"\n\n"
                    cm=confusion_matrix(y,Y_predicted)
                
                model=KNeighborsClassifier(n_neighbors=n_neighbors)
                model.fit(X,y)
                fig = plt.figure(facecolor='lightgreen')
                plt.imshow(cm)
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.ylabel('True Label')
                plt.xlabel('Predicated Label')
                output=output+"predicted y value of input x:\t"+str(model.predict(X))+"\n\n"
                if pre!="":output=output+"predicted y value of user's input x:\t"+str(model.predict(pred))+"\n\n"
                fig.canvas.draw()
                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                pil_im=Image.fromarray(img)
                buff=io.BytesIO()
                pil_im.save(buff,format="PNG")
                img_str=base64.b64encode(buff.getvalue())
                output=output+"Confusion matrix :"+"\n"
                output=output+"\t\t\t\t\t\t\tpredicted label"+"\n"
                tl=["True","label"," "]
                k=0
                for i in cm:
                        if k<2:
                                output=output+tl[k]+"\t\t"+str(i)+"\n"
                        else:
                                output=output+"\t\t\t\t\t"+"\t\t\t\t\t"+str(i)+"\n"
                        k+=1
                return output , ""+str(img_str,'utf-8')
        except:
                output="Something went wrong... Please give correct input"
                return output,"error"


def svml(Xs,Y,thres,ite,tt,stt,ts,rs,ccv,pre):
        try:

                output=""
                b=Xs.split(';')
                x1=[list(map(float,i.split(','))) for i in b]
                yy=list(map(float,Y.split(',')))
                if thres!="":
                        c=float(thres) 
                else:
                        c=0.01

                if ite!="":
                        niters=int(ite)
                else:
                        niters=2000
                
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]
                
                if ccv!="":
                        cv=int(ccv)
                else:
                        cv=0
                if stt==1:
                        stratifyy=''
                else:
                        stratifyy='y'
                if ts!="":
                        test_size=int(ts)
                else:
                        test_size=0
                if rs!="":
                        random_state=int(rs)
                else:
                        random_state=0
                
                train_test_split_validation=tt
                X=np.array(x1)
                y=np.array(yy)

                graph='True'
                n=len(X)
                X=X.transpose()
                if len(np.unique(y))>cv or cv>len(y) or cv==1 or cv==0:
                    cv=len(np.unique(y))
                if test_size>len(y) or test_size!=0 or len(np.unique(y))>test_size :
                    test_size=len(np.unique(y))+1
                if stratifyy=='y':
                        stratify=y
                else:
                        stratifyy=''
                if c==0:
                    c=0.01

                Accuracy=[]

                if 1:
                    svm=SVM(c=c,niters=niters)
                    if train_test_split_validation==1:
                        output=output+"Comes under train_test_split validation\n\n"
                        y=np.where(y==0,-1,1)
                        if stratifyy!='' and test_size!=0 and random_state!=0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify,test_size=test_size)
                        if stratifyy!='' and test_size!=0 and random_state==0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify,test_size=test_size)
                        if stratifyy!='' and test_size==0 and random_state!=0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify)
                        if stratifyy!='' and test_size==0 and random_state==0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify)
                        if stratifyy=='' and test_size!=0 and random_state!=0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,test_size=test_size)
                        if stratifyy=='' and test_size!=0 and random_state==0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
                        if stratifyy=='' and test_size==0 and random_state!=0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state)
                        if stratifyy=='' and test_size==0 and random_state==0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y)
                        svm.fit(X_train,y_train)
                        Y_predicted=svm.predict(X_test)
                        output=output+"Predicted "+str(sum((y_test==Y_predicted)))+" Correctly Out of "+str(y_test.shape[0])+"\n\n"
                        output=output+"Predicted "+str(round((sum((y_test==Y_predicted))/y_test.shape[0])*100))+"% Correctly\n\n"
                        cm=confusion_matrix(Y_predicted,y_test,labels=[1,0])
                        if cm[0,0]+cm[0,1]!=cm[0,0]:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                        else:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                        output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                        if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                        else:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                        output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"
                    elif train_test_split_validation==2:
                        output=output+"Comes under K-fold cross validation\n\n"
                        y=np.where(y==0,-1,1)
                        kf=KFold(n_splits=cv,shuffle=True)
                        for train_index,test_index in kf.split(X):
                            X_train,X_test=X[train_index],X[test_index]
                            y_train,y_test=y[train_index],y[test_index]
                            svm.fit(X_train,y_train)
                            Y_predicted=svm.predict(X_test)
                            t=0
                            for i in range(len(y_test)):
                                if y_test[i]==Y_predicted[i]:
                                    t+=1
                            Accuracy.append(t)
                        output=output+"Accuracy :"+str(np.mean(Accuracy))+"\n\n"
                    elif train_test_split_validation==0:
                        output=output+"Comes under normal validation\n\n"
                        y=np.where(y==0,-1,1)
                        svm.fit(X,y)
                        Y_predicted=svm.predict(X)
                        output=output+"Predicted "+str(sum((y==Y_predicted)))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                        output=output+"Predicted "+str(round((sum((y==Y_predicted))/y.shape[0])*100))+"% Correctly\n\n"
                        cm=confusion_matrix(Y_predicted,y,labels=[1,0])
                        if cm[0,0]+cm[0,1]!=cm[0,0]:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that"+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                        else:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                        output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                        if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                        else:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                        output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"

                    svm=SVM(c=c,niters=niters)
                    svm.fit(X,y)
                    if pre!="":
                            output=output+"predicted y value of user's input x:\t"+str(np.where(svm.predict(pred)==-1,0,1))+"\n\n"
                    output=output+"values of w :"+str(svm.w)+"\n\n"
                    output=output+"values of b :"+str(svm.b)+"\n\n"
                    output=output+"below graph was build on first 2 X values\n\n"
                    if graph=='True':
                        fig=plt.figure(facecolor='lightgreen')
                        ax=fig.add_subplot(1,1,1)
                        ax.set_facecolor('lightgreen')
                        plt.scatter(X[:,0],X[:,1],marker='o',c=y)
                        min1=np.amin(X[:,0])
                        max1=np.amax(X[:,0])
                        pl1=(-svm.w[0]*min1-svm.b)/svm.w[1]
                        pl11=(-svm.w[0]*max1-svm.b)/svm.w[1]
                        pl2=(-svm.w[0]*min1-svm.b-1)/svm.w[1]
                        pl22=(-svm.w[0]*max1-svm.b-1)/svm.w[1]
                        pl3=(-svm.w[0]*min1-svm.b+1)/svm.w[1]
                        pl33=(-svm.w[0]*max1-svm.b+1)/svm.w[1]
                        plt.plot([min1,max1],[pl1,pl11],'k')
                        plt.plot([min1,max1],[pl2,pl22],'y--')
                        plt.plot([min1,max1],[pl3,pl33],'y--')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                return output, ""+str(img_str,'utf-8')
        except:
                output="Something went wrong... Please give correct input"
                return output,"error"


def svmk(Xs,Y,thres,gama,deg,typ,tt,stt,ts,rs,ccv,pre):
        try:
                
                output=""
                b=Xs.split(';')
                x1=[list(map(float,i.split(','))) for i in b]
                yy=list(map(float,Y.split(',')))
                if thres!="":
                        c=list(map(float,thres.split(','))) 
                else:
                        c=[0.01]    
                if gama!="":
                        gamma=list(map(float,gama.split(','))) 
                else:
                        gamma=[100]
                if deg!="":
                        degree=list(map(float,deg.split(','))) 
                else:
                        degree=[3]
                ty=["linear","poly","rbf","sigmoid"]
                type=ty[typ]
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]
                        
                if ccv!="":
                        cv=int(ccv)
                else:
                        cv=0
                if stt==1:
                        stratifyy=''
                else:
                        stratifyy='y'
                if ts!="":
                        test_size=int(ts)
                else:
                        test_size=0
                if rs!="":
                        random_state=int(rs)
                else:
                        random_state=0
                        
                train_test_split_validation=tt




                X=np.array(x1)
                y=np.array(yy)
                n=len(X)
                X=X.transpose()
                if len(np.unique(y))>cv or cv>len(y) or cv==1 or cv==0:
                    cv=len(np.unique(y))
                if test_size>len(y) or test_size==0 or len(np.unique(y))>test_size :
                    test_size=len(np.unique(y))

                if stratifyy=='y':
                    stratify=y
                else:
                    stratifyy=''
                if c==[0]:
                    c=[0.01]

                
                prm_grid = {
                'gamma':gamma,
                'C':c,
                'degree':degree
                }
                svm=SVC(kernel=type,random_state=random_state)
                gs=GridSearchCV(svm,prm_grid,cv=cv)
                gs.fit(X,y)
                c=gs.best_params_['C']
                gamma=gs.best_params_['gamma']
                degree=gs.best_params_['degree']
                    
                Accuracy=[]

                if 1:
                    svm=SVC(kernel=type,C=c,gamma=gamma,degree=degree)
                    if train_test_split_validation==1:
                        output=output+"Comes under train_test_split validation\n\n"
                        if stratifyy!='' and test_size!=0 and random_state!=0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify,test_size=test_size)
                        if stratifyy!='' and test_size!=0 and random_state==0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify,test_size=test_size)
                        if stratifyy!='' and test_size==0 and random_state!=0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify)
                        if stratifyy!='' and test_size==0 and random_state==0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify)
                        if stratifyy=='' and test_size!=0 and random_state!=0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,test_size=test_size)
                        if stratifyy=='' and test_size!=0 and random_state==0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
                        if stratifyy=='' and test_size==0 and random_state!=0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state)
                        if stratifyy=='' and test_size==0 and random_state==0:
                            X_train,X_test,y_train,y_test=train_test_split(X,y)
                        svm.fit(X_train,y_train)
                        Y_predicted=svm.predict(X_test)
                        output=output+"Predicted "+str(sum((y_test==Y_predicted)))+" Correctly Out of "+str(y_test.shape[0])+"\n\n"
                        output=output+"Predicted "+str(round((sum((y_test==Y_predicted))/y_test.shape[0])*100))+"% Correctly"+"\n\n"
                        cm=confusion_matrix(Y_predicted,y_test,labels=[1,0])
                        if cm[0,0]+cm[0,1]!=cm[0,0]:
                           output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                        else:
                           output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                        output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                        if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                        else:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                        output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"
                    elif train_test_split_validation==2:
                        output=output+"Comes under K-fold cross validation\n\n"
                        kf=KFold(n_splits=cv,shuffle=True)
                        for train_index,test_index in kf.split(X):
                            X_train,X_test=X[train_index],X[test_index]
                            y_train,y_test=y[train_index],y[test_index]
                            svm.fit(X_train,y_train)
                            Y_predicted=svm.predict(X_test)
                            t=0
                            for i in range(len(y_test)):
                                if y_test[i]==Y_predicted[i]:
                                    t+=1
                            Accuracy.append(t)
                        output=output+"Accuracy : "+str(np.mean(Accuracy))+"\n\n"
                    elif train_test_split_validation==0:
                        output=output+"Comes under normal validation\n\n"
                        svm.fit(X,y)
                        Y_predicted=svm.predict(X)
                        output=output+"Predicted "+str(sum((y==Y_predicted)))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                        output=output+"Predicted "+str(round((sum((y==Y_predicted))/y.shape[0])*100))+"% Correctly"+"\n\n"
                        cm=confusion_matrix(Y_predicted,y,labels=[1,0])
                        if cm[0,0]+cm[0,1]!=cm[0,0]:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                        else:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                        output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                        if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                        else:
                            output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data"+"\n\n"
                        output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"
                    svm=SVC(kernel=type,C=c,gamma=gamma,degree=degree)
                    svm.fit(X,y)
                    output=output+"Coefficients :"+str(svm.dual_coef_)+"\n\n"
                    output=output+"intercept :"+str(svm.intercept_)+"\n\n"                                                                                      
                    output=output+'best C value:'+str(c)+'\n'+'best gamma gamma:'+str(gamma)+'\n'+'best degree value:'+str(degree)+"\n\n"
                    if pre!="":
                        output=output+"predicted y value of user's input x:\t"+str(svm.predict(pred))+"\n\n"
                    return output,""
        except:
                output="Something went wrong... Please give correct input"
                return output,"error"
        
def navs(Xs,Y,tt,stt,ts,rs,pre):
        def accuracy(y_true,y_pred):
                    acc=np.sum(y_true==y_pred)/len(y_true)
                    return acc,"Predicted"+" "+str(np.sum(y_true==y_pred))+" "+"correctly out of"+" "+str(len(y_true))
        try:
                
                output=""
                b=Xs.split(';')
                x1=[i.split(',') for i in b]
                yy=Y.split(',')
                if pre!="":
                        bb=pre.split(';')
                        User_needs_to_predict=[i.split(',') for i in bb]
                if stt==1:
                        stratifyy=''
                else:
                        stratifyy='y'
                if ts!="":
                        test_size=int(ts)
                else:
                        test_size=0
                if rs!="":
                        random_state=int(rs)
                else:
                        random_state=0
                train_test_split_validation=tt



                

                X=np.array(x1)
                y=np.array(yy)
                yu=np.unique(y)
                Xx=X.transpose()
                if test_size>len(y):
                    test_size=3
                if test_size<len(yu):
                    test_size=len(yu)
                if stratifyy=='y':
                    stratify=y
                else:
                    stratifyy=''
                ay=y
                nb=NaiveBayes()
                if (train_test_split_validation==1):
                    output=output+"Comes under train_test_split validation\n\n"
                    if stratifyy!='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(Xx,y,random_state=random_state,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(Xx,y,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(Xx,y,random_state=random_state,stratify=stratify)
                    if stratifyy!='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(Xx,y,stratify=stratify)
                    if stratifyy=='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(Xx,y,random_state=random_state,test_size=test_size)
                    if stratifyy=='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(Xx,y,test_size=test_size)
                    if stratifyy=='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(Xx,y,random_state=random_state)
                    if stratifyy=='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(Xx,y)
                    nb.fit(X_train.transpose(),y_train)
                    pred=nb.predict(X_test)
                    ay=y_test
                    predicted_y=[]
                    predicted_prob=[]
                    for i in pred:
                        predicted_y.append(i[0])
                    output=output+"Accuracy :"+str(accuracy(ay,predicted_y))+"\n\n"
                else:
                    nb.fit(X,y)
                    pred=nb.predict(Xx)
                    predicted_y=[]
                    predicted_prob=[]
                    for i in pred:
                        predicted_y.append(i[0])
                    output=output+"Accuracy :"+str(accuracy(y,predicted_y))+"\n\n"

                nb=NaiveBayes()
                nb.fit(X,y)
                if pre!="":
                        pred=nb.predict(User_needs_to_predict)
                        predicted_y=[]
                        predicted_prob=[]
                        for i in pred:
                            bb=[]
                            for j in range(len(yu)):
                                ppt="Probability of occuring "+" "+str(yu[j])+" "+"is"+" "+str(i[1][j])+"\n"   
                                bb.append(ppt)
                                predicted_prob.append(bb)
                            predicted_y.append(i[0])
                        output=output+"predicted output probability of user's input x :\n\n"
                        for i in range(len(pred)):
                            output=output+str(User_needs_to_predict[i])+"\n\n"
                            output=output+str('\n'.join(predicted_prob[i]))+"\n"                                   
                            output=output+"Predicted output :"+str(predicted_y[i])+"\n\n"
                return output,""
        except:
                output="Something went wrong... Please give correct input"
                return output,"error"
        

def navi(Xs,Y,pre):
        def accuracy(y_true,y_pred):
                acc=np.sum(y_true==y_pred)/len(y_true)
                return acc,"Predicted"+" "+str(np.sum(y_true==y_pred))+" "+"correctly out of"+" "+str(len(y_true))

        try:
                
                output=""
                b=Xs.split(';')
                x1=[list(map(float,i.split(','))) for i in b]
                yy=list(map(float,Y.split(',')))
                if pre!="":
                        bb=pre.split(';')
                        User_needs_to_predict=[list(map(float,i.split(','))) for i in bb]

                X=np.array(x1)
                y=np.array(yy)
                X=X.transpose()
                yu=np.unique(y)
                nb=NB()
                nb.fit(X,y)
                pred=nb.predict(X)
                predicted_y=[]
                predicted_prob=[]
                for i in pred:
                    predicted_y.append(i[0])
                output=output+"Accuracy: "+str(accuracy(y,predicted_y))+"\n"
                

                nb=NB()
                nb.fit(X,y)
                if pre!="":
                        pred=nb.predict(User_needs_to_predict)
                        predicted_y=[]
                        predicted_prob=[]
                        for i in pred:
                            bb=[]
                            for j in range(len(yu)):
                                ppt="Probability of occuring"+" "+str(yu[j])+" "+"is"+" "+str(i[1][j])+"\n"   
                                bb.append(ppt)
                                predicted_prob.append(bb)
                            predicted_y.append(i[0])
                        output=output+"predicted output probability of user's input x :\n\n"
                        for i in range(len(pred)):
                            output=output+str(User_needs_to_predict[i])+"\n\n"
                            output=output+str('\n'.join(predicted_prob[i]))+"\n"                                          
                            output=output+"Predicted output:"+str(predicted_y[i])+"\n\n"
                return output,""
        except:
                output="Something went wrong... Please give correct input... Note only already builded input x sholud be given to predict"
                return output,"error"


def fuzyyc(Xs,n_cluster,fuzziness_paramete,stop_valu,data_visualiz,normalize_dat,pre):
        try:
                
                outpu=""
                b=Xs.split(';')
                xx=[list(map(float,i.split(','))) for i in b]
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]
                if n_cluster!="":
                        n_clusters=int(n_cluster)
                else:
                        n_clusters=2
                if fuzziness_paramete!="":
                        fuzziness_parameter=int(fuzziness_paramete)
                else:
                        fuzziness_parameter=2
                if stop_valu!="":
                        stop_value=float(stop_valu)            
                else:
                        stop_value=0.01
                data_visualize=data_visualiz
                normalize_data=normalize_dat





                if pre!="":
                        for i in range(len(xx)):
                                for j in pred:
                                        xx[i].append(j[i])
                if normalize_data==0:
                    XX = preprocessing.normalize(xx)
                    x = np.array(XX)
                else:
                    x=xx

                Xs=np.array(x)
                X=Xs.transpose()


                fc=fuzzyClustering()
                output=fc.fit(X,n_clusters=n_clusters,fuzziness_parameter=fuzziness_parameter,stop_value=stop_value)
                clusters=[]
                color=[]
                outpu=outpu+"[data_points]->"
                for i in range(1,n_clusters+1):
                    color.append(i)
                    clusters.append("Clust_"+str(i))
                outpu=outpu+"\t"+str(clusters)+"\n\n"
                membership_for_each_datapoint=[]
                for i in range(len(output[0])):
                    k=[]
                    for j in output:
                        k.append(j[i])
                    membership_for_each_datapoint.append(k)
                for i in range(len(membership_for_each_datapoint)):
                    if pre!="":
                            if i!=len(membership_for_each_datapoint)-len(pred):
                                    outpu=outpu+str(X[i])+'->'+str(membership_for_each_datapoint[i])+' belongs to cluster_ '+str(color[membership_for_each_datapoint[i].index(max(membership_for_each_datapoint[i]))])+"\n\n"
                            else:
                                    outpu=outpu+"\n\npredicted output probability of user's input x :\n"
                                    outpu=outpu+str(X[i])+'->'+str(membership_for_each_datapoint[i])+' belongs to cluster_ '+str(color[membership_for_each_datapoint[i].index(max(membership_for_each_datapoint[i]))])+"\n\n"
                    else:
                            outpu=outpu+str(X[i])+'->'+str(membership_for_each_datapoint[i])+' belongs to cluster_ '+str(color[membership_for_each_datapoint[i].index(max(membership_for_each_datapoint[i]))])+"\n\n"
                            
                color_labels_for_datapoints=[]
                for i in membership_for_each_datapoint:
                    color_labels_for_datapoints.append(color[i.index(max(i))]+1)
                if data_visualize==0:
                    fig=plt.figure(facecolor='lightgreen')
                    plt.title("Fuzzy Clustering")
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)
                    ay.scatter(x[0],x[1],c=color_labels_for_datapoints)
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    return outpu,""+str(img_str,'utf-8')
                else:
                      return outpu,""
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error"



def dbc(Xs,rad,minsam,normalize_dat,dm,data_visualiz,pre):
        try:
                output=""
                b=Xs.split(';')
                xx=[list(map(float,i.split(','))) for i in b]
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]
                distance_metrics=['euclidean','manhattan','chebyshev','minkowski']
                if rad!="":
                        radius=float(rad)
                else:
                        radius=0.5
                if minsam!="":
                        min_sam=int(minsam)
                else:
                        min_sam=3
                distance_metric=dm
                data_visualize=data_visualiz
                normalize_data=normalize_dat
                """ uses distance matrix for this algorithm """




                if pre!="":
                        for i in range(len(xx)):
                                for j in pred:
                                        xx[i].append(j[i])
                if normalize_data==0:
                    XX = preprocessing.normalize(xx)
                    x = np.array(XX)
                else:
                    x=np.array(xx)
                X=x.transpose()
                db = DBSCAN(eps=radius, min_samples=min_sam , metric=distance_metrics[distance_metric])
                db.fit(X)


                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = []
                output=output+"[data_points] belongs to [Clusters i]\n\n"
                for i in range(len(X)):
                    if pre!="":
                            if i!=len(X)-len(pred):
                                    output=output+str(X[i])+' belongs to cluster '+str(np.abs(db.labels_[i]))+"\n\n"
                                    labels.append(np.abs(db.labels_[i]))
                            else:
                                    output=output+"\n\npredicted output of user's input x :\n"
                                    output=output+str(X[i])+' belongs to cluster '+str(np.abs(db.labels_[i]))+"\n\n"
                                    labels.append(np.abs(db.labels_[i]))
                    else:
                           output=output+str(X[i])+' belongs to cluster '+str(np.abs(db.labels_[i]))+"\n\n"
                           labels.append(np.abs(db.labels_[i]))
                            
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                
                for i in range(len(labels)):
                    if labels[i]==-1:
                        labels[i]=-5
                    if labels[i]==1:
                        labels[i]=labels[i]*2

                if data_visualize==0:
                    fig=plt.figure(facecolor='lightgreen')
                    plt.title("Density Clustering")
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)
                    ay.scatter(x[0],x[1],c=labels,marker='o')
                    plt.title('number of clusters: %d' % n_clusters_)
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    return output,""+str(img_str,'utf-8')
                else:
                        return output,""
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error"

def hc(Xs,nc,nd,normalize_dat,dm,lin,data_visualiz,pre):
        try:
                output=""
                b=Xs.split(';')
                xx=[list(map(float,i.split(','))) for i in b]
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]
                if nc!="":
                        n_clusters=int(nc)
                else:
                        n_clusters=2
                normalize_data=normalize_dat
                link=['ward', 'complete', 'average', 'single']
                a=lin
                distance_metrics=['euclidean','manhattan','chebyshev','minkowski']
                dsm=dm
                visual_data=data_visualiz
                need_den=nd
                if pre!="":
                        for i in range(len(xx)):
                                for j in pred:xx[i].append(j[i])
                x=np.array(xx)
                Xx=x.transpose()
                if normalize_data==0:
                    XX = preprocessing.normalize(Xx)
                    X = np.array(XX)
                else:
                    X = np.array(Xx)
                if len(x[0])<n_clusters:n_clusters=len(x[0])
                gh=""
                gh2=""
                if need_den==0:
                        fig=plt.figure(facecolor='lightgreen')
                        plt.title("Dendrograms")
                        dend = shc.dendrogram(shc.linkage(X, method=link[a]))
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        gh=gh+str(img_str,'utf-8')
                
                cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=distance_metrics[dsm],linkage=link[a])
                pred_data=cluster.fit_predict(X)
                output=output+"[data_points] belongs to [Clusters i]\n\n"
                for i in range(len(Xx)):
                    if pre!="":
                            if i!=len(X)-len(pred):
                                    output=output+str(Xx[i])+' belongs to cluster '+str(pred_data[i])+"\n\n"
                            else:
                                    output=output+"\n\npredicted output of user's input x :\n"
                                    output=output+str(Xx[i])+' belongs to cluster '+str(pred_data[i])+"\n\n"
                    else:
                            output=output+str(Xx[i])+' belongs to cluster '+str(pred_data[i])+"\n\n"
                                   
                if visual_data==0:
                    fig=plt.figure(facecolor='lightgreen')
                    plt.title("Hierarchical clustering")
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)
                    ay.scatter(X.transpose()[0], X.transpose()[1], c=cluster.labels_)
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh2=gh2+str(img_str,'utf-8')
                return output,gh,gh2
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error",""

def kmc(Xs,check_bestk,nc,maxit,sd,vs,pre):
        try:
                output=""
                b=Xs.split(';')
                x=[list(map(float,i.split(','))) for i in b]
                check_best_K=check_bestk
                if nc!="":
                        n_clusters=int(nc)
                else:
                        n_clusters=2
                if maxit!="":
                        maxt=int(maxit)
                else:
                        maxt=100
                Standard_Scale_the_data=sd
                visualising_the_data=vs
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]

                X=np.array(x)
                gh=""
                if visualising_the_data==0 and Standard_Scale_the_data!=0:
                    fig=plt.figure(facecolor='lightgreen')
                    plt.title("Before clustering")
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)
                    ay.scatter(X[0],X[1],c='black')
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh=gh+str(img_str,'utf-8')
                X=X.transpose()


                
                if Standard_Scale_the_data==0:
                    scale = StandardScaler()
                    scale.fit(X)
                    X_scaled = scale.transform(X)
                    if visualising_the_data==0:
                        fig=plt.figure(facecolor='lightgreen')
                        plt.title("Before Kmeans clustering")
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        ay.scatter(X_scaled[:,0],X_scaled[:,1],c='black')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        gh=gh+str(img_str,'utf-8')
                else:
                    X_scaled=X
        
                gh2=""
                if check_best_K==0:
                    inertia = []
                    for i in range(1, len(x[0])):
                            model=KMeans(K=i,max_iters=maxt)
                            model.fit(X_scaled)
                            inertia.append(sum(model.inertia_()))
                    fig=plt.figure(facecolor='lightgreen')
                    plt.title("Best n_clusters(Elbow one)")
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)
                    ay.plot(np.arange(1, len(x[0])), inertia, marker='o')
                    plt.xlabel('Number of clusters')
                    plt.ylabel('Inertia')
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh2=gh2+str(img_str,'utf-8')
                

                gh3=""
                if 1:
                    model=KMeans(K=n_clusters,max_iters=maxt)
                    lab=model.fit(X_scaled)
                    predicted=lab[0].tolist()
                    output=output+"Cluster\t\tcounts"+"\n\n"
                    uni_y=np.unique(predicted)
                    for i in uni_y:
                            output=output+str(i)+"\t\t\t\t\t\t\t\t\t\t\t\t\t\t"+str(predicted.count(i))+"\n\n"
                    output=output+"cluster_centers_:"+str(lab[1])+"\n\n"
                    if (Standard_Scale_the_data==0 and visualising_the_data==0):
                        fig=plt.figure(facecolor='lightgreen')
                        plt.title("After Kmeans clustering")
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        ay.scatter(X_scaled[:,0],X_scaled[:,1],c=predicted)
                        ay.scatter(lab[1][:,0],lab[1][:,1],marker="*",s=100,c='red')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        gh3=gh3+str(img_str,'utf-8')
                        
                    elif (Standard_Scale_the_data==1 and visualising_the_data==0):
                        fig=plt.figure(facecolor='lightgreen')
                        plt.title("After Kmeans clustering")
                        ax=plt.axes()
                        ax.set_facecolor('lightgreen')
                        ay=fig.add_subplot(1,1,1)
                        ay.scatter(X_scaled[:,0],X_scaled[:,1],c=predicted)
                        ay.scatter(lab[1][:,0],lab[1][:,1],marker="*",s=100,c='red')
                        fig.canvas.draw()
                        img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                        pil_im=Image.fromarray(img)
                        buff=io.BytesIO()
                        pil_im.save(buff,format="PNG")
                        img_str=base64.b64encode(buff.getvalue())
                        gh3=gh3+str(img_str,'utf-8')
                    output=output+"[data_points] belongs to [Clusters i]\n\n"
                    for i in range(len(X_scaled)):
                            output=output+str(X_scaled[i])+' belongs to cluster '+str(lab[0][i])+"\n\n"
                    if pre!="":
                        output=output+'Given user datum :'+"\n\n"
                        for i in pred:
                                pre=model._closest_centroid(i,lab[1])
                                output=output+str(i)+' belongs to cluster '+str(pre)+"\n"
                return output,gh2,gh,gh3
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error","",""
         




        
def pca(Xs,sd,nc):
        try:
                output=""
                b=Xs.split(';')
                x=[list(map(float,i.split(','))) for i in b]
                Standard_Scale_the_data=sd


                if nc!="":
                    n_components=int(nc)
                else:
                    n_components=len(x)

                X=np.array(x)
                X=X.transpose()
                if Standard_Scale_the_data==0:
                    scale = StandardScaler()
                    scale.fit(X)
                    X = scale.transform(X)
                pca=PCA(n_components)
                en=pca.fit(X)
                X_projected=pca.transform(X)

                output=output+'Shape of X:'+str(X.shape)+"\n\n"
                output=output+'Shape of transformed X:'+str(X_projected.shape)+"\n\n"
                output=output+'mean:'+str(en[2])+"\n\n"
                output=output+'Covariance matrix :\n'+str(en[3])+"\n\n"
                output=output+"eigenvalues :\n"+str(en[0])+"\n\n"
                output=output+"normalized eigenvectors :\n"+str(en[1])+"\n\n"

                xs=[]
                for i in range(n_components):
                    xs.append(X_projected[:,i])
                    output=output+'Principle Component '+str(i+1)+':'+"\n"+str(X_projected[:,i])+' (Score percentage='+str(round((en[0][i]/sum(en[0])*100),2))+'%)'+"\n\n"
                gh=""
                if n_components==2:
                    fig=plt.figure(facecolor='lightgreen')
                    plt.title("PCA")
                    plt.xlabel('Principle Component 1')
                    plt.ylabel('Principle Component 2')
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)
                    ay.scatter(xs[0],xs[1])
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh=gh+str(img_str,'utf-8')
                elif n_components==1:
                    y=np.ones(len(x[0]))
                    fig=plt.figure(facecolor='lightgreen')
                    plt.title('Principle Component 1')
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)
                    ay.hlines(1,min(xs[0])-1,max(xs[0])+1)
                    ay.plot(xs[0],y,'|',ms=40)
                    ay.axis('off')
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh=gh+str(img_str,'utf-8')
                    
                return output,gh
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error"


def kpca(Xs,kn,sd,nc,gm,dg):

        try:
                outpu=""
                b=Xs.split(';')
                x=[list(map(float,i.split(','))) for i in b]
                Standard_Scale_the_data=sd
                k=kn

                kernel=["linear","poly","rbf","sigmoid"]
                if nc!="":
                    n_components=int(nc)
                else:
                    n_components=len(x)

                if gm!="":
                    gamma=int(gm)
                else:
                    gamma=100

                if dg!="":
                    degree=int(dg)
                else:
                    degree=3

                X=np.array(x)
                X=X.transpose()
                if Standard_Scale_the_data==0:
                    scale = StandardScaler()
                    scale.fit(X)
                    X = scale.transform(X)


                kpca=KernelPCA(n_components=n_components,kernel=kernel[k],gamma=gamma,degree=degree)
                output=kpca.fit_transform(X)
                outpu=outpu+'Shape of X: '+str(X.shape)+"\n\n"
                outpu=outpu+'Shape of transformed X: '+str(output.shape)+"\n\n"
                for i in range(len(output[0])):
                    outpu=outpu+'Principle Component '+str(i+1)+":\n"+str(output[:,i])+"\n\n"
                gh=""
                if n_components==2:
                    fig=plt.figure(facecolor='lightgreen')
                    plt.title("Kernel PCA")
                    plt.xlabel('Principle Component 1')
                    plt.ylabel('Principle Component 2')
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)
                    ay.scatter(output[:,0],output[:,1])
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh=gh+str(img_str,'utf-8')
                elif n_components==1:
                    fig=plt.figure(facecolor='lightgreen')
                    plt.title('Principle Component 1')
                    y=np.ones(len(x[0]))
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    ay=fig.add_subplot(1,1,1)
                    ay.hlines(1,min(output[:,0])-1,max(output[:,0])+1)
                    ay.plot(output[:,0],y,'|',ms=40)
                    ay.axis('off')
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh=gh+str(img_str,'utf-8')
                return outpu,gh
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error"




def ztest(ts,subts,los,str4="",str5="",str6="",str7="",str8="",str9="",str10="",str11=""):
        try:
                
                output=""
                tests=['Single mean','Difference of mean','Single proportion','Difference of proportion']
                i=ts
                test=['Two tail test','Left tail test','Right tail test']
                j=subts
                zalpha=float(los)       
                
                
                if tests[i]=='Single mean':
                     n=float(str4)     
                     sig=float(str5)  
                     x=float(str6)   
                     mu0=float(str7)   
                     



                     if test[j]=='Two tail test':
                          zalp=stats.norm.ppf((zalpha/100)/2)
                          output=output+"ztest critical value:"+str([zalp,abs(zalp)])+"\n\n"
                     elif test[j]== 'Right tail test':
                          zalp=stats.norm.ppf(1-(zalpha/100))
                          output=output+"ztest critical value:"+str(zalp)+"\n\n"
                     else:
                          zalp=stats.norm.ppf((zalpha/100))
                          output=output+"ztest critical value:"+str(zalp)+"\n\n"
                     sd=sig/(n**0.5)

                     zts=(x-mu0)/sd
                     
                     output=output+"S.E:"+str(sd)+"\n\n"
                     output=output+"zts:"+str(zts)+"\n\n"
                     
                     
                    
                     if test[j]=='Two tail test':
                         output=output+"-zalp<zts<zalp:"+str(zalp<zts<abs(zalp))+str(" Accept H0" if -zalp<zts<abs(zalp) else " Reject H0" )+"\n\n"
                     elif test[j]== 'Right tail test':
                         output=output+"zts<zalp:"+str(zts<zalp)+str(" Accept H0" if zts<zalp else " Reject H0")+"\n\n"
                     else:
                         output=output+"-zalp<zts:"+str(zalp<zts)+str(" Accept H0" if zalp<zts else " Reject H0")+"\n\n"

                     output=output+"Confidence Interval for sample"+"\n\n"
                     if test[j]=='Two tail test':
                         cl=mu0-(zalp*sd)<x<mu0+(abs(zalp)*sd)
                         output=output+"mu0-(zalp*S.E)<x<mu0+(zalp*S.E) "+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                         cl=x<mu0+(zalp*sd)
                         output=output+"x<mu0+(zalp*S.E) "+str(cl)+"\n\n"
                     else:
                         cl=mu0-(zalp*sd)<x
                         output=output+"mu0-(zalp*S.E)<x "+str(cl)+"\n\n"
                     output=output+"Confidence Interval for population"+"\n\n"
                     if test[j]=='Two tail test':
                         cl=x-(zalp*sd)<mu0<x+(abs(zalp)*sd)
                         output=output+"x-(zalp*S.E)<mu0<x+(zalp*S.E) "+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                         cl=mu0<x+(zalp*sd)
                         output=output+"mu0<x+(zalp*S.E) "+str(cl)+"\n\n"
                     else:
                         cl=x-(zalp*sd)<mu0
                         output=output+"x-(zalp*S.E)<mu0 "+str(cl)+"\n\n"
                         
                elif tests[i]=='Difference of mean':
                     x1=float(str4)     
                     x2=float(str5)     
                     if str6!="":
                             mu1=float(str6)      
                     else:
                             mu1=0
                     if str7!="":
                             mu2=float(str7)      
                     else:
                             mu2=0
                     n1=float(str8)    
                     n2=float(str9)    
                     sd1=float(str10)    
                     sd2=float(str11)    
                     



                     if test[j]=='Two tail test':
                          zalp=stats.norm.ppf((zalpha/100)/2)
                          output=output+"ztest critical value:"+str([zalp,abs(zalp)])+"\n\n"
                     elif test[j]== 'Right tail test':
                          zalp=stats.norm.ppf(1-(zalpha/100))
                          output=output+"ztest critical value:"+str(zalp)+"\n\n"
                     else:
                          zalp=stats.norm.ppf((zalpha/100))
                          output=output+"ztest critical value:"+str(zalp)+"\n\n"
                         
                     sig12=(sd1**2)
                     sig22=(sd2**2)
                     
                     
                     SE=((sig12/n1)+(sig22/n2))**0.5
                     
                     zts=(((x1-x2))-(mu1-mu2))/SE
                     
                     output=output+"SE:"+str(SE)+"\n\n"
                     
                     output=output+"zts:"+str(zts)+"\n\n"
                     
                     
                     
                     if test[j]=='Two tail test':
                         output=output+"-zalp<zts<zalp:"+str(zalp<zts<abs(zalp))+str(" Accept H0" if zalp<zts<abs(zalp) else " Reject H0")+"\n\n"
                     elif test[j]== 'Right tail test':
                         output=output+"zts<zalp:"+str(zts<zalp)+str(" Accept H0" if zts<zalp else " Reject H0")+"\n\n"
                     else:
                         output=output+"-zalp<zts:"+str(zalp<zts)+str(" Accept H0" if zalp<zts else " Reject H0")+"\n\n"

                         
                     output=output+"Confidence Interval for samples\n\n"
                     if test[j]=='Two tail test':
                         cl=((mu1-mu2)-(zalp*SE))<(x1-x2)<((mu1-mu2)+(abs(zalp)*SE))
                         output=output+"(mu1-mu2)-(zalp*SE))<(x1-x2)<((mu1-mu2)+(zalp*SE) "+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                         cl=(x1-x2)<((mu1-mu2)+(zalp*SE))
                         output=output+"(x1-x2)<((mu1-mu2)+(zalp*SE)) "+str(cl)+"\n\n"
                     else:
                         cl=((mu1-mu2)-(zalp*SE))<(x1-x2)
                         output=output+"((mu1-mu2)-(zalp*SE))<(x1-x2) "+str(cl)+"\n\n"
                     
                     output=output+"Confidence Interval for population\n\n"
                     if test[j]=='Two tail test':
                         cl=((x1-x2)-(zalp*SE))<(mu1-mu2)<((x1-x2)+(abs(zalp)*SE))
                         output=output+"((x1-x2)-(zalp*SE))<(mu1-mu2)<((x1-x2)+(zalp*SE)) "+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                         cl=(mu1-mu2)<((x1-x2)+(zalp*SE))
                         output=output+"(mu1-mu2)<((x1-x2)+(zalp*SE)) "+str(cl)+"\n\n"
                     else:
                         cl=((x1-x2)-(zalp*SE))<(mu1-mu2)
                         output=output+"((x1-x2)-(zalp*SE))<(mu1-mu2) "+str(cl)+"\n\n"
                     
                elif tests[i]=='Single proportion':
                     n=float(str4)           
                     p=float(str5)           
                     P0=float(str6)         
                     


                     if test[j]=='Two tail test':
                          zalp=stats.norm.ppf((zalpha/100)/2)
                          output=output+"ztest critical value:"+str([zalp,abs(zalp)])+"\n\n"
                     elif test[j]== 'Right tail test':
                          zalp=stats.norm.ppf(1-(zalpha/100))
                          output=output+"ztest critical value:"+str(zalp)+"\n\n"
                     else:
                          zalp=stats.norm.ppf((zalpha/100))
                          output=output+"ztest critical value:"+str(zalp)+"\n\n"
                     
                     
                     SE=((P0*(1-P0))/n)**0.5
                     
                     zts=(p-P0)/SE
                     
                     output=output+"SE:"+str(SE)+"\n\n"
                     output=output+"zts:"+str(zts)+"\n\n"
                     
                     
                     
                     
                     if test[j]=='Two tail test':
                         output=output+"-zalp<zts<zalp:"+str(zalp<zts<abs(zalp))+str(" Accept H0" if zalp<zts<abs(zalp) else " Reject H0")+"\n\n"
                     elif test[j]== 'Right tail test':
                         output=output+"zts<zalp:"+str(zts<zalp)+str(" Accept H0" if zts<zalp else " Reject H0")+"\n\n"
                     else:
                         output=output+"-zalp<zts:"+str(zalp<zts)+str(" Accept H0" if zalp<zts else " Reject H0")+"\n\n"
                         
                     output=output+"Confidence Interval for samples\n\n"
                     if test[j]=='Two tail test':
                         cl=(P0-(zalp*SE))<p<(P0+(abs(zalp)*SE))
                         output=output+"(P0-(zalp*SE))<p<(P0+(zalp*SE)) "+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                         cl=(P0-(zalp*SE))<p
                         output=output+"(P0-(zalp*SE))<p "+str(cl)+"\n\n"
                     else:
                         cl=p<(P0+(zalp*SE))
                         output=output+"p<(P0+(zalp*SE)) "+str(cl)+"\n\n"
                     
                     
                     output=output+"Confidence Interval for population\n\n"
                     if test[j]=='Two tail test':
                         cl=(p-(zalp*SE))<P0<(p+(abs(zalp)*SE))
                         output=output+"(p-(zalp*SE))<P0<(p+(zalp*SE)) "+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                         cl=P0<(p+(zalp*SE))
                         output=output+"P0<(p+(zalp*SE)) "+str(cl)+"\n\n"
                     else:
                         cl=(p-(zalp*SE))<P0
                         output=output+"(p-(zalp*SE))<P0 "+str(cl)+"\n\n"
                     
                     
                elif tests[i]=='Difference of proportion':
                     n1=float(str4)   
                     n2=float(str5)    
                     p1=float(str6)    
                     p2=float(str7)   
                     if str8!="":
                             P1=float(str8)      
                     else:
                             P1=0
                     if str9!="":
                             P2=float(str9)      
                     else:
                             P2=0
                     same_population=str10



                     if test[j]=='Two tail test':
                          zalp=stats.norm.ppf((zalpha/100)/2)
                          output=output+"ztest critical value:"+str([zalp,abs(zalp)])+"\n\n"
                     elif test[j]== 'Right tail test':
                          zalp=stats.norm.ppf(1-(zalpha/100))
                          output=output+"ztest critical value:"+str(zalp)+"\n\n"
                     else:
                          zalp=stats.norm.ppf((zalpha/100))
                          output=output+"ztest critical value:"+str(zalp)+"\n\n"
                     
                     

                     if same_population==0:
                          P=((p1*n1)+(p2*n2))/(n1+n2)
                          SE=(((P*(1-P))*((1/n1)+(1/n2))))*0.5
                     else:
                          SE=(((p1*(1-p1))/n1)+((p2*(1-p2))/n2))**0.5
                     
                     zts=((p1-p2)-(P1-P2))/SE
                     
                     output=output+"SE:"+str(SE)+"\n\n"
                     output=output+"zts:"+str(zts)+"\n\n"
                     
                     
                     
                     if test[j]=='Two tail test':
                         output=output+"-zalp<zts<zalp:"+str(zalp<zts<abs(zalp))+str(" Accept H0" if zalp<zts<abs(zalp) else " Reject H0")+"\n\n"
                     elif test[j]== 'Right tail test':
                         output=output+"zts<zalp:"+str(zts<zalp)+str(" Accept H0" if zts<zalp else " Reject H0")+"\n\n"
                     else:
                         output=output+"-zalp<zts:"+str(zalp<zts)+str(" Accept H0" if zalp<zts else "Reject H0")+"\n\n"

                     output=output+"Confidence Interval for samples\n\n"
                     if test[j]=='Two tail test':
                         cl=(P1-P2)-(zalp*SE)<p1-p2<(P1-P2)+(abs(zalp)*SE)
                         output=output+"(P1-P2)-(zalp*SE)<p1-p2<(P1-P2)+(zalp*SE) "+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                         cl=p1-p2<(P1-P2)+(zalp*SE)
                         output=output+"p1-p2<(P1-P2)+(zalp*SE) "+str(cl)+"\n\n"
                     else:
                         cl=(P1-P2)-(zalp*SE)<p1-p2
                         output=output+"(P1-P2)-(zalp*SE)<p1-p2 "+str(cl)+"\n\n"
                                                             
                     output=output+"Confidence Interval for population"+"\n\n"
                     if test[j]=='Two tail test':
                         cl=(p1-p2)-(zalp*SE)<P1-P2<(p1-p2)+(abs(zalp)*SE)
                         output=output+"(p1-p2)-(zalp*SE)<P1-P2<(p1-p2)+(zalp*SE) "+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                         cl=P1-P2<(p1-p2)+(zalp*SE)
                         output=output+"P1-P2<(p1-p2)+(zalp*SE) "+str(cl)+"\n\n"
                     else:
                         cl=(p1-p2)-(zalp*SE)<P1-P2
                         output=output+"(p1-p2)-(zalp*SE)<P1-P2 "+str(cl)+"\n\n"
                return output,""
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error"


def ttest(ts,subts,los,str4="",str5="",str6="",str7="",str8="",str9="",str10="",str11="",str12="",str13="",str14=""):
        try:
                
                output=""
                i=ts
                tests=['Single mean','Difference of mean']
                test=['Two tail test','Left tail test','Right tail test']
                j=subts
                talpha=float(los)       


                if tests[i]=='Single mean':
                   n=int(str4)               
                   x=float(str5)             
                   if str6!="":
                           mu=float(str6)             
                   else:
                           mu=0
                   sd=float(str7)            
                   standard_deviation_given_directly=int(str8)





                   if standard_deviation_given_directly==0:
                      SE=sd/((n-1)**0.5)
                   else:
                      SE=sd/(n**0.5)
                      
                   ts=(x-mu)/SE
                   output=output+"S.E:"+str(SE)+"\n\n"
                   output=output+"ts:"+str(ts)+"\n\n"

                   
                   if test[j]=='Two tail test':
                         talp=stats.t.ppf(q=(talpha/100)/2,df=n-1)
                         output=output+"t_alpha_value:"+str([talp,abs(talp)])+"\n\n"
                         output=output+"-talp<ts<zalp:"+str(talp<ts<abs(talp))+str(" Accept H0" if talp<ts<abs(talp) else " Reject H0")+"\n\n"
                   elif test[j]== 'Right tail test':
                         talp=abs(stats.t.ppf(q=talpha/100,df=n-1))
                         output=output+"t_alpha_value:"+str(talp)+"\n\n"
                         output=output+"ts<talp:"+str(ts<talp)+str(" Accept H0" if ts<talp else " Reject H0")+"\n\n"
                   else:
                         talp=stats.t.ppf(q=talpha/100,df=n-1)
                         output=output+"t_alpha_value:"+str(talp)+"\n\n"
                         output=output+"-talp<ts:"+str(-talp<ts)+str(" Accept H0" if -talp<ts else " Reject H0")+"\n\n"

                   output=output+"Confidence Interval for samples\n\n"
                   if test[j]=='Two tail test':
                      cl=mu-(talp*sd)<x<mu+(abs(talp)*sd)
                      output=output+"mu-(talp*sd)<x<mu+(talp*sd)"+str(cl)+"\n\n"
                   elif test[j]== 'Right tail test':
                      cl=x<mu+(talp*sd)
                      output=output+"x<mu+(talp*sd)"+str(cl)+"\n\n"
                   else:
                      cl=mu-(talp*sd)<x
                      output=output+"mu-(talp*sd)<x"+str(cl)+"\n\n"
                   output=output+"Confidence Interval for population"+"\n\n"
                   if test[j]=='Two tail test':
                      cl=x-(talp*sd)<mu<x+(abs(talp)*sd)
                      output=output+"x-(talp*sd)<mu<x+(talp*sd)"+str(cl)+"\n\n"
                   elif test[j]== 'Right tail test':
                      cl=mu<x+(talp*sd)
                      output=output+"mu<x+(talp*sd)"+str(cl)+"\n\n"
                   else:
                      cl=x-(talp*sd)<mu
                      output=output+"x-(talp*sd)<mu"+str(cl)+"\n\n"


                elif tests[i]=='Difference of mean':
                     different_population=int(str4)
                     x1=float(str5)                    
                     x2=float(str6)                     
                     n1=int(str7)                        
                     n2=int(str8)                          
                     if str9!="":
                             mu1=float(str9)                         
                     else:
                             mu1=0
                     if str10!="":
                             mu2=float(str10)                        
                     else:
                             mu2=0
                     if different_population==1 or different_population==2 or different_population==0:
                             sig12=float(str11)                      
                             sig22=float(str12)                      
                     if different_population==3:
                             sd=float(str11)


                     if different_population==1 or different_population==0:          
                           SE=((sig12/n1)+(sig22/n2))**0.5
                     if different_population==2:       
                           sd=((((n1-1)*sig12)+((n2-1)*sig22))/(n1+n2-2))**0.5
                           SE=sd*(((1/n1)+(1/n2))**0.5)
                     if different_population==3:   
                           SE=sd*(((1/n1)+(1/n2)))**0.5
                     

                     ts=((x1-x2)-(mu1-mu2))/SE
                     
                     output=output+"SE:"+str(SE)+"\n\n"
                     
                     output=output+"ts:"+str(ts)+"\n\n"
                     
                     talp=talpha

                     if test[j]=='Two tail test':
                        talp=stats.t.ppf(q=(talpha/100)/2,df=n1+n2-2)
                        output=output+"t_alpha_value:"+str([talp,abs(talp)])+"\n\n"
                        output=output+"-talp<ts<zalp:"+str(talp<ts<abs(talp))+str(" Accept H0" if talp<ts<abs(talp) else " Reject H0")+"\n\n"
                     elif test[j]== 'Right tail test':
                        talp=abs(stats.t.ppf(q=talpha/100,df=n1+n2-2))
                        output=output+"t_alpha_value:"+str(talp)+"\n\n"
                        output=output+"ts<talp:"+str(ts<talp)+str(" Accept H0" if ts<talp else " Reject H0")+"\n\n"
                     else:
                        talp=stats.t.ppf(q=talpha/100,df=n1+n2-2)
                        output=output+"t_alpha_value:"+str(talp)+"\n\n"
                        output=output+"-talp<ts:"+str(-talp<ts)+str(" Accept H0" if -talp<ts else " Reject H0")+"\n\n"

                         
                     output=output+"Confidence Interval for samples\n\n"
                     if test[j]=='Two tail test':
                        cl=(mu1-mu2)-(talp*SE)<(x1-x2)<(mu1-mu2)+(abs(talp)*SE)
                        output=output+"(mu1-mu2)-(talp*SE)<(x1-x2)<(mu1-mu2)+(talp*SE)"+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                        cl=(x1-x2)<(mu1-mu2)+(talp*SE)
                        output=output+"(x1-x2)<(mu1-mu2)+(talp*SE)"+str(cl)+"\n\n"
                     else:
                        cl=(mu1-mu2)-(talp*SE)<(x1-x2)
                        output=output+"(mu1-mu2)-(talp*SE)<(x1-x2)"+str(cl)+"\n\n"

                     output=output+"Confidence Interval for population\n\n"
                     if test[j]=='Two tail test':
                        cl=(x1-x2)-(talp*SE)<(mu1-mu2)<(x1-x2)+(abs(talp)*SE)
                        output=output+"(x1-x2)-(talp*SE)<(mu1-mu2)<(x1-x2)+(talp*SE)"+str(cl)+"\n\n"
                     elif test[j]== 'Right tail test':
                        cl=(x1-x2)-(talp*SE)<(mu1-mu2)
                        output=output+"(x1-x2)-(talp*SE)<(mu1-mu2)"+str(cl)+"\n\n"
                     else:
                        cl=(mu1-mu2)<(x1-x2)+(talp*SE)
                        output=output+"(mu1-mu2)<(x1-x2)+(talp*SE)"+str(cl)+"\n"
                return output,""
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error"


def ftest(ts,subts,los,str4="",str5="",str6="",str7=""):
        try:
                
                output=""
                test=['Two tail test','Left tail test','Right tail test']
                j=ts
                n1=int(str4)
                n2=int(str5)
                critical_value=float(los)                    
                s1_s2_are_given=subts 

                if s1_s2_are_given==0:
                   s1=float(str6)       
                   s2=float(str7)      
                   sig12=(n1*(s1))/(n1-1)
                   sig22=(n2*(s2))/(n2-1)
                else:            
                   sig12=float(str6)  
                   sig22=float(str7)  




                if sig12>sig22:
                   fts=sig12/sig22
                else:
                   fts=sig22/sig12
                   
                output=output+"variance of 1st population:"+str(sig12)+"\n\n"
                output=output+"variance of 2nd population:"+str(sig22)+"\n\n"
                   
                output=output+"fts:"+str(fts)+"\n\n"

                if test[j]=='Two tail test':
                   falpL=stats.f.ppf(q=(critical_value/100)/2,dfn=n1-1,dfd=n2-1)
                   falpR=1/falpL
                else:
                   falpL=stats.f.ppf(q=(critical_value/100),dfn=n1-1,dfd=n2-1)
                   falpR=1/falpL

                output=output+"falpL:"+str(falpL)+"\n\n"
                output=output+"falpR:"+str(falpR)+"\n\n"

                if test[j]=='Two tail test':
                   output=output+"falpL<fts<falpR:"+str(falpL<fts<falpR)+str(" Accept H0" if falpL<fts<falpR else " Reject H0")+"\n\n"
                elif test[j]== 'Right tail test':
                   output=output+"fts<falpR:"+str(fts<falpR)+str(" Accept H0" if fts<falpR else " Reject H0")+"\n\n"
                else:
                   output=output+"falpL<fts:"+str(falpL<fts)+str(" Accept H0" if falpL<fts else " Reject H0")+"\n\n"
                return output,""
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error"




def chitest(ts,los,Xs,ds="",Y=""):
        try:
                
                def dof(j):
                        if distibustion[j]=="Normal distribustion":degree_of_freedom[0]=len(expectations)-3
                        elif distibustion[j]=="Binomial distribustion":degree_of_freedom[0]=len(expectations)-1
                        elif distibustion[j]=="Poisson distribustion":degree_of_freedom[0]=len(expectations)-2
                        elif distibustion[j]=="None":degree_of_freedom[0]=len(expectations)-1
            

                output=""
                Test_of_Independence_of_attributes=ts    
                alpha=float(los)                             
                if Test_of_Independence_of_attributes==0:
                    b=Xs.split(';')
                    attributes_data=[list(map(float,i.split(','))) for i in b]
                    total=0
                    col_sum=[]
                    for i in range(len(attributes_data[0])):
                        k=0
                        for j in attributes_data:
                            k+=j[i]
                        col_sum.append(k)
                    for i in attributes_data:
                        total+=sum(i)
                    observations=[]
                    expectations=[]
                    k=0
                    for i in attributes_data:
                        for j in i:
                            observations.append(j)
                            exp=col_sum[k]*sum(i)/total
                            expectations.append(int(exp))
                        k+=1
                    j=[(len(attributes_data)-1)*(len(attributes_data[0])-1)]
                else :  
                    observations=list(map(float,Xs.split(',')))
                    expectations=list(map(float,Y.split(',')))
                    distibustion=["None","Normal distribustion","Binomial distribustion","Poisson distribustion"]
                    j=ds
                    







                degree_of_freedom=[0]
                if Test_of_Independence_of_attributes!=0:
                    dof(j)
                else:
                    degree_of_freedom=j
                obs_exp=[[observations[i],expectations[i]]for i in range(len(observations))]
                ky=lambda x: x[1] 
                obs_exp.sort(key=ky)
                grouped=0
                less_than_five=[]
                for i in obs_exp:
                    if i[1]<5:
                        less_than_five.append(i)
                        del observations[observations.index(i[0])]
                        del expectations[expectations.index(i[1])]
                        grouped=1
                obs=0
                exp=0
                for i in less_than_five:
                    obs+=i[0]
                    exp+=i[1]
                if grouped==1:
                    observations.append(obs)
                    expectations.append(exp)
                    obs_exp=[[observations[i],expectations[i]]for i in range(len(observations))]
                    ky=lambda x: x[1]
                    obs_exp.sort(key=ky)
                    k1=0
                    k2=0
                    ind=0
                    if obs_exp[0][1]>5:
                        output=output+"observations and expectations are grouped"+"\n\n"
                        output=output+str(observations)+"\n\n"
                        output=output+str(expectations)+"\n\n"
                        if Test_of_Independence_of_attributes!=0:
                                dof(j)
                        else:
                                degree_of_freedom=j
                    else:
                        sobs=[]
                        sexp=[]
                        for i in obs_exp:
                            sobs.append(i[0])
                            sexp.append(i[1])
                        for i in range(len(sexp)):
                            if sum(sexp[:i])>=5:
                                k1=sum(sexp[:i])
                                k2=sum(sobs[:i])
                                ind=i
                                break
                        del sobs[:ind]
                        del sexp[:ind]
                        sobs.append(k2)
                        sexp.append(k1)
                        del observations[:]
                        del expectations[:]
                        for i in range(len(sexp)):
                            observations.append(sobs[i])
                            expectations.append(sexp[i])
                        output=output+"observations and expectations are grouped"+"\n\n"
                        output=output+str(observations)+"\n\n"
                        output=output+str(expectations)+"\n\n"
                        if Test_of_Independence_of_attributes!=0:
                                dof(j)
                        else:
                                degree_of_freedom=j
                        
                Xv=stats.chi2.ppf(q=1-(alpha/100),df=degree_of_freedom[0])
                output=output+"Chi-square critical value:"+str(Xv)+"\n\n"
                X=[]
                for i in range(len(observations)):
                    if (observations[i]-expectations[i])!=0:
                            X.append(((observations[i]-expectations[i])**2)/expectations[i])
                    else:
                            X.append(0)
                Xts=sum(X)
                output=output+"Xts:"+str(Xts)+"\n\n"

                if Test_of_Independence_of_attributes==1:
                    if Xts<Xv:
                        output=output+"Accept H0\n"
                        output=output+"Fit is good\n"
                    else:
                        output=output+"Reject H0\n\n"
                        output=output+"Fit is not good\n\n"
                else:
                    output=output+"observations:"+str(observations)+"\n\n"
                    output=output+"expectations:"+str([float(i) for i in expectations])+"\n\n"
                    if Xts<Xv:
                        output=output+"Accept H0\n\n"
                        output=output+"Attributes are dependent\n\n"
                    else:
                        output=output+"Reject H0\n\n"
                        output=output+"Attributes are not dependent\n\n"
                return output,""
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error"


def dt(fnn,Xs,Y,cini,tt,sty,sc,md="",msl="",mln="",ccv="",rs="",ts="",pre=""):
        try:
                output=""
                gh1=""
                gh2=""
                fn=list(map(str,fnn.split(',')))
                b=Xs.split(';')
                x1=[list(map(float,i.split(','))) for i in b]
                yy=list(map(float,Y.split(',')))
                if cini==0:
                        criterion='gini'
                else:
                        criterion='entropy'
                if md!="":
                        max_depth=list(map(int,md.split(',')))
                else:
                         max_depth=[]
                if msl!="":
                        min_samples_leaf=list(map(int,msl.split(',')))
                else:
                        min_samples_leaf=[]
                if mln!="":
                        max_leaf_nodes=list(map(int,mln.split(',')))
                else:
                        max_leaf_nodes=[]
                s=['accuracy','f1','precision','recall']
                scoring=s[sc]
                if ccv!="":
                        cv=int(ccv)
                else:
                        cv=3
                if sty==0:
                        stratifyy='y'
                else:
                        stratifyy=''
                if ts!="":
                        test_size=int(ts)
                else:
                        test_size=0
                if rs!="":
                        random_state=int(rs)
                else:
                        random_state=0
                train_test_split_validation=tt
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]



                X=np.array(x1)
                y=np.array(yy)
                if stratifyy=='y':
                    stratify=y
                else:
                    stratifyy=''
                if len(np.unique(y))<cv or cv>len(y) or cv==1 or cv==0:
                    cv=len(np.unique(y))
                X=X.transpose()
                if test_size>len(y) or test_size==0 or len(np.unique(y))>test_size :
                    test_size=len(np.unique(y))





                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes!=[]:
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes
                    }
                    dt=DecisionTreeClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes==[]: 
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf
                    }
                    dt=DecisionTreeClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes!=[]: 
                    param={
                    'max_depth':max_depth,
                    'max_leaf_nodes':max_leaf_nodes
                    }
                    dt=DecisionTreeClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_        
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes==[]: 
                    param={
                    'max_depth':max_depth,
                    }
                    dt=DecisionTreeClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes!=[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes
                    }
                    dt=DecisionTreeClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes==[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf
                    }
                    dt=DecisionTreeClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf==[] and max_leaf_nodes!=[]: 
                    param={
                    'max_leaf_nodes':max_leaf_nodes
                    }
                    dt=DecisionTreeClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[]:
                    max_depth=best_params['max_depth']
                else:
                    max_depth=0
                    
                if min_samples_leaf!=[]:
                    min_samples_leaf=best_params['min_samples_leaf']
                else:
                    min_samples_leaf=0
                    
                if max_leaf_nodes!=[]:
                    max_leaf_nodes=best_params['max_leaf_nodes']
                else:
                    max_leaf_nodes=0



                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0: 
                    model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0: 
                    model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0: 
                    model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0: 
                    model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0: 
                    model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0: 
                    model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0: 
                    model=DecisionTreeClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0: 
                    model=DecisionTreeClassifier(max_depth=max_depth)
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0: 
                    model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0: 
                    model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0: 
                    model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0: 
                    model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0: 
                    model=DecisionTreeClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0: 
                    model=DecisionTreeClassifier(criterion=criterion)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0: 
                    model=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0: 
                    model=DecisionTreeClassifier()



                Accuracy=[]

                if train_test_split_validation==1:
                    output=output+"Comes under train_test_split validation\n\n"
                    if stratifyy!='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify)
                    if stratifyy!='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify)
                    if stratifyy=='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,test_size=test_size)
                    if stratifyy=='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
                    if stratifyy=='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state)
                    if stratifyy=='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y)
                    model.fit(X_train,y_train)
                    Y_predicted=model.predict(X_test)
                    output=output+"Predicted "+str(sum((y_test==Y_predicted)))+" Correctly Out of "+str(y_test.shape[0])+"\n\n"
                    output=output+"Predicted "+str(round((sum((y_test==Y_predicted))/y_test.shape[0])*100))+"% Correctly"+"\n\n"
                    cm=confusion_matrix(Y_predicted,y_test,labels=[1,0])
                    if cm[0,0]+cm[0,1]!=cm[0,0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                    output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                    if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                    output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(max_depth=max_depth)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier()
                    model.fit(X,y)
                    Y_predicted=model.predict(X)
                    fig=plt.figure(facecolor="lightgreen")
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    plt.title("Decision tree visualization\n")
                    tree.plot_tree(model,feature_names=fn)
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh2=gh2+str(img_str,'utf-8')
                   

                elif train_test_split_validation==2:
                    output=output+"Comes under K-fold cross validation\n\n"
                    kf=KFold(n_splits=cv,shuffle=True)
                    for train_index,test_index in kf.split(X):
                        X_train,X_test=X[train_index],X[test_index]
                        y_train,y_test=y[train_index],y[test_index]
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0:
                            model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0:
                            model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0:
                            model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0:
                            model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0:
                            model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0:
                            model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0:
                            model=DecisionTreeClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0:
                            model=DecisionTreeClassifier(max_depth=max_depth)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0:
                            model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0:
                            model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0:
                            model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0:
                            model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0:
                            model=DecisionTreeClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0:
                            model=DecisionTreeClassifier(criterion=criterion)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0:
                            model=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0:
                            model=DecisionTreeClassifier()
                        model.fit(X_train,y_train)
                        Accuracy.append(model.score(X_test,y_test))
                    output=output+"Accuracy :"+str(np.mean(Accuracy))+"\n\n"
                    output=output+"Normal validation"+"\n\n"
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(max_depth=max_depth)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier()
                    model.fit(X,y)
                    Y_predicted=model.predict(X)
                    output=output+"Predicted "+str(sum((y==Y_predicted)))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                    cm=confusion_matrix(Y_predicted,y,labels=[1,0])
                    if cm[0,0]+cm[0,1]!=cm[0,0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                    output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                    if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                    output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"
                    model1=DecisionTreeClassifier()
                    model1.fit(X,y)
                    fig=plt.figure(facecolor="lightgreen")
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    plt.title("Decision tree visualization\n")
                    tree.plot_tree(model,feature_names=fn)
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh2=gh2+str(img_str,'utf-8')


                elif train_test_split_validation==0 :
                    output=output+"Comes under Normal validation\n\n"
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(max_depth=max_depth)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier(criterion=criterion)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0:
                        model=DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0:
                        model=DecisionTreeClassifier()
                    model.fit(X,y)
                    Y_predicted=model.predict(X)
                    output=output+"Predicted "+str(sum((y==Y_predicted)))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                    output=output+"Predicted "+str(round((sum((y==Y_predicted))/y.shape[0])*100))+"% Correctly\n\n"
                    cm=confusion_matrix(Y_predicted,y,labels=[1,0])
                    if cm[0,0]+cm[0,1]!=cm[0,0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                    output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                    if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                    output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"
                    fig=plt.figure(facecolor="lightgreen")
                    ax=plt.axes()
                    ax.set_facecolor('lightgreen')
                    tree.plot_tree(model,feature_names=fn)
                    plt.title("Decision tree visualization\n")
                    fig.canvas.draw()
                    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    pil_im=Image.fromarray(img)
                    buff=io.BytesIO()
                    pil_im.save(buff,format="PNG")
                    img_str=base64.b64encode(buff.getvalue())
                    gh2=gh2+str(img_str,'utf-8')

                model=DecisionTreeClassifier()
                model.fit(X,y)
                ft_imp=pd.Series(model.feature_importances_,index=fn).sort_values(ascending=False)
                output=output+"Feature Importance:\n\n"
                output=output+str(ft_imp.head(len(fn)))+"\n\n"
                fig = plt.figure(facecolor='lightgreen')
                plt.imshow(cm)
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.ylabel('True Label')
                plt.xlabel('Predicated Label')
                fig.canvas.draw()
                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                pil_im=Image.fromarray(img)
                buff=io.BytesIO()
                pil_im.save(buff,format="PNG")
                img_str=base64.b64encode(buff.getvalue())
                gh1=gh1+str(img_str,'utf-8')
                u=model.predict(X)
                output=output+"Predicted output for X's:"+str(model.predict(X))+"\n\n"
                if pre!="":
                        output=output+"User given datum predicted:\n\n"   
                        output=output+str(model.predict(pred))+"\n\n"
                output=output+"Confusion matrix :"+"\n"
                output=output+"\t\t\t\t\t\t\tpredicted label"+"\n"
                tl=["True","label"," "]
                k=0
                for i in cm:
                        if k<2:
                                output=output+tl[k]+"\t\t"+str(i)+"\n"
                        else:
                                output=output+"\t\t\t\t\t"+"\t\t\t\t\t"+str(i)+"\n"
                        k+=1
                return output,gh1,gh2
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error",""



def rf(fnn,Xs,Y,cini,tt,sty,sc,md="",msl="",mln="",ens="",mf="",ccv="",rs="",ts="",pre=""):
        try:
                
                output=""
                gh=""
                fn=list(map(str,fnn.split(',')))
                b=Xs.split(';')
                x1=[list(map(float,i.split(','))) for i in b]
                yy=list(map(float,Y.split(',')))
                if cini==0:
                        criterion='gini'
                else:
                        criterion='entropy'
                if md!="":
                        max_depth=list(map(int,md.split(',')))
                else:
                        max_depth=[]
                if msl!="":
                        min_samples_leaf=list(map(int,msl.split(',')))
                else:
                        min_samples_leaf=[]
                if mln!="":
                        max_leaf_nodes=list(map(int,mln.split(',')))
                else:
                        max_leaf_nodes=[]
                if ens!="":
                        n_estimators=list(map(int,mln.split(',')))
                else:
                        n_estimators=[]
                if mf!="":
                        max_features=list(map(int,mln.split(',')))
                else:
                        max_features=[]
                s=['accuracy','f1','precision','recall']
                scoring=s[sc]
                if ccv!="":
                        cv=int(ccv)
                else:
                        cv=3
                if sty==0:
                        stratifyy='y'
                else:
                        stratifyy=''
                if ts!="":
                        test_size=int(ts)
                else:
                        test_size=0
                if rs!="":
                        random_state=int(rs)
                else:
                        random_state=0
                train_test_split_validation=tt
                if pre!="":
                        bb=pre.split(';')
                        pred=[list(map(float,i.split(','))) for i in bb]







                X=np.array(x1)
                y=np.array(yy)
                for i in range(len(max_features)):
                    if max_features[i]>len(X):
                        max_features[i]=len(X)
                X=X.transpose()
                if stratifyy=='y':
                    stratify=y
                else:
                    stratifyy=''
                if len(np.unique(y))<cv or cv>len(y) or cv==1 or cv==0:
                    cv=len(np.unique(y))
                if test_size>len(y) or test_size==0 or len(np.unique(y))>test_size :
                    test_size=len(np.unique(y))





                    
                    
                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes!=[] and n_estimators!=[] and max_features!=[]:
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes,
                    'n_estimators':n_estimators,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes!=[] and n_estimators!=[] and max_features==[]:
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes,
                    'n_estimators':n_estimators
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes!=[] and n_estimators==[] and max_features!=[]:
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes!=[] and n_estimators==[] and max_features==[]:
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes==[] and n_estimators!=[] and max_features!=[]: 
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf,
                    'n_estimators':n_estimators,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes==[] and n_estimators!=[] and max_features==[]: 
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf,
                    'n_estimators':n_estimators
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes==[] and n_estimators==[] and max_features!=[]: 
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf!=[] and max_leaf_nodes==[] and n_estimators==[] and max_features==[]: 
                    param={
                    'max_depth':max_depth,
                    'min_samples_leaf':min_samples_leaf
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes!=[] and n_estimators!=[] and max_features!=[]: 
                    param={
                    'max_depth':max_depth,
                    'max_leaf_nodes':max_leaf_nodes,
                    'n_estimators':n_estimators,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes!=[] and n_estimators!=[] and max_features==[]: 
                    param={
                    'max_depth':max_depth,
                    'max_leaf_nodes':max_leaf_nodes,
                    'n_estimators':n_estimators
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes!=[] and n_estimators==[] and max_features!=[]: 
                    param={
                    'max_depth':max_depth,
                    'max_leaf_nodes':max_leaf_nodes,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes!=[] and n_estimators==[] and max_features==[]: 
                    param={
                    'max_depth':max_depth,
                    'max_leaf_nodes':max_leaf_nodes
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes==[] and n_estimators!=[] and max_features!=[]: 
                    param={
                    'max_depth':max_depth,
                    'n_estimators':n_estimators,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes==[] and n_estimators!=[] and max_features==[]: 
                    param={
                    'max_depth':max_depth,
                    'n_estimators':n_estimators
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes==[] and n_estimators==[] and max_features!=[]: 
                    param={
                    'max_depth':max_depth,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth!=[] and min_samples_leaf==[] and max_leaf_nodes==[] and n_estimators==[] and max_features==[]: 
                    param={
                    'max_depth':max_depth
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes!=[] and n_estimators!=[] and max_features!=[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes,
                    'n_estimators':n_estimators,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes!=[] and n_estimators!=[] and max_features==[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes,
                    'n_estimators':n_estimators
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                    
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes!=[] and n_estimators==[] and max_features!=[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes!=[] and n_estimators==[] and max_features==[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf,
                    'max_leaf_nodes':max_leaf_nodes
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes==[] and n_estimators!=[] and max_features!=[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf,
                    'n_estimators':n_estimators,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes==[] and n_estimators!=[] and max_features==[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf,
                    'n_estimators':n_estimators
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes==[] and n_estimators==[] and max_features!=[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf!=[] and max_leaf_nodes==[] and n_estimators==[] and max_features==[]: 
                    param={
                    'min_samples_leaf':min_samples_leaf
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf==[] and max_leaf_nodes!=[] and n_estimators!=[] and max_features!=[]: 
                    param={
                    'max_leaf_nodes':max_leaf_nodes,
                    'n_estimators':n_estimators,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf==[] and max_leaf_nodes!=[] and n_estimators!=[] and max_features==[]: 
                    param={
                    'max_leaf_nodes':max_leaf_nodes,
                    'n_estimators':n_estimators
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf==[] and max_leaf_nodes!=[] and n_estimators==[] and max_features!=[]: 
                    param={
                    'max_leaf_nodes':max_leaf_nodes,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf==[] and max_leaf_nodes==[] and n_estimators!=[] and max_features!=[]: 
                    param={
                    'n_estimators':n_estimators,
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf==[] and max_leaf_nodes==[] and n_estimators!=[] and max_features==[]: 
                    param={
                    'n_estimators':n_estimators
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    output=output+"best_params:"+str(best_params)+"\n\n"
                if max_depth==[] and min_samples_leaf==[] and max_leaf_nodes==[] and n_estimators==[] and max_features!=[]: 
                    param={
                    'max_features':max_features
                    }
                    dt=RandomForestClassifier()
                    if scoring!='':
                        gs=GridSearchCV(dt,param,scoring=scoring,cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                    else:
                        gs=GridSearchCV(dt,param,scoring='f1',cv=cv)
                        gs.fit(X,y)
                        best_params=gs.best_params_
                        output=output+"best_params:"+str(best_params)+"\n\n"


                if max_depth!=[]:
                    max_depth=best_params['max_depth']
                else:
                    max_depth=0
                    
                if min_samples_leaf!=[]:
                    min_samples_leaf=best_params['min_samples_leaf']
                else:
                    min_samples_leaf=0
                if max_leaf_nodes!=[]:
                    max_leaf_nodes=best_params['max_leaf_nodes']
                else:
                    max_leaf_nodes=0
                if n_estimators!=[]:
                    n_estimators=best_params['n_estimators']
                else:
                    n_estimators=0
                if max_features!=[]:
                    max_features=best_params['max_features']
                else:
                    max_features=0


                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)    
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)    
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators)
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_features=max_features)
                if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,max_depth=max_depth)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(max_depth=max_depth)
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)    
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_features=max_features)
                if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_features=max_features)
                if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(min_samples_leaf=min_samples_leaf)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators,max_features=max_features)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(criterion=criterion,max_features=max_features)
                if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(criterion=criterion)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0: 
                    model=RandomForestClassifier(n_estimators=n_estimators,max_features=max_features)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0: 
                    model=RandomForestClassifier(n_estimators=n_estimators)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0: 
                    model=RandomForestClassifier(max_features=max_features)
                if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0: 
                    model=RandomForestClassifier()



                Accuracy=[]

                if train_test_split_validation==1:
                    output=output+"Comes under train_test_split validation\n\n"
                    if stratifyy!='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify,test_size=test_size)
                    if stratifyy!='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,stratify=stratify)
                    if stratifyy!='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=stratify)
                    if stratifyy=='' and test_size!=0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state,test_size=test_size)
                    if stratifyy=='' and test_size!=0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
                    if stratifyy=='' and test_size==0 and random_state!=0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state)
                    if stratifyy=='' and test_size==0 and random_state==0:
                        X_train,X_test,y_train,y_test=train_test_split(X,y)
                    model.fit(X_train,y_train)
                    Y_predicted=model.predict(X_test)
                    output=output+"Predicted "+str(sum((y_test==Y_predicted)))+" Correctly Out of "+str(y_test.shape[0])+"\n\n"
                    output=output+"Predicted "+str(round((sum((y_test==Y_predicted))/y_test.shape[0])*100))+"% Correctly\n\n"
                    cm=confusion_matrix(Y_predicted,y_test,labels=[1,0])
                    if cm[0,0]+cm[0,1]!=cm[0,0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                    output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                    if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                    output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"
                    
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier()
                    model.fit(X,y)
                    Y_predicted=model.predict(X)

                elif train_test_split_validation==2:
                    output=output+"Comes under K-fold cross validation\n\n"
                    kf=KFold(n_splits=cv,shuffle=True)
                    for train_index,test_index in kf.split(X):
                        X_train,X_test=X[train_index],X[test_index]
                        y_train,y_test=y[train_index],y[test_index]
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_depth=max_depth)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                        if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(max_depth=max_depth)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_features=max_features)
                        if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_features=max_features)
                        if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(min_samples_leaf=min_samples_leaf)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators,max_features=max_features)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(criterion=criterion,max_features=max_features)
                        if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(criterion=criterion)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                            model=RandomForestClassifier(n_estimators=n_estimators,max_features=max_features)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                            model=RandomForestClassifier(n_estimators=n_estimators)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                            model=RandomForestClassifier(max_features=max_features)
                        if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                            model=RandomForestClassifier()
                        model.fit(X_train,y_train)
                        Accuracy.append(model.score(X_test,y_test))
                    output=output+"Accuracy :"+str(np.mean(Accuracy))+"\n\n"
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_depth=max_depth)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_depth=max_depth)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(criterion=criterion,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(criterion=criterion)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model=RandomForestClassifier(n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model=RandomForestClassifier(n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model=RandomForestClassifier(max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model=RandomForestClassifier()
                    model.fit(X,y)
                    Y_predicted=model.predict(X)
                    output=output+"Predicted "+str(sum((y==Y_predicted)))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                    cm=confusion_matrix(Y_predicted,y,labels=[1,0])
                    if cm[0,0]+cm[0,1]!=cm[0,0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives "+"\n\n"
                    output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                    if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                    else:
                        output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                    output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"
                    

                if  1:
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_depth=max_depth)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(max_depth=max_depth,max_features=max_features)
                    if max_depth!=0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(max_depth=max_depth)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(min_samples_leaf=min_samples_leaf,max_features=max_features)
                    if max_depth==0 and min_samples_leaf!=0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(min_samples_leaf=min_samples_leaf)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(criterion=criterion,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion!='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(criterion=criterion)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes!=0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier(max_leaf_nodes=max_leaf_nodes)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features!=0:
                        model1=RandomForestClassifier(n_estimators=n_estimators,max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators!=0 and max_features==0:
                        model1=RandomForestClassifier(n_estimators=n_estimators)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features!=0:
                        model1=RandomForestClassifier(max_features=max_features)
                    if max_depth==0 and min_samples_leaf==0 and criterion=='' and max_leaf_nodes==0 and n_estimators==0 and max_features==0:
                        model1=RandomForestClassifier()
                    model1.fit(X,y)
                    Y_predicted=model1.predict(X)
                    if train_test_split_validation==0:
                            output=output+"Comes under normal validation\n\n"
                            output=output+"Predicted "+str(sum((y==Y_predicted)))+" Correctly Out of "+str(y.shape[0])+"\n\n"
                            output=output+"Predicted "+str(round((sum((y==Y_predicted))/y.shape[0])*100))+"% Correctly\n\n"
                            cm=confusion_matrix(Y_predicted,y,labels=[1,0])
                            if cm[0,0]+cm[0,1]!=cm[0,0]:
                                output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively but from that "+str(cm[0,0])+" are actualy positives remaining all negatives\n\n"
                            else:
                                output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positively which predicted all are actualy positives\n\n"
                            output=output+"So Precision is "+str((cm[0,0]/(cm[0,0]+cm[0,1])))+"\n\n"
                            if cm[0,0]+cm[0,1]!=cm[0,0]+cm[1][0]:
                                output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly but total actualy positives are "+str(cm[0,0]+cm[1][0])+"\n\n"
                            else:
                                output=output+"Predicted "+str(cm[0,0]+cm[0,1])+" Positive correctly which is equal to total actualy positives given in data\n\n"
                            output=output+"So recall is "+str((cm[0,0]/(cm[0,0]+cm[1,0])))+"\n\n"  

                
                ft_imp=pd.Series(model1.feature_importances_,index=fn).sort_values(ascending=False)
                fig = plt.figure(facecolor='lightgreen')
                plt.imshow(cm)
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.ylabel('True Label')
                plt.xlabel('Predicated Label')
                fig.canvas.draw()
                img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
                img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                pil_im=Image.fromarray(img)
                buff=io.BytesIO()
                pil_im.save(buff,format="PNG")
                img_str=base64.b64encode(buff.getvalue())
                gh=gh+str(img_str,'utf-8')
                output=output+"Predicted output of given X's:"+str(Y_predicted)+"\n\n"
                output=output+"Feature Importance:\n\n"
                output=output+str(ft_imp.head(len(fn)))+"\n\n"
                if pre!="":
                        output=output+"User given datum predicted:\n\n"   
                        output=output+str(model1.predict(pred))+"\n\n"
                output=output+"Confusion matrix :"+"\n"
                output=output+"\t\t\t\t\t\t\tpredicted label"+"\n"
                tl=["True","label"," "]
                k=0
                for i in cm:
                        if k<2:
                                output=output+tl[k]+"\t\t"+str(i)+"\n"
                        else:
                                output=output+"\t\t\t\t\t"+"\t\t\t\t\t"+str(i)+"\n"
                        k+=1
                return output,gh
        except:
                output="Something went wrong... Please give correct input..."
                return output,"error"
                


            
