import random
import numpy as np

class fuzzyClustering:
    def fit(self,data,n_clusters,fuzziness_parameter=2,stop_value=0.01):
        #Step 1: Initialize the data points into desired number of clusters randomly.
        self.n_clu=n_clusters
        self.dt=data
        self.init=[]
        self.fuzzi_para=fuzziness_parameter
        for i in range(self.n_clu):
            a=[]
            for j in range(len(self.dt)):
                a.append(random.randint(0,10)/10)
            self.init.append(a)


        Xs=[]
        for i in range(len(self.dt[0])):
            x=[]
            for j in self.dt:
                x.append(j[i])
            Xs.append(x)

        cmp_centroids=[]
        Vij=[]                        #Centroids [[data_points1(x1=[1,2],x2,.)XΣi=1(i*C1'values)/sum(C1'values)]]
        do=1
        for i in range(len(self.init)):
            for j in range(len(self.init[i])):
                self.init[i][j]=round(self.init[i][j]**self.fuzzi_para,2)  #membership_value+m
                
        for i in self.init:
            xs=[]
            for j in Xs:
                mm=0
                for a,b in zip(i,j):
                    mm=mm+(a*b)
                xs.append(round(mm/sum(i),3))
            cmp_centroids.append(xs) 
        while cmp_centroids!=Vij and do==1:
            cmp_centroids.clear()
            for i in Vij:cmp_centroids.append(i)
            Vij.clear()
            #Step 2: Find out the centroid.
            #(x1,x2)
            #Vij i=ith cluster j=jth datapoint(x1,x2...)
            # formula Vij= CΣk=1( 2Σb=1 ((nΣ1 membership_value+m of Ck * xb)/(nΣ1 membership+m of Ck))) for 2 datapoints
            #m is the fuzziness parameter (generally taken as 2)
            for i in range(len(self.init)):
                for j in range(len(self.init[i])):
                    self.init[i][j]=round(self.init[i][j]**self.fuzzi_para,2)  #membership_value+m
            for i in self.init:
                xs=[]
                for j in Xs:
                    mm=0
                    for a,b in zip(i,j):
                        mm=mm+(a*b)
                    xs.append(round(mm/sum(i),3))
                Vij.append(xs)
            #print(Vij)

                
            #Step 3: Find out the distance of each point from centroid.
            distance=[]         # [[data_points1(1,2)->center1,data_points1(1,2)->center2],[]...]
            for i in self.dt:
                ds=[]
                for j in Vij:
                    aa=round(sum((np.array(i)-np.array(j))**2)**0.5,2)
                    ds.append(aa)
                distance.append(ds)
            #print(distance)

                
            #Step 4: Updating membership values.
            #formula μij=CΣi=1(nΣj=1((Dik^2/Djk^2)^(1/m-1))^-1)
            #Each membership value is inversely proportional to the distance of each dataPoint from each centres So if distance is high then membership value is low
            mus=[]
            for k in range(self.n_clu):
                h=[]
                for i in distance:
                    mu=i[k]
                    mu1=0
                    for j in i:
                        if j!=0:
                            mu1=mu1+((mu/j)**2)**(1/(self.fuzzi_para-1))
                        else:
                            mu1=mu1+0
                    if mu1!=0 and str(round(mu1**-1,3))!='nan':
                        if mu1!=1:
                            h.append(round(mu1**-1,3))
                        else:
                            h.append(0)
                    else:
                        h.append(1)
                mus.append(h)
            if self.init==mus or max(sum(np.array(mus)-np.array(self.init)))<=stop_value:
                do=0
            else:
                self.init=mus
        return mus
    
    
        
            
    
    

    
    
            
        
    
            
