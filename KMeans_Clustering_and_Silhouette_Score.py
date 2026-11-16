import math
points=[("P1",1,1),("P2",1,2),("P3",2,1),("P4",2,2),("P5",8,8),("P6",8,9),("P7",9,8),("P8",9,9)]
centroids=[[1.,1.],[8.,8.]]
def d(p,c): return math.sqrt((p[1]-c[0])**2+(p[2]-c[1])**2)
for iteration in range(10):
 clusters=[[],[]]
 for p in points: clusters[0 if d(p,centroids[0])<=d(p,centroids[1]) else 1].append(p)
 new=[]
 for g in clusters: new.append([sum(p[1] for p in g)/len(g),sum(p[2] for p in g)/len(g)])
 print("Iteration",iteration+1,"Cluster1",[p[0] for p in clusters[0]],"Cluster2",[p[0] for p in clusters[1]],"Centroids",new)
 if new==centroids: break
 centroids=new
labels={p[0]:0 for p in clusters[0]}; labels.update({p[0]:1 for p in clusters[1]})
scores=[]
for p in points:
 same=[q for q in points if labels[q[0]]==labels[p[0]] and q[0]!=p[0]]
 other=[q for q in points if labels[q[0]]!=labels[p[0]]]
 a=sum(d(p,[q[1],q[2]]) for q in same)/len(same); b=sum(d(p,[q[1],q[2]]) for q in other)/len(other)
 scores.append((b-a)/max(a,b))
print("FINAL CENTROIDS:",centroids)
print("Average silhouette score:",round(sum(scores)/len(scores),4))
