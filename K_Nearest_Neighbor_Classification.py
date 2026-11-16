import math
rows=[(5.1,3.5,1.4,0.2,"Setosa"),(4.9,3.0,1.4,0.2,"Setosa"),(5.0,3.4,1.5,0.2,"Setosa"),(6.4,3.2,4.5,1.5,"Versicolor"),(6.9,3.1,4.9,1.5,"Versicolor"),(5.9,3.0,4.2,1.5,"Versicolor"),(6.3,3.3,6.0,2.5,"Virginica"),(6.5,3.0,5.8,2.2,"Virginica"),(7.1,3.0,5.9,2.1,"Virginica")]
print("Records:",len(rows),"Classes:",sorted(set(r[4] for r in rows)))
for col in range(4):
 vals=[r[col] for r in rows]; print("Feature",col+1,"min=",min(vals),"max=",max(vals))
correct=0
for i,r in enumerate(rows):
 best=(float("inf"),None)
 for j,q in enumerate(rows):
  if i==j: continue
  d=math.sqrt(sum((r[k]-q[k])**2 for k in range(4)))
  if d<best[0]: best=(d,q[4])
 correct+=best[1]==r[4]; print("Record",i+1,"actual=",r[4],"predicted=",best[1])
print("1-NN leave-one-out accuracy:",round(100*correct/len(rows),2),"%")
print("WEKA/KNIME: load dataset -> normalize -> classifier -> cross-validation -> compare accuracy/confusion matrix.")
