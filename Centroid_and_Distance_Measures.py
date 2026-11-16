import math
points=[("P1",2,3,4),("P2",3,5,2),("P3",8,7,6),("P4",7,8,5),("P5",1,2,3),("P6",9,6,7)]
centroid=[sum(p[i] for p in points)/len(points) for i in range(1,4)]
def eu(p,c): return math.sqrt(sum((p[i]-c[i])**2 for i in range(3)))
def man(p,c): return sum(abs(p[i]-c[i]) for i in range(3))
def mink(p,c,k=3): return sum(abs(p[i]-c[i])**k for i in range(3))**(1/k)
print("CENTROID:",[round(x,4) for x in centroid])
for n,x,y,z in points:
 p=(x,y,z); print(n,"Euclidean=",round(eu(p,centroid),4),"Manhattan=",round(man(p,centroid),4),"Minkowski(k=3)=",round(mink(p,centroid),4))
print("\nPAIRWISE EUCLIDEAN DISTANCE MATRIX")
for p in points: print(p[0]," ".join(f"{eu(p[1:],q[1:]):.3f}" for q in points))
print("Closest point:",min(points,key=lambda p:eu(p[1:],centroid))[0])
print("Farthest point:",max(points,key=lambda p:eu(p[1:],centroid))[0])
