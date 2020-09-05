import numpy as np
#it shows version of numpy
print(np.__version__)

#it shows in array 10 zeros
print(np.zeros((3,10),dtype='int'))

#it show ones 3*5 matrix
print(np.ones((3,5),dtype=int))

#it shows the space values betweem values
print(np.linspace(0,5,10))

#it shows the diagnol matrix is 1
print(np.eye(1))

#it shows none value
print(np.random.seed())

#it show the  int values a number
print(np.random.randint(10,size=(3)))

print(np.random.randint(10,size=(3,4)))

print(np.random.randint(10,size=(3,4,5)))

# array indexing
x1=np.array([[4,8,5,6,7,9],[0,1,5,7,8,2]])
print(x1)
#
print(x1[1])
#
print(x1[0,-2])
#slicing  all elements
print(x1[:])
#same
print(x1[::])

print(x1[::2])

#slicing in different ways
import numpy as np

x=np.arange(10)
print(x)
print(x[2])
print(x[::2])
print(x[:-1])
print(x[::-1])

#array concatenation
a=np.array([1,2,3])
b=np.array([3,2,1])
z=[21,22,23]
print(np.concatenate([a,b,z]))



#n dimensional array
import numpy as np
a=np.array([(1,2,3),(4,5,6)])
print(a)
print(a.ndim)

#image 
from IPython.display import Image
i=Image(filename='C:/Users/Public/Pictures/Sample Pictures/koala.jpg')
print(i)

#three d array
three=np.array([[[1,2,3],[4,5,6]],[[7,8,9,],[10,11,12]]])
print(three)
print('returns the second list inside first list {}'.format(three[0][1]))
print(three[0])
 
#copied values 
copy=three[0].copy()
three[0]=99
print('New value of three : {}'.format(three))
three[0]=copy
print('three again: {}'.format(three))

#matrix slicing
matrix=np.array([[3,4,5],[6,7,8],[9,5,1]])
print(matrix)
print(matrix[:2,1:])
print(matrix[:,2:])
print(matrix[:,:2])
print(matrix[1:,1:])
#print(matrix[[0,1][1,0][1,2]])
print(matrix[:,:1])
print(matrix[:,1:2])


# Boolean 
pers=np.array(['Manu','Jeevan','Prakash','Jeevan','Manu'])
print(pers)
print(pers=='Manu')

#random checking
from numpy import random
random_no=random.randn(5,4)
print(random_no)
print(random_no[pers=='Manu'])
print(random_no[pers=='Manu',2:])

#
new_variable=(pers=='Manu')|(pers=='Jeevan')
print(new_variable)
print(random_no[new_variable])
random_no[random_no<0]=0
print(random_no)
random_no[pers!='Manu']=9
print(random_no)

#Fancy indexing
from numpy import random
algebra=random.randn(7,4)
print(algebra)
for j in range(7):
    print(j)
    algebra[j]=j
print(algebra)
print(algebra[[4,5,1]])

#
fancy=np.arange(36).reshape(9,4)
print(fancy)
# inside array
print(fancy[[1,4,3,2]][:,[3,2,1,0]])

#another mathod for above
print(fancy[np.ix_([1,4,3,2],[3,2,1,0])])

#Transporing Arrays
transpose=np.arange(12).reshape(3,4)
print(transpose)
print(transpose.T)
print(np.dot(transpose.T,transpose))

#universal functions

funky=np.arange(8)
print(funky)
print(np.sqrt(funky))
print(np.exp(funky))

x=np.random.random(10)
y=np.random.randn(10)
print(x)
print(y)
print(np.maximum(x,y))
print(np.modf(y)) #function modf returns the fractional and integral part of floting point arrays


#binary functions
#Data Processing using Arrays
matrices=np.arange(-5,5,1)
print(matrices)
x,y=np.meshgrid(matrices,matrices)
print('matrix values of y :{}'.format(y))
print('matrix values of x :{}'.format(x))

#zip lists
l1=[1,2,3,4,5]
l2=['a','b','c','d','e']
l3=['Sun','Mon','Tue','Wed','Thr']
zip_result=zip(l1,l2,l3)
print(zip_result)
for x,y,z in zip_result:
    print(x,y,z)

x1=np.array([1,2,3,4,5])
y1=np.array([6,7,8,9,10])
cond=[True,False,True,False,True]
z1=[(x,y,z) for x,y,z in zip(x1,y1,cond)]
print(z1)
print(np.where(cond,x1,y1))

#statistical methods
thie=np.random.randn(5,5)
print(thie)
print(thie.mean())
print(np.mean(thie))
print(thie.sum())

#sum of arrays
jp=np.arange(12).reshape(4,3)
print('The arrays are :{}'.format(jp))
print('The sum of rows are :{}'.format(jp.sum(0)))
print('The sum of Columns are :{}'.format(jp.sum(1)))
print(jp.cumsum(0))
print(jp.cumsum(1))
xp=np.random.randn(100)
print (xp)
print(xp>0).sum()
print(xp<0).sum()
tanf=np.array([True,False,True,False,True,False])
print(tanf.any()) #checks if any of the values are true
print(tanf.all()) #return false even if a single value is false

#Sorting
lp=np.random.randn(8)
print(lp)
print(lp.sort())
print(lp)

tp=np.random.randn(4,4)
print(tp)
print(tp.sort(1))
print(tp)                                                                                                                                                             

#unique names in array 
name=np.array(['Manu','Abdul','Prakash','Ameer','Abdul','Prakash'])
print(np.unique(name))

print(set(name))

print(name=='Manu')


print(np.in1d(name,['Manu'])) #in1d function checks for the value
                                                                             
#Linear algebra 


newSeries = pd.Series([10,20,30,40],index=['LONDON','NEWYORK','Washington','Manchester'])
newSeries1 = pd.Series([10,20,35,46],index=['LONDON','NEWYORK','Istanbul','Karachi'])
print(newSeries,newSeries1)


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        







        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               