
# coding: utf-8

# # Python Basics

# In[2]:


# Printing a string
print("Hello, Digital Worlds")


# ### Variables

# In[2]:


# defining a variable : In Python there is no need to mention the data type

var1 = 10      # An integer assignment
var2 = 3.146   # A floating point
var3 = "Hello" # A string

print(var1,' ',var2,' ',var3)


# ### Assignment

# In[3]:


# Assigning same value to multiple variables

var1 = var2 = var3 = 1
print(var1,' ',var2,' ',var3)

# Assigning Different values to variable in a single expression

var1, var2, var3 = 1, 2.5, "john"
print(var1,' ',var2,' ',var3)

# Note: commas can be used for multi-assignments


# ### Slicing

# In[4]:


# String operations

str = 'Hello World!'  # A string

print(str)          # Prints complete string
print(str[0])       # Prints first character of the string
print(str[2:5])     # Prints characters starting from 3rd to 5th
print(str[2:])      # Prints string starting from 3rd character
print(str[:2])
print(str * 2)      # Prints string two times
print(str + "TEST") # Prints concatenated string


# ### Data types

# In[5]:


# Python Lists
list = [ 'abcd', 786 , 2.23, 'john', 70.2 ]  # A list
tuple = ( 'abcd', 786 , 2.23, 'john', 70.2  ) # A tuple. Tuples are immutable, i.e. cannot be edit later

print(list)            # Prints complete list
print(list[0])         # Prints first element of the list
print(tuple[1:3])        # Prints elements starting from 2nd till 3rd 


# In[6]:


# Lists are ordered sets of objects, whereas dictionaries are unordered sets. But the main difference is that items in dictionaries are accessed via keys and not via their position.
tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127
print(tel)
print(tel['jack'])
del tel['sape']
tel['irv'] = 4127
print(tel)
print(tel.keys())
print(sorted(tel.keys()))
print(sorted(tel.values()))
print('guido' in tel)
print('jack' not in tel)


# ###  Conditioning and looping

# In[7]:


# Square of odd numbers

for i in range(0,10):
    if i%2 == 0:
        print("Square of ",i," is :",i)
    else:
        print(i,"is an odd number")


# ### Built-in Functions

# In[8]:


print("Sum of array: ",sum([1,2,3,4]))
print("Length of array: ",len([1,2,3,4]))
print("Absolute value: ",abs(-1234))
print("Round value: ",round(1.2234))

import math as mt      # importing a package
print("Log value: ",mt.log(10))


# ### Functions

# In[9]:


def area(length,width):
    return length*width

print("Area of rectangle:",area(10,20))


# ### Broadcasting
# * Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes
# 
# ### NumPy 
# * Numpy is the fundamental package for scientific computing with Python. It contains among other things:
# * a powerful N-dimensional array object
# * sophisticated (broadcasting) functions
# * tools for integrating C/C++ and Fortran code
# * useful linear algebra, Fourier transform, and random number capabilities

# In[10]:


import numpy as np   # Importing libraries

a = np.array([0, 1, 2])
b = np.array([5, 5, 5])

print("Matrix A\n", a)
print("Matrix B\n", b)

print("Regular matrix addition A+B\n", a + b)

print("Addition using Broadcasting A+5\n", a + 5)


# ### Broadcasting Rules
# When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when
# 
# 1. they are equal, or
# 2.  one of them is 1
# 

# In[11]:


# Lets go for a 2D matrix
c = np.array([[0, 1, 2],[3, 4, 5],[6, 7, 8]])
d = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])

e = np.array([1, 2, 3])

print("Matrix C\n", c)
print("Matrix D\n", d)
print("Matrix E\n", d)

print("Regular matrix addition C+D\n", c + d)

print("Addition using Broadcasting C+E\n", c + e)


# In[12]:


M = np.ones((3, 3))
print("Matrix M:\n",M)


# In[13]:


print("Dimension of M: ",M.shape)
print("Dimension of a: ",a.shape)
print("Addition using Broadcasting")
print(M + a)
# Broadcasting array with matrix


# ## All in one program

# In[14]:


# Importing libraries
import timeit

# Usage of builtin functions
start = timeit.default_timer()   

# Defining a list
array_list = [10,11,15,19,21,32]      
array_np_list = []

# Print the list
print("Original List",array_list,"\n")   

# Defining a function
def prime(num):      
    if num > 1:     
        
        # check for factors
        # Iterating a range of numbers
        for i in range(2,num):    
            if (num % i) == 0:
                
                # Appending data to list
                array_np_list.append(num)           
                print(num,"is not a prime number (",i,"times",num//i,"is",num,")")
                
                # Terminating a loop run
                break         
        else:
            print(num,"is a prime number")
            
# Iterating a list
for item in array_list:
    
    # Calling a function
    prime(item)         

print("\nNon-prime List",array_np_list,"\n")

end = timeit.default_timer()

# Computing running time
print("Time Taken to run the program:",end - start, "seconds")       


# ### Note:
# * Python is a procedural Language
# * Two versions of Python 2 vs 3
# * No braces. i.e. indentation
# * No need to explicitly mention data type

# ## Unvectorized vs Vectorized Implementations

# In[15]:


# Importing libraries
import numpy as np

# Defining matrices
mat_a = [[6, 7, 8],[5, 4, 5],[1, 1, 1]]
mat_b = [[1, 2, 3],[1, 2, 3],[1, 2, 3]]

# Getting a row from matrix
def get_row(matrix, row):
    return matrix[row]

# Getting a coloumn from matrix
def get_column(matrix, column_number):
    column = []
 
    for i in range(len(matrix)):
        column.append(matrix[i][column_number])
 
    return column

# Multiply a row with coloumn
def unv_dot_product(vector_one, vector_two):
    total = 0
 
    if len(vector_one) != len(vector_two):
        return total
 
    for i in range(len(vector_one)):
        product = vector_one[i] * vector_two[i]
        total += product
 
    return total

# Multiply two matrixes
def matrix_multiplication(matrix_one, matrix_two):
    m_rows = len(matrix_one)
    p_columns = len(matrix_two[0])
    result = []
    
    for i in range(m_rows):
        row_result = []
 
        for j in range(p_columns):
            row = get_row(matrix_one, i)
            column = get_column(matrix_two, j)
            product = unv_dot_product(row, column)
            
            row_result.append(product) 
        result.append(row_result)
        
    return result

print("Matrix A: ", mat_a,"\n")
print("Matrix B: ", mat_b,"\n")

print("Unvectorized Matrix Multiplication\n",matrix_multiplication(mat_a,mat_b),"\n")


# In[16]:


# Vectorized Implementation
npm_a = np.array(mat_a)
npm_b = np.array(mat_b)

print("Vectorized Matrix Multiplication\n",npm_a.dot(npm_b),"\n") 
# A.dot(B) is a numpy built-in function for dot product


# ### Tip:
# * Vectorization reduces number of lines of code
# * Always prefer libraries and avoid coding from scratch

# ## Essential Python Packages: Numpy, Pandas, Matplotlib

# In[17]:


# Load library
import numpy as np


# In[18]:


# Create row vector
vector = np.array([1, 2, 3, 4, 5, 6])
print("Vector:",vector)

# Select second element
print("Element 2 in Vector is",vector[1])


# In[19]:


# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("Matrix\n",matrix)

# Select second row
print("Second row of Matrix\n",matrix[1,:])
print("Third coloumn of Matrix\n",matrix[:,2])


# In[20]:


# Create Tensor
tensor = np.array([ [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
                    [[[3, 3], [3, 3]], [[4, 4], [4, 4]]] ])

print("Tensor\n",tensor)


# ### Matrix properties

# In[21]:


# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("Matrix Shape:",matrix.shape)
print("Number of elements:",matrix.size)
print("Number of dimentions:",matrix.ndim)
print("Average of matrix:",np.mean(matrix))
print("Maximum number:",np.max(matrix))
print("Coloumn with minimum numbers:",np.min(matrix, axis=1))
print("Diagnol of matrix:",matrix.diagonal())
print("Determinant of matrix:",np.linalg.det(matrix))


# ### Matrix Operations

# In[22]:


print("Flattened Matrix\n",matrix.flatten())
print("Reshaping Matrix\n",matrix.reshape(9,1))
print("Inversed Matrix\n",np.linalg.inv(matrix))
print("Transposed Matrix\n",matrix.T)


# In[23]:


# Create matrix
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])

# Create matrix
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

print("Matrix Addition\n",np.add(matrix_a, matrix_b))
print("Scalar Multiplication\n",np.multiply(matrix_a, matrix_b))
print("Matrix Addition\n",np.dot(matrix_a, matrix_b))


# ### Pandas

# In[24]:


import pandas as pd


# In[25]:


df=pd.read_csv("Income.csv")
print("Data\n")
df


# In[26]:


print("Top Elements\n")
df.head(3)


# In[27]:


print("Bottom Elements\n")
df.tail(3)


# In[28]:


print("Specific Coloumn\n")
df['State'].head(3)


# In[29]:


print("Replace negative numbers with NaN\n")
df.replace(-999,np.nan)


# ## Matplotlib

# In[5]:


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


# ### Line Plot

# In[31]:


# Line plot
plt.plot([1,2,3,4],[3,4,5,6])
plt.xlabel('some numbers')
plt.ylabel('some numbers')
plt.show()


# In[32]:


### Adding elements to line plots
t = np.arange(0.0, 2.0, 0.01) # Generate equally space numbers between 0 and 2
s = 1 + np.sin(2*np.pi*t)  # Apply sin function to the random numbers
plt.plot(t, s)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.savefig("test.png") # Save a plot. Check the directory
plt.show()


# ### Bar Plot

# In[7]:


y = [3, 10, 7, 5, 3, 4.5, 6, 8.1]
x = range(len(y))
width = 1/5
plt.bar(x, y, width, color="blue")
plt.show()


# ### Scatter Plot

# In[34]:


N = 50
# Generate random numbers
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()


# ### Histogram

# In[35]:


mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000) # Generate random values with some distribution

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()


# ### Pie Chart

# In[36]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

