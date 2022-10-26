import numpy as np 
import matplotlib.pyplot as plt 
import torch
  
# Y-axis values 
y1 = [2, 3, 4.5] 
  
# Y-axis values  
y2 = [1, 1.5, 5] 
  
# Function to plot   
plt.plot(y1, label='blue') 
plt.plot(y2, label ='green') 
  
# Function add a legend   
plt.legend(loc ="lower right") 
  
# function to show the plot 
plt.show()
print("hi"+"nihao")

a = np.array(
    [[1,2,3],
     [4,5,6],
     [7,8,9]]
)

a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])