import numpy as np     
import pandas as pd #para dataframes
import sys 
import time
import matplotlib.pyplot as plt

def main():
  
    print(sys.version)
    print(time.gmtime(0))

    # x axis values 
    x = [1,2,3] 
    # corresponding y axis values 
    y = [2,4,1] 
    
    # plotting the points  
    plt.plot(x, y) 
    
    # naming the x axis 
    plt.xlabel('x - axis') 
    # naming the y axis 
    plt.ylabel('y - axis') 
    
    # giving a title to my graph 
    plt.title('My first graph!') 
    
    # function to show the plot 
    plt.show() 