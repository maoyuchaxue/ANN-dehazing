import matplotlib.pyplot as plt  
import numpy as np  
from scipy.interpolate import spline  
import pandas as pd

# T = np.array([6, 7, 8, 9, 10, 11, 12])  
# power = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])  
# xnew = np.linspace(T.min(),T.max(),300) #300 represents number of points to make between T.min and T.max  
  
# power_smooth = spline(T,power,xnew)  
  
# plt.plot(xnew,power_smooth)  
# plt.show()

def draw(filename, label, color):
    df = pd.read_csv(filename,header=0,sep=',')
    x = list(df['Step'])
    y = list(df['Value'])
    step = 400
    curstep = 0
    nextstep = step
    cnt = 0
    total = 0.0
    new_x = []
    new_y = []
    for i in range(len(x)):
        if x[i] > nextstep:
            if cnt > 0:
                new_x.append(curstep)
                new_y.append(total/cnt)
                cnt = 0
                total = 0.0
            curstep = nextstep
            nextstep += step
        cnt += 1
        total += y[i]
    #plt.plot(x, y, color=color, label=label)
    plt.plot(new_x, new_y, color, label=label)
# ""    
# plt.figure('g_loss_from_d')
# draw('./result/gan_eloss_no_ploss/g_loss_from_d.csv', 'dloss+eloss+tloss', 'r')
# draw('./result/gan_ploss_no_eloss/g_loss_from_d.csv', 'dloss+ploss+tloss', 'g')
# draw('./result/test_new_A/g_loss_from_d.csv', 'tloss', 'b')
# draw('./result/test1/g_loss_from_d.csv', 'dloss=0.1', 'y')
# draw('./result/test2/g_loss_from_d.csv', 'dloss=0.005', 'c')
# plt.xlabel('step')
# plt.title('G loss from D')
# plt.legend() # show label
# plt.show()

def showfig(name):
    plt.figure(name)
    draw('./result/gan_eloss_no_ploss/' + name + '.csv', 'no Lp', 'r')
    draw('./result/gan_ploss_no_eloss/' + name + '.csv', 'no Le', 'g')
    draw('./result/test_new_A/' + name + '.csv', 'no La', 'b')
    draw('./result/test1/' + name + '.csv', 'no Lt', 'y')
    #draw('./result/test2/' + name + '.csv', 'all', 'c')
    draw('./result/newG/' + name + '.csv', 'all', 'm')
    plt.xlabel('step')
    plt.title(name)
    plt.legend() # show label
    plt.xlim(0,10000) 
    plt.show()

showfig('d_loss')
showfig('d_fake_prob')
showfig('d_real_prob')
showfig('g_loss')
showfig('g_loss_from_d')
showfig('t_loss')
showfig('p_loss')
