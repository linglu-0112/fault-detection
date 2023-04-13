import numpy as np
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.utils.fixes import parse_version
# %matplotlib inline 
plt.style.use('seaborn-darkgrid')
# 输入数据
## 


# 模型

# 训练

# 测试


#计算控制限
def compute_threshold(data, bw=None, alpha=0.99):
    data = data.reshape(-1,1)
    Min = np.min(data)
    Max = np.max(data)
    Range = Max-Min
    x_start = Min-Range
    x_end = Max+Range
    nums = 2**15
    dx = (x_end-x_start)/(nums-1)
    data_plot = np.linspace(x_start, x_end, nums)
    if bw is None:
        data_median = np.median(data)
        new_median = np.median(np.abs(data-data_median))/0.6745 + 0.00000001
        bw = new_median*((4/(3*data.shape[0]))**0.2)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data.reshape(-1,1))
    log_pdf = kde.score_samples(data_plot.reshape(-1, 1))
    pdf = np.exp(log_pdf)
    CDF = 0
    index = 0

    while CDF <= alpha:
        CDF += pdf[index]*dx
        index += 1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(data_plot, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
    ax.hist(data[:,0], 100, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
    ax.legend(loc='upper left')
    ax.set_xlim(data.min(), data.max())

    return data_plot[index]

#计算检测率和误报率
def getT2_DR_FDR(T_99_limit,t_fault,num_fault,num_normal,fault_point):
    num1,num2 = 0,0
    index = np.array(np.where(np.array(t_fault) > T_99_limit)).reshape(-1)
    num1 = (index > (fault_point-1)).astype(int).sum(0)
    DR = num1/num_fault
    num2 = (index < fault_point).astype(int).sum(0)
    FDR = num2/num_normal
    return DR,FDR
    
def getSPE_DR_FDR(SPE_99_limit,q_fault,num_fault,num_normal,fault_point):
    num1,num2 = 0,0
    index = np.array(np.where(np.array(q_fault) > SPE_99_limit)).reshape(-1)
    num1 = (index > (fault_point-1)).astype(int).sum(0)
    DR = num1/num_fault
    num2 = (index < fault_point).astype(int).sum(0)
    FDR = num2/num_normal
    return DR,FDR

#输出所有控制图 spe.shape(20,960),t2.shape(20,960)
def plot_fault_statistic(spe,t2,spe_lim,t2_lim):
   
    fig, ax = plt.subplots(5, 4,figsize=(20, 20))
    
    fig.subplots_adjust(hspace=0.5)
    spe = np.array(spe)
    t2 = np.array(t2)
    lim = np.ones_like(spe)
    spe_lim = spe_lim*(lim)
    t2_lim = t2_lim*(lim)
    
    for i in range(20):
        ax.ravel()[i].plot(np.arange(len(spe[i])), spe[i])
        ax.ravel()[i].plot(np.arange(len(spe[i])),spe_lim[i],ls='--',label = 'spe_threshold',color='red')
        ax.ravel()[i].set_title(f'IDV({i+1})')
        ax.ravel()[i].set_ylabel('SPE')
        ax.ravel()[i].legend()


    fig1, ax1 = plt.subplots(5, 4,figsize=(20, 20))
    
    fig1.subplots_adjust(hspace=0.5)
    
    for i in range(20):
        ax1.ravel()[i].plot(np.arange(len(t2[i])), t2[i])
        ax1.ravel()[i].plot(np.arange(len(t2[i])),t2_lim[i],ls='--',label = 't2_threshold',color='red')
        ax1.ravel()[i].set_title(f'IDV({i+1})')
        ax1.ravel()[i].set_ylabel('T2')
        ax1.ravel()[i].legend()

#画单个故障的控制图 fault.shape(20,960)
def plot_fault(fault,lim,fault_num,name,title):
    x = np.array(fault)
    plt.figure(figsize=(10,6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(np.arange(x.shape[1]),x[fault_num,:],color='blue')
    plt.axhline(lim,ls='--',label = 'threshold',color='red')
    plt.xlabel('samples',fontsize=18)
    plt.ylabel(name,fontsize=15)
    plt.legend(loc='upper right',fontsize=12)
    plt.title(title,fontsize=18)
    plt.show()



