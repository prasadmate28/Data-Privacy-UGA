import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#global variables
data = []
P_data={} #probability distribution of original data
S_data={} #probability distribution of synthetic data

#helper functions

def build_histogram():

    #compute distribution P and return
    counts = data.TL_BR.value_counts().values
    bins = data.TL_BR.value_counts().index._data
    total = sum(counts)
    distribution = dict(zip(list(bins),list(counts)))
    # ------------------------------------------------
    plot_histogram(distribution,num=1,title='Histogram of Original DataSet')
    #--------------------------------------------------
    for key,value in distribution.items():
        distribution[key]=float(value/total)
    return distribution


def build_noisy_histogram(sensitivity=1,epsilon=0.1):
    list_of_counts = data.TL_BR.value_counts().values
    bins = data.TL_BR.value_counts().index._data
    for i in range(len(list_of_counts)):
        list_of_counts[i] = list_of_counts[i] + _laplaceNoise_(sensitivity,epsilon)
        if list_of_counts[i] < 0:
            list_of_counts[i] = 0

    bin_count_map = dict(zip(bins, list_of_counts))
    #plot_histogram(bin_count_map)

    return bin_count_map

def plot_histogram(bin_to_count_map,num = 1,title = 'Histogram'):
    print("################ Plotting ", title ," #########################")
    bins = list(bin_to_count_map.keys())
    dummyX = np.arange(1, len(bins) + 1, 1)
    freq = list(bin_to_count_map.values())
    plt.figure(num = num,facecolor='gray')
    plt.bar(dummyX, freq)
    plt.xticks(dummyX, bins, rotation='horizontal')
    plt.xlabel(" BINS (TL + BR) ")
    plt.ylabel(" Frequency ")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_lineGraph(epsilons,distance):
    print('PLOTTING DISTANCE VS PRIVACY BUDGET')
    plt.figure(num=3,facecolor='green')
    plt.plot(epsilons,distance)
    plt.xlabel(" Privacy Budget ")
    plt.ylabel(" DTV ")
    plt.title(r'$\mathrm{Histogram\ of\ Distribution Variance  }$')
    plt.grid(True)
    plt.show()
    pass

def build_normalized_histogram(bins_to_count_map):
    totalFrq = sum(bins_to_count_map.values())
    for key,value in bins_to_count_map.items():
        bins_to_count_map[key] = value/totalFrq;

    plot_histogram(bins_to_count_map,num=3,title='Histogram of Normalized Sanitized Data')
    return bins_to_count_map

def _laplaceNoise_(sensitivity,epsilon,return_size=1):
    if(epsilon > 0):
        y = sensitivity / epsilon
    else:
        y = 10

    noise = np.random.laplace(loc=0, scale=y, size = return_size )
    return noise


def build_sanitized_database(bin_map_count_dist):
    bin_index = list(bin_map_count_dist.keys());
    sanitized_bin_count = (np.random.multinomial(958, list(bin_map_count_dist.values()), size=1)).tolist()
    sanitized_bin_count_map = dict(zip(bin_index,sanitized_bin_count[0]))
    return sanitized_bin_count_map

def computeSanitizedDistribution(epsilon = 0.1):
    ep = epsilon
    bin_map_count_dist = build_noisy_histogram(sensitivity=1, epsilon=ep)  # returns noisy bin counts
    total = sum(bin_map_count_dist.values())

    for key, value in bin_map_count_dist.items():
        # calculate probability disturibution for each bin
        S_data[key] = (value / total)

    return S_data

def computeTotalVariationDistance():
    global P_data
    distance =[]
    epsilons = np.arange(0.1,1.1,0.1)
#do not generate new noise each time
    for ep in epsilons:
        for it in range(40):
            S = 0
            s_avg = 0
            S_data = computeSanitizedDistribution(epsilon=ep)
            for k in P_data.keys():
                S = S + abs(P_data[k] - S_data[k])
            S = 0.5*S
            s_avg = s_avg + S
        distance.append(float(s_avg/40))
    plot_lineGraph(epsilons,distance)



def readFileData(file):
    data = pd.read_csv(file)
    return data;

#main functions

"""
    The program control begins here.
"""
def runExp2():
   global data,P_data
   S_data={}

   print('----------------------  Reading TIC TAC TOE DATA ---------------')
   data = readFileData('dataset/tic-tac-toe.txt');

#--------------PREPARE DATA-------------------------
#Make additional column TL_BR by concatenation TL and BR column values
   newCol = []
   for i in range(len(data)):
       newCol.append(data.TL.values[i] + data.BR.values[i])
   se = pd.Series(newCol)
   data['TL_BR'] = se.values

   data.to_csv('dataset/temp.txt')
#----------------------------------------------------------

   # plot histogram of TL and BR squares of tic tac toe game
   P_data = build_histogram() #returns distribution of original data say 'P"

   bin_map_count_dist = build_noisy_histogram(sensitivity=1,epsilon=0.1) #returns noisy bin counts
    #------------Normalizing bin count------------------
   total = sum(bin_map_count_dist.values())

   for key,value in bin_map_count_dist.items():
       #calculate probability disturibution for each bin
        S_data[key] = (value/total)

   sanitized_bin_count_map = build_sanitized_database(S_data); # builds sanitized database using multinomial sampling from noisy database and returns sanitized bin_counts
   #-----------------------------------------------
   plot_histogram(sanitized_bin_count_map,num=2,title='Histogram of Sanitized Data')
   build_normalized_histogram(sanitized_bin_count_map)

   computeTotalVariationDistance()

#initialize procedure calls
runExp2();

