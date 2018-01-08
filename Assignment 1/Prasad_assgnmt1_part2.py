import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


#global variables
file_data = pd.read_csv(r"UCI Adut data/adultData.csv")

condition = True

#helper functions
def _laplaceNoise_(sensitivity,epsilon,return_size=1):

    if(epsilon > 0):
        y = sensitivity / epsilon
    else:
        y = 10

    noise = np.random.laplace(loc=0, scale=y, size = return_size )
    return noise

def buildNoisyHistogram(noisy_data):

        print("################ Plotting noisy histogram #########################")
        x = list(noisy_data.keys())
        dummyX = np.arange (1, len(x) + 1, 1)
        y = list(noisy_data.values())
        plt.bar(dummyX,y)
        plt.xticks(dummyX,x,rotation='vertical')
        plt.xlabel(" Native Country ")
        plt.ylabel(" Frequency ")
        plt.title(r'$\mathrm{Noisy Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

def buildHistogram(D,idx):

    if idx == 1:

        plt.hist(file_data['age'],bins=10,color='Green')
        plt.xlabel("AGE")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()


    if idx == 2:
        plt.hist(file_data['fnlwgt'], bins=10, color='Green')
        plt.xlabel("fnlwgt")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()


    if idx == 3:
        plt.hist(file_data['education-num'], bins=10, color='Green')
        plt.xlabel("education-num")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set}$')
        plt.grid(True)
        plt.show()


    if idx == 4:
        plt.hist(file_data['capital-gain'], bins=10, color='Green')
        plt.xlabel("capital-gain")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()


    if idx == 5:
        plt.hist(file_data['capital-loss'], bins=10, color='Green')
        plt.xlabel("capital-loss")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()


    if idx == 6:
        plt.hist(file_data['Dummy'], bins=10, color='Green')
        plt.xlabel("Dummy")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

    #-------------categorial attributes----------------------------------------
    if idx == 7:
        file_data.sex.value_counts().plot(kind='bar')
        plt.xlabel("Sex")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

    if idx == 8:
        file_data.workclass.value_counts().plot(kind='bar')
        plt.xlabel("workclass")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

    if idx == 9:
        file_data.education.value_counts().plot(kind='bar')
        plt.xlabel("education")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

    if idx == 10:
        file_data.maritalstatus.value_counts().plot(kind='bar')
        plt.xlabel("marital-status")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

    if idx == 11:
        file_data.occupation.value_counts().plot(kind='bar')
        plt.xlabel("occupation")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

    if idx == 12:
        file_data.relationship.value_counts().plot(kind='bar')
        plt.xlabel("relationship")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

    if idx == 13:
        file_data.race.value_counts().plot(kind='bar')
        plt.xlabel("race")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

    if idx == 14:
        file_data.nativecountry.value_counts().plot(kind='bar')
        plt.xlabel("native-country")
        plt.ylabel("Frequency")
        plt.title(r'$\mathrm{Histogram\ of\ Adult Data Set:}$')
        plt.grid(True)
        plt.show()

def evaluateUtility(attributecount_dict):


    scores = []
    totalCount = 0
    for key in attributecount_dict.keys():
        totalCount += attributecount_dict[key]
    for key in attributecount_dict.keys():
        perc = round(((attributecount_dict[key] / totalCount) * 100), 5)
        if perc > 75.0:
            scores.append(attributecount_dict[key] - round((attributecount_dict[key] * 0.9), 5))
        else:
            scores.append((attributecount_dict[key] - round((attributecount_dict[key] * 0.20), 5)))
        #scores.append(attributecount_dict[key] - round(((attributecount_dict[key] /totalCount) * 500),5 ))

    print("Scoress____________________________")
    print(scores)
    return scores

def exponentialNoise(attributecount_dict,epsilon =0.1):
    exponential_noise = []
    scores = evaluateUtility(attributecount_dict)

    for score in scores:
        exponential_noise.append(round(math.exp((epsilon*score)/2),5))

    return exponential_noise


def normalizedExpNoise(exponential_noise):
    normalized_exp_noise = []
    total = 0.0
    for exp in exponential_noise:
        total += exp
    for exp in exponential_noise:
         normalized_exp_noise.append(((exp/total)))

    return normalized_exp_noise

# call function build histogram

#call problem stubs
def problem3 ():
    print("############# Solving problem no.  3  ######################")
    print("1.age \n    2.fnlwgt  \n     3.education-num   \n  4.capital-gain  \n    5.capital-loss  \n    6.Dummy  \n    7.Sex \n    8.workclass \n    9.education \n    10.marital-status \n    11.occupation \n    12.relationship \n 13.race \n    14.native-country \n      ")
    index = int(input("Enter the index of attribute" ))
    #Enter the number corresponding to the attribute as second argument to buildhistogram function
    if(index >=1 & index < 15):
        buildHistogram(file_data,index)
    else :
        print(" %%%%%%%%%% INVALID INPUT OF ATTRIBUTE INDEX         EXITING THE CODE    %%%%%%%%%%%%%%%%")
        global condition
        condition = False


def problem4 ():
    print("############# Solving problem no.  4  ######################")
    native_country_values = file_data.nativecountry

    print(":::::::::::::: The sensitivity for count queries is 1. Hence for this histogram on native country attribute = 1 ::::::::::::")
    mean_sq_err = []

    print("::::::::::::::::::  Releasing noisy histogram with epsilon = 0.1  ::::::::::::::::::::::")
    native_country_values_dict = native_country_values.value_counts().to_dict()

    # Converts values in dict to int
    for key, value in native_country_values_dict.items():
            native_country_values_dict[key] = int(value)

    #adds laplace noise to count values
    for k in native_country_values_dict.keys():
        nois = int(_laplaceNoise_(1.0, 0.1))
        native_country_values_dict[k] += nois

    #Build noisy histogram with epsilon = 0.1
    buildNoisyHistogram(native_country_values_dict)

    e = np.arange (0.1 , 1.1 , 0.1)
    for ep in e:
        # adds laplace noise to count values
        for k in native_country_values_dict.keys():
            mse=0
            for i in range(50):  # randomly generate noise 50 times
                noise = int(_laplaceNoise_(sensitivity=1.0, epsilon= ep))
                native_country_values_dict[k] += noise
                mse += math.pow(noise,2) #(f'(x) - f(x))^2 = noise^2

        mean_sq_err.append(mse/50)#mean sq error on 50 run for  a particular epsilon value

    plt.plot(e, mean_sq_err)
    plt.xlabel(" Privacy Budget ")
    plt.ylabel(" MSE ")
    plt.title(r'$\mathrm{Histogram\ of\    Adult Data Set   :   Mean Square Error }$')
    plt.grid(True)
    plt.show()



def problem5 ():
    print("################### Solving problem no.  5  ######################")

    native_country_values_dict = file_data.nativecountry.value_counts().to_dict()
    # Converts values in dict to int
    for key, value in native_country_values_dict.items():
        native_country_values_dict[key] = int(value)

    exponential_noise = exponentialNoise(native_country_values_dict)
    #print(exponential_noise)

    normalized_exp_noise = normalizedExpNoise(exponential_noise)
    print("normalized noise ____________________________")
    print(normalized_exp_noise)







def _init_Assignment1():

    global condition

    while(condition):
        inpt = int(input("Enter problem number :: 3/4/5   "))

        if inpt == 3:
            problem3()
        elif inpt == 4:
            problem4()
        elif inpt == 5:
            problem5()
        else:
            condition = False
            print("$$$$$$$$$$$$$$$$$$ INVALID INPUT    EXITING THE PROGRAM  $$$$$$$$$$$$$$$$$$")

_init_Assignment1()
