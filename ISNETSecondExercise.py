import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import matplotlib.image as img
Pi=np.pi








##################################
###########Quantities the user might want to play with
##################################

#I have desing the graphic part of this code to make the plots for the default values. If this values are changed the plotting part of the code might get weird, but the numerical results should be fine and they will be printed into .txt files, so you can go and take a look there if the graphs are nonsense.
########################################################

#Range of momentum transfers (in fm-2) where the data will be equidistantly spaced.
Qmin=0.1
Qmax=2

#Number of data points
NPoints=10

#Error bar size for the measured points
sigmaErr=0.002




##################################
###########Technical definitions and details
##################################


#This is the scale for the bar plots of the MSE, so they all have the same scale and is easier to compare
BarsPlotsScale=0.07

#Scale to divide the errors by when calculating H1. That way the inverse matrix is easier to compute. 
ErrScale=0.005

#Gaussian distribution for plotting gaussians
def gaussian(x, mu, sig):
    return 1.0/np.sqrt(2*Pi*sig**2)*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#Function I created to convert a float to a string in scientific notation with a defined precision. We will use it when writing the results into the .txt files
def ScNotListToString(ListVals,pres):
    ListIn=[]
    for kIn in range(len(ListVals)):
        formin='{:.'+str(pres)+'e}'
        ListIn.append(formin.format(ListVals[kIn]))
    return str(    "["+",".join(map(str,ListIn)) +"]"    )



##################################
###########Data prepatation
##################################


#Error (sigma) as a function of q
def ErrorQ(q):
    return sigmaErr

#Function that takes a list of qs and returns a list of [q, Error(q)]
def ErrorQList(qListE):
    EQLIn=np.array([])
    for iE in range(len(qListE)):
        EQLIn=np.append(EQLIn,ErrorQ(qListE[iE]))
    return EQLIn


#In this file we don't import the data from somewhere else, we use the following truth generator:
def TruthF(q):
    return (1.0+q/17.0068)**(-2)


#Function that takes a list of qs and returns a list of [q, Ftrue(q)]
def DataMaker(qListIn):
    return (np.array((np.array(qListIn) , np.array(TruthF(qListIn))   )).T)


#Datapoints of the true function to be used for plotting
ExpChFF = DataMaker(np.linspace(0, 1.2*Qmax, num=101))

#This is the true value of the radius. If you want to change this value you need to make sure to modify the true function (TruthF) accordingly.
TruthRadius=0.84


#Here are the points where we are "measuring". I am distribuiting them equidistantly between Qmin and Qmax, but you can create your own list. Also, the errors are the same for all but that can be eddited too!
qlocations=np.linspace(Qmin,Qmax,NPoints)
data=DataMaker(qlocations)
errors=ErrorQList(qlocations)


##################################
###########Model Loop
##################################

#This list will keep track of all the results so we can print them to a file at the end
ResultsAllModels=[]


#How many models are we testing. I wrote everything for three, things in the plotting area might get weird if you change this number but the printing of results should be fine. Be sure to modify the first part of the loop to include or remove the models if you modify this number.
NumbOfModels=3

#Since we are making a plot that compares the different models, we are going to create a for loop that runs over them. Here we define the figure that we will be building in the process
fig=plt.figure(num=None, figsize=(8*NumbOfModels, 10), dpi=80, facecolor='w', edgecolor='k')

#We import here the relevant information for each model as the loop progresses
for ijk in range(NumbOfModels):
    if ijk==0:
        from Models.LineModel import *

    if ijk==1:
        from Models.ParabModel import *
    
    if ijk==2:
        from Models.CubicModel import *
    
    #This variable is meant to aid in the location of the subplots on the plot grid latter
    plotcount=ijk+1



    ##################################
    ###########Fitting Stuff
    ##################################


    
    #Main function that fits the parameters
    def ParamFitter(Data,Errors,P0):
        popt, pcov =curve_fit(FNoList, Data.T[0],Data.T[1],p0=P0,sigma=Errors)
        return popt
    
    #Function used to fit the Optimal parameters. In this case, some of the parameters are fixed so the bias in the radius estimation is as small as possible. 
    def OptParamFitter(Data,Errors,P0):
        popt, pcov =curve_fit(FOpt, Data.T[0],Data.T[1],p0=P0,sigma=Errors)
        return popt

    #Function that calculates the etas (difference between the model with the Optimal or Central parameters and the actual data)
    def eta(params,QYDataTruths):
        etasin=np.array([])
        for jeta in range(len(QYDataTruths)):
            etasin=np.append(etasin,QYDataTruths[jeta][1]-F(QYDataTruths[jeta][0],params))
        return etasin

    #Function that retunrs the model radius
    def MR(FParams):
        return ModelRadius(FParams)


    #Here we create the optimal parameters for the model. If the model has only two parameters (like the line) requiring that the radius is the correct one already fixes them, so the OptParams will be read from the file directly. If the model has more parameters, the Optimal set will depend on the data we are using, which is why we need to fit the rest of the parameters in the list.
    if Pz>2:
        LastOptParams=OptParamFitter(data,errors,OptParS0)
        for ij in range(len(LastOptParams)):
            FirstOptParam.append(LastOptParams[ij])
        OptParams=FirstOptParam
    else:
        OptParams=OptParamsFile

    
    #This is the Irreducible Error, the bias obtained with the optimal parameters. In our case it should be zero given how we choose the OptParams. When IrrErrorRad is zero the bias is totally dominated by the TF times etas.
    IrrErrorRad=MR(OptParams)-TruthRadius






    
    ##################################
    ###########Transfer Function Stuff
    ##################################

    
    #This is the inverse Hessian Matrix for chi2. (By the way, I define my \chi2 with a 1/2 extra, so \chi2 is everything that sits in the exponential. This makes the Hessian matrix to lack its original extra 1/2. All the calculations are the same, is just a convention)
    def H1(qDataH1,errListH1,qGradlistH1,paramlistH1):
                
        NyIn=len(qDataH1)
        
        H1In=np.zeros((Pz,Pz))
        
        qPredH1=np.array((np.array(qDataH1[:,0]),FList(qDataH1[:,0],paramlistH1))).T
        
        #We divide the errors by the Error Scale so the matrix we are going to invert has their entries more managables for inversion. At the end we multiply by that same scale so everything stays the same.
        errListIn=np.array(errListH1)*1.0/ErrScale
        
        for iH1 in range(Pz):
            for kH1 in range(iH1,Pz):
                for jH1 in range(NyIn):
                    
                    H1In[iH1][kH1]=H1In[iH1][kH1]+((qGradlistH1[jH1][iH1]*qGradlistH1[jH1][kH1]) -(qDataH1[jH1][1]-qPredH1[jH1][1])*(HessF(qDataH1[jH1][0],paramlistH1, iH1,kH1)))*1.0/(errListIn[jH1]**2)
                #H is symmetric, so no need to re calculate the other half:    
                H1In[kH1][iH1]= H1In[iH1][kH1]
                
        return np.linalg.inv(H1In)*(ErrScale**2)



    #Function that creates the transfer functions of the parameters. The convention is that the first index is the parameter and the second the observation. For example, ParamTranFunc[1][3] describes how the second parameter is affected by changes in the fourth observation (remember that for python the lists starts at 0). 
    def ParamTranFunc(qDataPTF,errListPTF,paramlistPTF):
        gradFPTFin=GradFList(qDataPTF[:,0],paramlistPTF)
        matH1PTFin=H1(qDataPTF,errListPTF,gradFPTFin,paramlistPTF)
        
        for iPTF in range(len(gradFPTFin)):
            gradFPTFin[iPTF,:]*=(errListPTF[iPTF]**(-2))

        return np.matmul(matH1PTFin,(gradFPTFin.T))



    #Function that creates the transfer functions of the Radius. It has only one index and it describes how that observation affects the estimated radius.
    #(If you are reading this comment you get a free cookie with your coffee or a pretzel with your beer when you cash in your coupon. It means a lot to me that you are taking such an effort to understand this code and the TFs ideas) 
    def MRTranFunc(FParams,ParamTFIn):
        gradRIn=np.array([])
        for iM in range(Pz):
            gradRIn=np.append(gradRIn,GradRadius(FParams,iM))
        return np.matmul(gradRIn,ParamTFIn)






    ##################################
    ###########Results Calculation
    ##################################

    #Function that retunrs the bias, variance, mse, and other important stuff like the TF values times the respective etas and sigmas, as well as the parameters obtained. All of this is computed after specifying the data we are "measuring" and the associated errors
    def MSE(qDataM,ErrorsM):
        
        
        #Bias calculation:
        
        etasM=eta(OptParams,qDataM)
        qDataOptFunc=(np.array((np.array(qDataM.T[0]) , np.array(FList(qDataM.T[0],OptParams))   )).T)
        #Recall that the TFs used to calculate the bias need to be evaluated at the Optimal parameters and the data generated by the optimal function and NOT on the actual parameters and data.
        TFuncsM=ParamTranFunc(qDataOptFunc,ErrorsM,OptParams)
        newparamsM=OptParams+np.dot(TFuncsM,etasM)
        RadTF=MRTranFunc(OptParams,TFuncsM)

        TFEtasListM=[]
        
        biasInTot=IrrErrorRad
                
        for iM in range(len(ErrorsM)):
            biasInTot=biasInTot+RadTF[iM]*etasM[iM]
            TFEtasListM.append(RadTF[iM]*etasM[iM])



        #Variance calculation:
        
        newparams=OptParams+np.dot(TFuncsM,etasM)
        #For the variance the TFs need to be evaluated on the actual parameters obtained by the actual data, and the actual data as well.
        TFuncsM=ParamTranFunc(qDataOptFunc,ErrorsM,newparams)
        RadTF=MRTranFunc(newparams,TFuncsM)
        VarInTot=0
        TFSigListM=[]
        for jM in range(len(ErrorsM)):
            VarInTot=VarInTot+(RadTF[jM]*ErrorsM[jM])**2
            TFSigListM.append(abs(RadTF[jM]*ErrorsM[jM]))
        
        
        
        MSEMlist=np.array([abs(biasInTot),np.sqrt(VarInTot),np.sqrt(biasInTot**2+VarInTot)])
        return [MSEMlist,[TFEtasListM,TFSigListM],newparamsM]


    #Results for the given model, data, and errors:
    CalcResults=MSE(data,errors)
    
    #This is a global list that will keep the results of each model as the Loop progresses
    ResultsAllModels.append([ModelName[0],CalcResults])
    
    MSEResults=CalcResults[0]
    TFEtasList=CalcResults[1][0]
    TFSigList=CalcResults[1][1]
    newparams=CalcResults[2]




    ##################################
    ###########Plotting Stuff
    ##################################

    #Function that makes the bar plots that show the bias, variance, and MSE.
    def ResBarPlotterComparison(ResultsIN,FIGNAME):
        
        labelsscores=['Bias','Variance','MSE']
        x = np.arange(3)
        width = 1.2/(3)
        plt.gca().set_prop_cycle(None)
        FIGNAME.bar(x -1/2*width, ResultsIN, width,  color=['blue','orange','m'],edgecolor='k')
        FIGNAME.set_xticks(x)
        FIGNAME.tick_params(axis='both', which='major', labelsize=15)
        FIGNAME.set_xticklabels(labelsscores,fontsize=20)
        FIGNAME.set_ylim([0, BarsPlotsScale])



    #Here we start plotting. We will have two graphs for each model (ax1 is about the plot of the data, function fit, and TFs values. ax2 is about the bar plot with results for the bias, variance and MSE)
    ax1=fig.add_subplot(str(2) + str(NumbOfModels)+str(plotcount))
    ax1.plot(ExpChFF.T[0],ExpChFF.T[1],'k-',linewidth=4)
    ax1.plot(np.linspace(0, 1.2*Qmax, num=101),FList(np.linspace(0, 1.2*Qmax, num=101),newparams),color='orange',linewidth=4)

    #We are plotting the errors as 5*Sigma so they can be seen.
    plt.errorbar(data.T[0],data.T[1], yerr=errors*5,marker='o',color='red',markersize=10,linestyle='none')
    ax1.tick_params(axis='both', which='major', labelsize=15)
    
    #Projected radius extraction and its associated variance
    ax1.text(0.55,0.8, str(round(MR(newparams),3))+ '$\pm$' +str(round(MSEResults[1],3)) +'fm' , style='italic', fontsize=20,transform=ax1.transAxes,
            bbox={'facecolor': 'white', 'pad': 10})
    ax1.text(0.55,0.9,'Projected Radius' , style='italic', fontsize=18,transform=ax1.transAxes)


    #Radius transfer function values times their respective error added to the plot. eps is an "epsilon" to displace the printing of the TFs values away from the data centers so they don't graphically overlap
    eps=0.01
    for ij in range(len(TFSigList)):
        ax1.text(data[ij][0]+eps,data[ij][1]+eps, abs(round(TFSigList[ij],3)), style='italic', fontsize=14,
            bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 2})
        
    #Radius transfer function values times their respective etas added to the plot
    for ij in range(len(TFEtasList)):
        ax1.text(data[ij][0]-15*eps,data[ij][1]-2*eps, round(TFEtasList[ij],3), style='italic', fontsize=14,
            bbox={'facecolor': 'blue', 'alpha': 0.3, 'pad': 2})


    #Labeling the axes 
    ax1.set_xlabel('$Q^2 \  [fm^{-2}]$',fontsize=20)
    ax1.set_ylabel('$ G(Q^2) $',fontsize=20)
    ax1.set_title(ModelName[0],fontsize=25)


    #Now we go for ax2 which is the second plot that shows the bias, variance and MSE results
    plotcount=plotcount+NumbOfModels

    ax2=fig.add_subplot(str(2) + str(NumbOfModels)+str(plotcount))

    ax2.set_ylabel('Score [fm]',fontsize=20)

    ResBarPlotterComparison(MSEResults,ax2)


    





#We finished with making the plot, now we show it and we ask the question about which model should we use
print("\n\n\nWhich model should we use to extract the radius? After closing the graph type 1, 2, or 3 for choosing the Line, Parabola, or Cubic respectively\n\n\n")

plt.tight_layout()

#We also save the figure so we can take a look at it latter
plt.savefig('FIG1-ModelComparison.png', dpi=fig.dpi)

plt.show()






##################################
###########Calculating results for the selected model
##################################


ModNumber=input()

#Now we start to make the results with the choosen model
if ModNumber==1:
    from Models.LineModel import *

if ModNumber==2:
    from Models.ParabModel import *

if ModNumber==3:
    from Models.CubicModel import *


#Just in case we modified it before we are creating the base data again
qlocations=np.linspace(Qmin,Qmax,NPoints)
data=DataMaker(qlocations)
errors=ErrorQList(qlocations)

#We are creating a single realization of the data. Each point will be added a fluctuation \epsilon with mean 0 and standard deviation sigma (the associated error)
NoiseRealization=np.array([])
for ij in range(len(errors)):
    NoiseRealization=np.append(NoiseRealization,np.random.normal(0, errors[ij], 1))

#Now adding the fluctuation
for ij in range(len(data)):
    data[ij][1]=data[ij][1]+NoiseRealization[ij]


#Stuff we will need to make the results (parameters, transfer functions, extracted radius, etc. Everything is evaluated on the new fluctuated data to mimic a real experiment)
newparams= ParamFitter(data,errors,ParS0)
TFParamsValues=ParamTranFunc(data,errors,newparams)
TFRad=MRTranFunc(newparams,TFParamsValues)
ExtRad=MR(newparams)


#Lets calculate the variance on the radius for this particular fluctuated data
RadErr=0
for ij in range(len(TFRad)):
    RadErr=RadErr+(TFRad[ij]*errors[ij])**2

RadErr=np.sqrt(RadErr)



#Plotting stuff

#This is the function that plots a nice gaussian to show the estimated radius. We also have two vertical lines showing the true radius of 0.84 fm and a radius of 0.88 to mimic the proton puzzle situation
def GaussPlotter(FIGNAME,muG,sigG,trueval,colorvalue):
    x_values = np.linspace(min(muG-4*sigG,trueval-4*sigG), max(muG+4*sigG,trueval+4*sigG), 120)
    ys=gaussian(x_values, muG, sigG)
    FIGNAME.plot(x_values, gaussian(x_values, muG, sigG),linestyle='-',linewidth=4,color=colorvalue)
    FIGNAME.fill_between(x_values, gaussian(x_values, muG, sigG),alpha=0.3)
    FIGNAME.tick_params(axis='both', which='major', labelsize=20)
    plt.axvline(trueval,0.05,0.95,color='tab:pink',linewidth=6)
    plt.axvline(0.88,0.05,0.95,color='tab:purple',linewidth=6)



#We start plotting
fig=plt.figure(num=None, figsize=(8, 10), dpi=80, facecolor='w', edgecolor='k')
ax1=fig.add_subplot(211)

#True function
ax1.plot(ExpChFF.T[0],ExpChFF.T[1],'k-',linewidth=4)

#Our estimated function
ax1.plot(np.linspace(0, 1.2*Qmax, num=101),FList(np.linspace(0, 1.2*Qmax, num=101),newparams),color='orange',linewidth=4)

#Fluctuated data. We are plotting the errors as 5*Sigma so they can be seen.
plt.errorbar(data.T[0],data.T[1], yerr=errors*5,marker='o',color='red',markersize=10,linestyle='none')

ax1.tick_params(axis='both', which='major', labelsize=15)

#We write the extracted radius and error on the plot
ax1.text(0.55,0.8, str(round(ExtRad,3))+ '$\pm$' +str(round(RadErr,3)) +'fm' , style='italic', fontsize=20,transform=ax1.transAxes,
        bbox={'facecolor': 'white', 'pad': 10})

ax1.text(0.55,0.9,'Extracted Radius' , style='italic', fontsize=18,transform=ax1.transAxes)


#Radius transfer function values times their respective error added to the plot. eps is an "epsilon" to displace the printing of the TFs values away from the data centers so they don't graphically overlap    
eps=0.01
for ij in range(len(TFSigList)):
    ax1.text(data[ij][0]+eps,data[ij][1]+eps, abs(round(TFSigList[ij],3)), style='italic', fontsize=14,
        bbox={'facecolor': 'orange', 'alpha': 0.3, 'pad': 2})
#Radius transfer function values times their respective etas added to the plot
for ij in range(len(TFEtasList)):
    ax1.text(data[ij][0]-15*eps,data[ij][1]-2*eps, round(TFEtasList[ij],3), style='italic', fontsize=14,
        bbox={'facecolor': 'blue', 'alpha': 0.3, 'pad': 2})



#Labeling the axes
ax1.set_xlabel('$Q^2 \  [fm^{-2}]$',fontsize=20)
ax1.set_ylabel('$ G(Q^2) $',fontsize=20)
ax1.set_title(ModelName[0],fontsize=25)


#Lower plot that shows the gaussian
ax2=fig.add_subplot(212)

ax2.set_ylabel('Score [fm]',fontsize=20)

#Legend making
pink_patch = patches.Patch(color='tab:pink', label='0.84 fm' )
purple_patch = patches.Patch(color='tab:purple', label='0.88 fm')
blue_patch = patches.Patch(color='tab:blue', label='Ext. Radius')
ax2.legend(loc="upper left",handles=[pink_patch,purple_patch,blue_patch],fontsize=15)

#Plotting the gaussian results
GaussPlotter(ax2,ExtRad,RadErr,TruthRadius,'blue')

#The range of values for the gaussian plot. I want it to be fixed so the gaussian does not overlaps with the legend
ax2.set_xlim([0.76, 0.90])


#We show and save the graph
plt.tight_layout()
plt.savefig('FIG2-SelectedModelResults.png', dpi=fig.dpi)
plt.show()



##################################
###########Writing results to two .txt files
##################################

FILE=open("SummaryResultsModels.txt",'w')
FILE.write("Score (Bias, Variance, MSE) for the Radius for each model \n_________________________\n")
for ij in range(NumbOfModels):
    
    FILE.write("\n" +ResultsAllModels[ij][0] + "="+ ScNotListToString(ResultsAllModels[ij][1][0],4)+"\n")

    FILE2=open("DetailedResultsModel"+ResultsAllModels[ij][0]+".txt",'w')
    FILE2.write("Model: "+ResultsAllModels[ij][0] +"\n")
    FILE2.write("\n Data\n")
    FILE2.write('%s' %data)

    FILE2.write("\n \n Errors\n")
    FILE2.write('%s' %errors)
    FILE2.write("\n \n Transfer functions times sigmas (Variance)\n")
    #FILE2.write('%s' %ResultsAllModels[ij][1][1][0])
    FILE2.write(ScNotListToString(ResultsAllModels[ij][1][1][0],3))
    FILE2.write("\n \n Transfer functions times etas (Bias)\n")
    #FILE2.write('%s' %ResultsAllModels[ij][1][1][1])
    FILE2.write(ScNotListToString(ResultsAllModels[ij][1][1][1],3))
    FILE2.close()  




FILE.close()  





##################################
###########Coupon
##################################


#Finally, if the estimated radius for this realization of the data is such that we can rule out the wrong radius of 0.88 fm for a two sigma level, then we print the following coupon to celebrate that we have "solved" the proton puzzle
if abs(ExtRad-TruthRadius) < 2*RadErr and abs(ExtRad-0.88) > 2*RadErr:
    
    print("If you are using a web-browser terminal and you are reading this you should open the image called Coupon.png")
    #Code to print the coupon
    fig=plt.figure(figsize=(20, 20), dpi=200)

    im = img.imread('Coupon.png')

    plt.axis('off')
    plt.imshow(im) 
    plt.show()







