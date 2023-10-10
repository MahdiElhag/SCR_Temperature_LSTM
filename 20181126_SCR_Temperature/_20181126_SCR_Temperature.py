#################################################################################################

#importing essential libraries
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import time
from time import gmtime, strftime
import subprocess as sp
import os.path

#################################################################################################

#importing keras (tensorflow backended)
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM
from keras.callbacks import CSVLogger
from keras import optimizers
from keras.utils import plot_model

#################################################################################################

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

#################################################################################################

#importing sklern libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#################################################################################################

#training and validation parameters
data_percentage = 0.92 #0.90
cross_valid_prc = 0.3
iterations_numr = 3    #2
moving_avg_degr = 1    #1000
dropout_percent = 0.1  #0.15
lstm_time_stamp = 300  #50

#################################################################################################

#creating a record directory for the models and figues
naming = strftime("%Y_%m_%d_%H_%M_%S", gmtime())

current_dir = os.getcwd()
final_dir = os.path.join(current_dir, naming)
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

#################################################################################################

#preparing a log to get the correlation results
log = open(naming + "/" + naming + ".txt", "w")
log.write(naming + "\n\n")
log.write("data_percentage = " + str(data_percentage) + "\n")
log.write("moving_avg_degr = " + str(moving_avg_degr) + "\n")
log.write("dropout_percent = " + str(dropout_percent) + "\n")
log.write("lstm_time_stamp = " + str(lstm_time_stamp) + "\n")
log.write("iterations_numr = " + str(iterations_numr) + "\n\n")

#################################################################################################

#defining a csv logger for training losses
csv_logger = CSVLogger(naming + "/" + naming + '.csv', append=True, separator=',')

#################################################################################################

#reading training and validation csv file from local
dft = pd.read_csv("SCR_LSTM_ALL_DATA.csv", index_col=False)

#################################################################################################

#sorting data in ascending form
dft.sort_index(ascending = True,inplace = True)

dft['ASMod_dmEGFld__INDEX_0_AVG'] = dft['ASMod_dmEGFld__INDEX_0'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['ActMod_trqClth_AVG'] = dft['ActMod_trqClth'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['CEngDsT_t_AVG'] = dft['CEngDsT_t'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['CoEOM_numOpModeAct_AVG'] = dft['CoEOM_numOpModeAct'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['EnvP_p_AVG'] = dft['EnvP_p'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['EnvT_t_AVG'] = dft['EnvT_t'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['Epm_nEng_AVG'] = dft['Epm_nEng'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['VehV_v_AVG'] = dft['VehV_v'].rolling(window=moving_avg_degr).mean().fillna(0)
dft['T_iSCR1_AVG'] = dft['T_iSCR1'].rolling(window=moving_avg_degr).mean().fillna(0)

#################################################################################################

#normalizing csv data to be used in training and validation
scaler00 = MinMaxScaler(feature_range=(0, 1))
scaler01 = MinMaxScaler(feature_range=(0, 1))
scaler02 = MinMaxScaler(feature_range=(0, 1))
scaler03 = MinMaxScaler(feature_range=(0, 1))
scaler04 = MinMaxScaler(feature_range=(0, 1))
scaler05 = MinMaxScaler(feature_range=(0, 1))
scaler06 = MinMaxScaler(feature_range=(0, 1))
scaler07 = MinMaxScaler(feature_range=(0, 1))
scaler08 = MinMaxScaler(feature_range=(0, 1))

dft['ASMod_dmEGFld__INDEX_0_Norm'] = scaler00.fit_transform(dft['ASMod_dmEGFld__INDEX_0_AVG'].reshape(-1,1))
dft['ActMod_trqClth_Norm'] = scaler01.fit_transform(dft['ActMod_trqClth_AVG'].reshape(-1,1))
dft['CEngDsT_t_Norm'] = scaler02.fit_transform(dft['CEngDsT_t_AVG'].reshape(-1,1))
dft['CoEOM_numOpModeAct_Norm'] = scaler03.fit_transform(dft['CoEOM_numOpModeAct_AVG'].reshape(-1,1))
dft['EnvP_p_Norm'] = scaler04.fit_transform(dft['EnvP_p_AVG'].reshape(-1,1))
dft['EnvT_t_Norm'] = scaler05.fit_transform(dft['EnvT_t_AVG'].reshape(-1,1))
dft['Epm_nEng_Norm'] = scaler06.fit_transform(dft['Epm_nEng_AVG'].reshape(-1,1))
dft['VehV_v_Norm'] = scaler07.fit_transform(dft['VehV_v_AVG'].reshape(-1,1))
dft['T_iSCR1_Norm'] = scaler08.fit_transform(dft['T_iSCR1_AVG'].reshape(-1,1))

#################################################################################################

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

#################################################################################################

#plottting the features and output
plt.plot(dft['Time'], dft['ASMod_dmEGFld__INDEX_0_AVG'], '-r',label = 'Filtered')
plt.plot(dft['Time'], dft['ASMod_dmEGFld__INDEX_0'], '-b',label = 'Unfiltered')
plt.title("ASMod_dmEGFld__INDEX_0")
plt.xlabel("Sample Index")
plt.ylabel("ASMod_dmEGFld__INDEX_0")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'ASMod_dmEGFld__INDEX_0.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the featrues and output
plt.plot(dft['Time'], dft['ActMod_trqClth_AVG'], '-r',label = 'Filtered')
plt.plot(dft['Time'], dft['ActMod_trqClth'], '-b',label = 'UnFiltered')
plt.title("ActMod_trqClth")
plt.xlabel("Sample Index")
plt.ylabel("ActMod_trqClth")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'ActMod_trqClth.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the featrues and output
plt.plot(dft['Time'], dft['CEngDsT_t_AVG'], '-r',label = 'Filtered')
plt.plot(dft['Time'], dft['CEngDsT_t'], '-b',label = 'Unfiltered')
plt.title("CEngDsT_t")
plt.xlabel("Sample Index")
plt.ylabel("CEngDsT_t")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'CEngDsT_t.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the featrues and output
plt.plot(dft['Time'], dft['CoEOM_numOpModeAct_AVG'], '-r',label = 'Filtered')
plt.plot(dft['Time'], dft['CoEOM_numOpModeAct'], '-b',label = 'Unfiltered')
plt.title("CoEOM_numOpModeAct")
plt.xlabel("Sample Index")
plt.ylabel("CoEOM_numOpModeAct")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'CoEOM_numOpModeAct.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the featrues and output
plt.plot(dft['Time'], dft['EnvP_p_AVG'], '-r',label = 'Filtered')
plt.plot(dft['Time'], dft['EnvP_p'], '-b',label = 'Unfiltered')
plt.title("EnvP_p")
plt.xlabel("Sample Index")
plt.ylabel("EnvP_p")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'EnvP_p.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the featrues and output
plt.plot(dft['Time'], dft['EnvT_t_AVG'], '-r',label = 'Filtered')
plt.plot(dft['Time'], dft['EnvT_t'], '-b',label = 'Unfiltered')
plt.title("CEngDsT_t")
plt.xlabel("Sample Index")
plt.ylabel("EnvT_t")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'EnvT_t.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the featrues and output
plt.plot(dft['Time'], dft['Epm_nEng_AVG'], '-r',label = 'Filtered')
plt.plot(dft['Time'], dft['Epm_nEng'], '-b',label = 'Unfiltered')
plt.title("ASMod_dmEGFld__INDEX_0")
plt.xlabel("Sample Index")
plt.ylabel("Epm_nEng")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'Epm_nEng.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the featrues and output
plt.plot(dft['Time'], dft['VehV_v_AVG'], '-r',label = 'Filtered')
plt.plot(dft['Time'], dft['VehV_v'], '-b',label = 'Unfiltered')
plt.title("VehV_vh")
plt.xlabel("Sample Index")
plt.ylabel("VehV_v")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'VehV_v.png')
#plt.show()
plt.clf()

#################################################################################################

#plottting the featrues and output
plt.plot(dft['Time'], dft['T_iSCR1_AVG'], '-r',label = 'Filtered')
plt.plot(dft['Time'], dft['T_iSCR1'], '-b',label = 'Unfiltered')
plt.title("T_iSCR1")
plt.xlabel("Sample Index")
plt.ylabel("T_iSCR1")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'T_iSCR1.png')
#plt.show()
plt.clf()

#################################################################################################


#checking Pearsons Correlation between features and DOC_DS 
log.write("Correlation between T_iSCR1 and ASMod_dmEGFld__INDEX_0" + "\n")
log.write(np.array_str(np.corrcoef(dft.ASMod_dmEGFld__INDEX_0,dft.T_iSCR1)))
log.write("\n\n")

log.write("Correlation between T_iSCR1 and ActMod_trqClth" + "\n")
log.write(np.array_str(np.corrcoef(dft.ActMod_trqClth,dft.T_iSCR1)))
log.write("\n\n")

log.write("Correlation between T_iSCR1 and CEngDsT_t" + "\n")
log.write(np.array_str(np.corrcoef(dft.CEngDsT_t,dft.T_iSCR1)))
log.write("\n\n")

log.write("Correlation between T_iSCR1 and CoEOM_numOpModeAct" + "\n")
log.write(np.array_str(np.corrcoef(dft.CoEOM_numOpModeAct,dft.T_iSCR1)))
log.write("\n\n")

log.write("Correlation between T_iSCR1 and EnvP_p" + "\n")
log.write(np.array_str(np.corrcoef(dft.EnvP_p,dft.T_iSCR1)))
log.write("\n\n")

log.write("Correlation between T_iSCR1 and EnvT_t" + "\n")
log.write(np.array_str(np.corrcoef(dft.EnvT_t,dft.T_iSCR1)))
log.write("\n\n")

log.write("Correlation between T_iSCR1 and Epm_nEng" + "\n")
log.write(np.array_str(np.corrcoef(dft.Epm_nEng,dft.T_iSCR1)))
log.write("\n\n")

log.write("Correlation between T_iSCR1 and VehV_v" + "\n")
log.write(np.array_str(np.corrcoef(dft.VehV_v,dft.T_iSCR1)))
log.write("\n\n")

#################################################################################################

#taking 90% of the data points as train. This number can change.
train_size = int(data_percentage*len(dft))

#segregating the inputs and output on the test and train data
trainX = dft.loc[1:train_size,['ASMod_dmEGFld__INDEX_0_Norm', 'ActMod_trqClth_Norm', 'CEngDsT_t_Norm','CoEOM_numOpModeAct_Norm', 'EnvP_p_Norm', 'EnvT_t_Norm', 'Epm_nEng_Norm','VehV_v_Norm']]
trainY = dft.loc[1:train_size,['T_iSCR1_Norm']]
log.write("trainX = dft.loc[1:train_size,['ASMod_dmEGFld__INDEX_0_Norm', 'ActMod_trqClth_Norm', 'CEngDsT_t_Norm','CoEOM_numOpModeAct_Norm', 'EnvP_p_Norm', 'EnvT_t_Norm', 'Epm_nEng_Norm','VehV_v_Norm']]" + "\n")
log.write("trainY = dft.loc[1:train_size,['T_iSCR1_Norm']]" + "\n\n")

testX1 = dft.loc[1:train_size,['ASMod_dmEGFld__INDEX_0_Norm', 'ActMod_trqClth_Norm', 'CEngDsT_t_Norm','CoEOM_numOpModeAct_Norm', 'EnvP_p_Norm', 'EnvT_t_Norm', 'Epm_nEng_Norm','VehV_v_Norm']]
testY1 = dft.loc[1:train_size,['T_iSCR1_Norm']]
log.write("testX1 = dft.loc[1:train_size,['ASMod_dmEGFld__INDEX_0_Norm', 'ActMod_trqClth_Norm', 'CEngDsT_t_Norm','CoEOM_numOpModeAct_Norm', 'EnvP_p_Norm', 'EnvT_t_Norm', 'Epm_nEng_Norm','VehV_v_Norm']]" + "\n")
log.write("testY1 = dft.loc[1:train_size,['T_iSCR1_Norm']]" + "\n")

testX2 = dft.loc[train_size:len(dft),['ASMod_dmEGFld__INDEX_0_Norm', 'ActMod_trqClth_Norm', 'CEngDsT_t_Norm','CoEOM_numOpModeAct_Norm', 'EnvP_p_Norm', 'EnvT_t_Norm', 'Epm_nEng_Norm','VehV_v_Norm']]
testY2 = dft.loc[train_size:len(dft),['T_iSCR1_Norm']]
log.write("testX2 = dft.loc[train_size:len(dft),['ASMod_dmEGFld__INDEX_0_Norm', 'ActMod_trqClth_Norm', 'CEngDsT_t_Norm','CoEOM_numOpModeAct_Norm', 'EnvP_p_Norm', 'EnvT_t_Norm', 'Epm_nEng_Norm','VehV_v_Norm']]" + "\n")
log.write("testY2 = dft.loc[train_size:len(dft),['T_iSCR1_Norm']]" + "\n\n")

#################################################################################################

#The inputs needed to be reshaped in the format of  a 3d Tensor with dimesnions = [batchsize,timesteps,features]
trainX = np.reshape(np.array(trainX),(trainX.shape[0],1,trainX.shape[1]))
testX1 = np.reshape(np.array(testX1),(testX1.shape[0],1,testX1.shape[1]))
testX2 = np.reshape(np.array(testX2),(testX2.shape[0],1,testX2.shape[1]))

#################################################################################################

#structuring the LSTM RNN network
model = Sequential()
model.add(LSTM(lstm_time_stamp ,batch_input_shape=(1,trainX.shape[1],trainX.shape[2]),return_sequences = True))
model.add(Dropout(dropout_percent))
model.add(LSTM(lstm_time_stamp))
model.add(Dense(1))
#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam)
epochs = iterations_numr
start = time.time()
m = model.fit(trainX, np.array(trainY), epochs = epochs, batch_size=1, verbose=2,validation_split=cross_valid_prc, callbacks=[csv_logger])
print ("Compilation Time : ", time.time() - start)
log.write("Training Time : " + str(time.time() - start))

#closing the log file
log.close()

#################################################################################################

#save model in JSON format
model.save(naming + "/" + naming + ".h5")
print("Model is saved model to disk")

#################################################################################################

# summarizing model for loss
plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss-Mean Squared Error')
plt.xlabel('epoch')
plt.legend(['Train', 'Valid'], loc='best')
plt.savefig(naming + '/' + 'Model_Loss.png')
#plt.show()
plt.clf()

#################################################################################################

#comparing actual and predicted DOC_DS in training data by plotting
testPredict = model.predict(testX1,batch_size = 1)
plt.plot(scaler00.inverse_transform(testPredict.reshape(-1,1)),'-r',label = 'Model T_iSCR1')
plt.plot(scaler00.inverse_transform(testY1.T_iSCR1_Norm.reshape(-1,1)),'-b',label = 'Actual T_iSCR1_')    # Reshaped

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

plt.title("Actual & Predicted SCR Temperature (Training)")
plt.xlabel("Sample Index")
plt.ylabel("Temperature [C]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'T_iSCR1_Training_Output.png')
#plt.show()
plt.clf()

prediction = pd.DataFrame(testPredict, columns=['SCR_T_PRD_NORM_TRAIN']).to_csv(naming + "/" +'SCR_T_PRD_NORM_TRAIN.csv')
prediction = pd.DataFrame(testY1.T_iSCR1_Norm.reshape(-1,1), columns=['SCR_T_ORG_NORM_TRAIN']).to_csv(naming + "/" +'SCR_T_ORG_NORM_TRAIN.csv')

prediction = pd.DataFrame(scaler00.inverse_transform(testPredict.reshape(-1,1)), columns=['SCR_T_PRD_TRAIN']).to_csv(naming + "/" +'SCR_T_PRD_TRAIN.csv')
prediction = pd.DataFrame(scaler00.inverse_transform(testY1.T_iSCR1_Norm.reshape(-1,1)), columns=['SCR_T_ORG_TRAIN']).to_csv(naming + "/" +'SCR_T_ORG_TRAIN.csv')

#################################################################################################
#comparing actual and predicted DOC_DS in validation data by plotting
testPredict = model.predict(testX2,batch_size = 1)
plt.plot(scaler00.inverse_transform(testPredict.reshape(-1,1)),'-r',label = 'Model T_iSCR1')
plt.plot(scaler00.inverse_transform(testY2.T_iSCR1_Norm.reshape(-1,1)),'-b',label = 'Actual T_iSCR1')    # Reshaped

#cleaning DOS screen for new prompts
tmp = sp.call('cls',shell=True)

plt.title("Actual & Predicted SCR Temperature (Validation)")
plt.xlabel("Sample Index")
plt.ylabel("Temperature [C]")
plt.legend(loc='best')
plt.savefig(naming + '/' + 'T_iSCR1_Validation_Output.png')
#plt.show()
plt.clf()

prediction = pd.DataFrame(testPredict, columns=['SCR_T_PRD_NORM_VALID']).to_csv(naming + "/" +'SCR_T_PRD_NORM_VALID.csv')
prediction = pd.DataFrame(testY2.T_iSCR1_Norm.reshape(-1,1), columns=['SCR_T_ORG_NORM_VALID']).to_csv(naming + "/" +'SCR_T_ORG_VALID.csv')

prediction = pd.DataFrame(scaler00.inverse_transform(testPredict.reshape(-1,1)), columns=['SCR_T_PRD_VALID']).to_csv(naming + "/" +'SCR_T_PRD_VALID.csv')
prediction = pd.DataFrame(scaler00.inverse_transform(testY2.T_iSCR1_Norm.reshape(-1,1)), columns=['SCR_T_ORG_VALID']).to_csv(naming + "/" +'SCR_T_ORG_VALID.csv')

#################################################################################################
#transforming back the original predictions
#scaler00.inverse_transform(testPredict)

