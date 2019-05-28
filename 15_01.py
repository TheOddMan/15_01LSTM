import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from sklearn.metrics import mean_squared_error



def make_data(s=666,r_start=-50,r_end=51,sam=50):
    random.seed(s)
    x = np.array(random.sample(range(r_start,r_end), sam))
    # y = np.cos(x) + 2 * x + np.sin(x) + np.sqrt(np.power(x, 3))
    y = np.power(x,2)

    test_x = np.array(random.sample(range(-100,101), sam))
    test_y = np.power(test_x,2)

    print("X data : ", x)
    print("X label : ", y)

    x_r = x.reshape(x.shape[0], 1, 1)
    y_r = y.reshape(y.shape[0], 1)

    test_x = test_x.reshape(x.shape[0], 1, 1)
    test_y = test_y.reshape(y.shape[0], 1)

    print(x.shape)
    print(y.shape)

    return x_r,y_r,test_x,test_y

def scal_data(x,y,_plot=False):

    x = x.reshape(x.shape[0],1)

    scalerX = StandardScaler()
    scalerX.fit(x)
    x_scal = scalerX.transform(x)



    scalerY = StandardScaler()
    scalerY.fit(y)
    y_scal = scalerY.transform(y)

    if _plot:

        plt.figure(1,figsize=(10,11))
        plt.subplot(211)
        plt.title("origin data")

        plt.plot(x, "o", color='blue')
        plt.plot(y,  "o",color='orange')
        plt.subplot(212)
        plt.title("zero mean data")
        plt.plot(x_scal,"o", color='blue')
        plt.plot(y_scal,"o", color='orange')

        plt.show()

    x_scal = x_scal.reshape(x_scal.shape[0], 1, 1)

    return scalerX,x_scal,scalerY,y_scal



def build_model():

    model = Sequential()
    model.add(LSTM(4,input_shape=(1,1)))

    model.add(Dense(1))

    return model

def train_model(model,x,y,nepochs=100):

    adam = optimizers.Adam(lr=0.001)

    model.compile(loss='mean_squared_error',optimizer=adam)
    history = model.fit(x,y,epochs=nepochs,batch_size=dataLen,
                        validation_split=0.3,verbose=1)
    model.save("15_01.h5")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(['train','val'],loc='upper right')
    plt.show()

    return model





dataLen = 50

x,y,test_x,test_y = make_data(s=666,r_start=-50,r_end=51,sam=dataLen)

scalerX,x_scal,scalerY,y_scal = scal_data(x,y,_plot=False)

test_scalerX,test_x_scal,test_scalerY,test_y_scal = scal_data(test_x,test_y,_plot=False)

model = build_model()

trained_model = train_model(model,x=x_scal,y=y_scal,nepochs=2000)



yhat = trained_model.predict(x_scal)

mseResult = mean_squared_error(y_scal,yhat)
print("Mse of training data : ",round(mseResult,5))
plt.figure(1)
plt.subplot(211)
plt.plot(y_scal,color="blue",label='desired value')
plt.plot(yhat,color='orange',label='predicted value')
plt.legend(loc='upper right')
plt.title("zero mean range")

yhat = scalerY.inverse_transform(yhat)

plt.subplot(212)
plt.plot(y,color="blue",label='desired value')
plt.plot(yhat,color='orange',label='predicted value')
plt.legend(loc='upper right')
plt.title("origin range")

plt.show()



yhat = trained_model.predict(test_x_scal)

mseResult = mean_squared_error(test_y_scal,yhat)
print("Mse of testing data : ",round(mseResult,5))

plt.figure(1)
plt.subplot(211)
plt.plot(test_y_scal,color="blue",label='desired value')
plt.plot(yhat,color='orange',label='predicted value')
plt.legend(loc='upper right')
plt.title("zero mean range")

yhat = test_scalerY.inverse_transform(yhat)

plt.subplot(212)
plt.plot(test_y,color="blue",label='desired value')
plt.plot(yhat,color='orange',label='predicted value')
plt.legend(loc='upper right')
plt.title("origin range")

plt.show()












