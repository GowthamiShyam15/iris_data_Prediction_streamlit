import numpy as np
import pickle

load1 = pickle.load(open("model.sav","rb"))
load2 = pickle.load(open("scaler.sav","rb"))

input_data = (65,	5.6,	2.9, 3.6,	1.3)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = load2.transform(input_data_reshaped)
# print(std_data)

prediction = load1.predict(std_data)
print(prediction)

if (prediction[0] == "Iris-setosa"):
  print('The flower is Iris-setosa')
elif (prediction[0]=="Iris-versicolor"):
    print("The flower is Iris-versicolor")
else:
  print('The flower is Iris-virginica')