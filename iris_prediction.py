import numpy as np
import pickle
import streamlit as st

load1 = pickle.load(open("model.sav","rb"))
load2 = pickle.load(open("scaler.sav","rb"))

def iris_predict(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    std_data = load2.transform(input_data_reshaped)
    # print(std_data)

    prediction = load1.predict(std_data)
    return prediction

    if (prediction[0] == "Iris-setosa"):
        return 'The flower is Iris-setosa'
    elif (prediction[0]=="Iris-versicolor"):
        return "The flower is Iris-versicolor"
    else:
        return 'The flower is Iris-virginica'
    
def main():
    #title
    st.title("Iris Prediction Web Application")
    #giving input data
    id = st.text_input("enter id:")
    sepalLength = st.text_input("enter sepal length in cm")
    sepalWidth = st.text_input("enter sepal width in cm")
    petalLength = st.text_input("enter petal length in cm")
    petalWidth = st.text_input("enter petal width in cm")

    predictions = ""
    #creating a button
    if st.button("IRIS TEST RESULT"):
        predictions = iris_predict([id, sepalLength, sepalWidth, petalLength, petalWidth])

    st.success(predictions)


if __name__ == "__main__":
    main()