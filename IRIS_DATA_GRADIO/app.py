import gradio as gr
import pickle

def predict_iris_class(sepal_length, sepal_width, petal_length, petal_width):
    with open("iris_voting_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = loaded_model.predict(input_data)[0]

    return prediction


iface = gr.Interface(fn=predict_iris_class,
                     inputs=[gr.Number(label="sepal length (cm)",minimum=0),
                             gr.Number(label="sepal width (cm)",minimum=0),
                             gr.Number(label="petal length (cm)",minimum=0),
                             gr.Number(label="petal width (cm)",minimum=0)],
                             
                             outputs=gr.Textbox(label="flower iris prediction"),
                             title="Iris flower classifier",
                             description="Enters the flower's measurment to predict its class")

iface.launch()