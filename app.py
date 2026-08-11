import pickle
import numpy as np
import gradio as gr

# Load the saved model and feature names
with open("model.pkl", "rb") as file:
    model, feature_names = pickle.load(file)


def predict_housing_price(
    MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
):
    # Pack inputs into array matching feature order
    input_data = np.array(
        [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]]
    )

    # Predict
    prediction = model.predict(input_data)[0]

    # Note: California housing target is in hundreds of thousands ($100,000s)
    return f"${prediction * 100000:,.2f}"


# Define Gradio UI components
demo = gr.Interface(
    fn=predict_housing_price,
    inputs=[
        gr.Number(label="MedInc (Median Income in block group)", value=3.5),
        gr.Number(label="HouseAge (Median house age in years)", value=28.0),
        gr.Number(label="AveRooms (Average number of rooms per household)", value=5.0),
        gr.Number(label="AveBedrms (Average number of bedrooms per household)", value=1.0),
        gr.Number(label="Population (Block group population)", value=1425.0),
        gr.Number(label="AveOccup (Average number of household members)", value=3.0),
        gr.Number(label="Latitude (Block group latitude)", value=37.88),
        gr.Number(label="Longitude (Block group longitude)", value=-122.23),
    ],
    outputs=gr.Textbox(label="Predicted Median House Value"),
    title="California Housing Price Predictor",
    description="Enter the demographic and geographic attributes below to predict the median house value.",
)

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=True to get a public link for the app