from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Collect input data from the form
        
        coating = int(request.form.get("coating"))  # Numeric (e.g., 1 for Coated, 2 for Uncoated)
        material = int(request.form.get("material"))  # Numeric (e.g., 2=SS202, 3=SS304, 1=Mild Steel)
        ash = int(request.form.get("ash"))  # Numeric (e.g., 2 for Flyash, 1 for Bottom)
        time = float(request.form.get("time"))
        concentration = float(request.form.get("concentration"))
        speed = float(request.form.get("speed"))

        # Missing flags (set to 0 for no missing values)
        coating_is_missing = 0
        material_is_missing = 0
        ash_is_missing = 0

        # Prepare the input array
        input_data = np.array([[ coating, material, ash, time, concentration, speed,
                                coating_is_missing, material_is_missing, ash_is_missing]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction = np.round(prediction, 2)
        return render_template(
            "index.html", 
            prediction=prediction,
            coating=coating, 
            material=material,
            ash=ash,
            time=time,
            concentration=concentration,
            speed=speed
        )

    return render_template("index.html", prediction=prediction)



if __name__ == "__main__":
    app.run(debug=True)
