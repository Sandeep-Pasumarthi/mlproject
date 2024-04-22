from flask import Flask, request, render_template
from src.pipeline.predict import CustomData, PredictPipeline


application = Flask(__name__)
app =application

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        df = data.get_data_df()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(df)
        return render_template("home.html", results=prediction)
    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0')
