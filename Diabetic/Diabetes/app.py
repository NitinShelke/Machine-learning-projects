from flask import  Flask,request,render_template
import pickle
from com_in_nitin_predict.predictAPI import predict_diabetes

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    preg=int(request.form['pregnancies'])
    glucose=float(request.form['glucose'])
    bp=float(request.form['bloodpressure'])
    st=float(request.form['skinthickness'])
    insulin=float(request.form['insulin'])
    bmi=float(request.form['bmi'])
    dpf=float(request.form['dpf'])
    age=int(request.form['age'])

    prediction=predict_diabetes(preg,glucose,bp,st,insulin,bmi,dpf,age)

    return render_template('index.html', prediction_text=prediction)




if __name__=='__main__':
    app.run(debug=True)