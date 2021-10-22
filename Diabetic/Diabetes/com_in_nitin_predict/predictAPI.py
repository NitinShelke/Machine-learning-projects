from sklearn.preprocessing import StandardScaler
import pickle


def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    preg = int(Pregnancies)
    glucose = float(Glucose)
    bp = float(BloodPressure)
    st = float(SkinThickness)
    insulin = float(Insulin)
    bmi = float(BMI)
    dpf = float(DPF)
    age = int(Age)
    
    sc=StandardScaler()
    model=pickle.load(open('model/model.pkl','rb'))

    x = [[preg, glucose, bp, st, insulin, bmi, dpf, age]]
    x = sc.fit_transform(x)
    prediction= model.predict(x)
    if prediction:
        return 'Oops! You have diabetes.'
    else:
        return "Great! You don't have diabetes."

