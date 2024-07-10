from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import os

# Load the model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///anemia.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class AnemiaData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.Integer, nullable=False)
    hemoglobin = db.Column(db.Float, nullable=False)
    mch = db.Column(db.Float, nullable=False)
    mchc = db.Column(db.Float, nullable=False)
    mcv = db.Column(db.Float, nullable=False)
    result = db.Column(db.Integer, nullable=False)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        hemoglobin = float(request.form['hemoglobin'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])
        mcv = float(request.form['mcv'])
        
        gender_encode = 0 if gender == 'Male' else 1
    
        features = np.array([[gender_encode, hemoglobin, mch, mchc, mcv]])    
        features = scaler.transform(features)
        
        prediction = model.predict(features)
        result = 1 if prediction[0] == 1 else 0
        output = 'Anemia' if result == 1 else 'No Anemia'

        new_data = AnemiaData(gender=gender_encode, hemoglobin=hemoglobin, mch=mch, mchc=mchc, mcv=mcv, result=result)
        db.session.add(new_data)
        db.session.commit()
        
        return render_template('home.html', pred_out=f'Result: {output}')

def initialize_database():
    """Initializes the database and tables if they don't exist."""
    if not os.path.exists('anemia.db'):
        print("Creating database tables...")
        db.create_all()
        print("Tables created.")
    else:
        # Reflect the tables to check if they exist
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        if 'anemia_data' not in tables:
            print("Creating database tables...")
            db.create_all()
            print("Tables created.")

if __name__ == '__main__':
    with app.app_context():
        initialize_database()
    app.run(debug=True)
