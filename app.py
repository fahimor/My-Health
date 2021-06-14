# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from DiseasePrediction import DiseasePredict
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#List of the symptoms is listed here in list l1.

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']

precautions = [['Fungal infection','bath twice','use detol or neem in bathing water','keep infected area dry','use clean cloths'],
               ['Allergy','apply calamine','cover area with bandage','use ice to compress itching'],
               ['GERD','avoid fatty spicy food','avoid lying down after eating','maintain healthy weight','exercise'],
               ['Chronic cholestasis', 'cold baths','anti itch medicine','consult doctor','eat healthy'],
               ['Drug Reaction','stop irritation','consult nearest hospital','stop taking drug','follow up'],
               ['Peptic ulcer diseae','avoid fatty spicy food','consume probiotic food','eliminate milk','limit alcohol'],
               ['AIDS','avoid open cuts','wear ppe if possible','consult doctor','follow up'],
               ['Diabetes','have balanced diet','exercise','consult doctor','follow up'],
               ['Gastroenteritis','stop eating solid food for while', 'try taking small sips of water','rest','ease back into eating'],
               ['Bronchial Asthma','switch to loose cloothing','take deep breaths','get away from trigger','seek help'],
               ['Hypertension','meditation','salt baths','reduce stress','get proper sleep'],
               ['Migraine', 'meditation','reduce stress','use poloroid glasses in sun','consult doctor'],
               ['Cervical spondylosis', 'use heating pad or cold pack', 'exercise','take otc pain reliver','consult doctor'],
               ['Paralysis (brain hemorrhage)', 'massage', 'eat healthy', 'exercise','consult doctor'],
               ['Jaundice', 'drink plenty of water','consume milk thistle','eat fruits and high fiberous food','medication'],
               ['Malaria', 'Consult nearest hospital', 'avoid oily food', 'avoid non veg food','keep mosquitos out'],
               ['Chicken pox' ,'use neem in bathing','consume neem leaves','take vaccine','avoid public places'],
               ['Dengue',  'drink papaya leaf juice','avoid fatty spicy food','keep mosquitos away','keep hydrated'],
               ['Typhoid',  'eat high calorie vegitables', 'antiboitic therapy', 'consult doctor','medication'],
               ['hepatitis A',   'Consult nearest hospital','wash hands through','avoid fatty spicy food','medication'],
               ['Hepatitis B',  'consult nearest hospital','vaccination','eat healthy','medication'],
               ['Hepatitis C',  'consult nearest hospital','vaccination','eat healthy','medication'],
               ['Hepatitis D',   'consult doctor','medication','eat healthy','follow up'],
               ['Hepatitis E',  'stop alcohol consumption','rest','consult doctor','medication'],
               ['Alcoholic hepatitis', 'stop alcohol consumption','consult doctor','medication','follow up'],
               ['Tuberculosis',  'cover mouth','consult doctor','medication','rest'],
               ['Common Cold','drink vitamin c rich drinks','take vapour','avoid cold food','keep fever in check'],
               ['Pneumonia',  'consult doctor','medication','rest','follow up'],
               ['Dimorphic hemmorhoids(piles)',  'avoid fatty spicy food','consume witch hazel','warm bath with epsom salt','consume alovera juice'],
               ['Heart attack',  'call ambulance','chew or swallow asprin','keep calm'],
               ['Varicose veins',  'lie down flat and raise the leg high','use oinments','use vein compression','dont stand still for long'],
               ['Hypothyroidism',  'reduce stress','exercise','eat healthy','get proper sleep'],
               ['Hyperthyroidism',  'eat healthy','massage','use lemon balm','take radioactive iodine treatment'],
               ['Hypoglycemia',  'lie down on side','check in pulse','drink sugary drinks','consult doctor'],
               ['Osteoarthristis',  'acetaminophen','consult nearest hospital','follow up','salt baths'],
               ['Arthritis',  'exercise','use hot and cold therapy','try acupuncture','massage'],
               ['(vertigo) Paroymsal Positional Vertigo',  'lie down','avoid sudden change in body','avoid abrupt head movment','relax'],
               ['Acne',  'bath twice','avoid fatty spicy food','drink plenty of water','avoid too many products'],
               ['Urinary tract infection', 'drink plenty of water','increase vitamin c intake','drink cranberry juice','take probiotics'],
               ['Psoriasis',  'wash hands with warm soapy water','stop bleeding using pressure','consult doctor','salt baths'],
               ['Impetigo', 'soak affected area in warm water','use antibiotics','remove scabs with wet compressed cloth','consult doctor']]

disease=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_disease(data:DiseasePredict):
    data = data.dict()
    sym1=data['sym1']
    sym2=data['sym2']
    sym3=data['sym3']
    sym4=data['sym4']
    sym5=data['sym5']
    
    
    l2=[]
    for i in range(0,len(l1)):
        l2.append(0)
    
    if sym1 > 0:       
        l2[sym1-1] = 1
    if sym2 > 0:
        l2[sym2-1] = 1
    if sym3 > 0:
        l2[sym3-1] = 1
    if sym4 > 0:
        l2[sym4-1] = 1
    if sym5 > 0:
        l2[sym5-1] = 1
            
    inputtest = [l2]
    
    
    
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict(inputtest)
  
    prediction1=disease[prediction[0]]
    
    import json

    aList = precautions[prediction[0]]
    jsonStr = json.dumps(aList)
  
     
    return {
        'prediction': prediction1,
        'precautions to take': aList
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload