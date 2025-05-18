from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class UserInput(BaseModel):
    Age: int
    Gender: str
    Fitness_Goal: str
    Workout_Experience: str
    Hours_per_Week: int
    Workout_Type: str
    Timing: str
    Budget: str

@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict")
def predict_plan(user_input: UserInput):
    input_df = pd.DataFrame([{
        "Age": user_input.Age,
        "Gender": user_input.Gender,
        "Fitness Goal": user_input.Fitness_Goal,
        "Workout Experience": user_input.Workout_Experience,
        "Hours/Week": user_input.Hours_per_Week,
        "Workout Type": user_input.Workout_Type,
        "Timing": user_input.Timing,
        "Budget": user_input.Budget
    }])
    prediction = model.predict(input_df)[0]
    return {"Recommended Plan": prediction}
