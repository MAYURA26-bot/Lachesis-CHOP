# These recommendations are NOT real medical advice.
# They are part of a university capstone project and should only be used for academic purposes.
# Always consult a qualified health professional for real medical guidance.

lables = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Obesity Type_I",
    3: "Obesity Type_II",
    4: "Obesity Type_III",
    5: "Overweight Level_I",
    6: "Overweight Level_II"
}

info_link = " For more information please visit: https://www.health.vic.gov.au/preventive-health/healthy-eating-programs-and-services"

disclaimer = "This is not real medical advice."

recomendation = {
    "Insufficient Weight": {
        "Food/Drink": "Increase caloric intake with balanced, nutrient-rich meals. Incorporate healthy fats such as avocados, nuts, and olive oils.",
        "Exercise": "Include strength training to build muscle mass.",
        "Other": "Consult a GP or medical professional if underweight persists. " + info_link,
        "Note": disclaimer
    },
    "Normal Weight": {
        "Food/Drink": "Maintain a balanced diet (lean meats, fruits, vegetables).",
        "Exercise": "Continue low intensity exercise (e.g., brisk walk 30 mins daily) or high intensity exercise 2–3 times a week.",
        "Other": "Maintain regular health check-ups with a GP. " + info_link,
        "Note": disclaimer
    },
    "Overweight Level_I": {
        "Food/Drink": "Reduce sugary drinks and processed foods. Eat smaller portions and prioritise low-calorie foods such as fresh fruit and vegetables.",
        "Exercise": "Do moderate exercise daily for 30–60 minutes.",
        "Other": "Track meals and log food. " + info_link,
        "Note": disclaimer
    },
    "Overweight Level_II": {
        "Food/Drink": "Reduce sugary drinks and processed foods. Eat smaller portions and prioritise low-calorie and high-fibre foods such as fresh fruit and vegetables. Limit fast food and fried foods.",
        "Exercise": "Include structured exercise such as cardio plus resistance/weight training.",
        "Other": "Consider counselling from a medical professional such as a nutritionist. " + info_link,
        "Note": disclaimer
    },
    "Obesity Type_I": {
        "Food/Drink": "Reduce sugary drinks and processed foods. Eat smaller portions, prioritise lean meats, and low-calorie, high-fibre foods such as fresh fruit and vegetables. Limit fast food and fried foods.",
        "Exercise": "Limit sedentary time and increase overall activity levels with daily brisk walks as well as structured exercise.",
        "Other": "Consider counselling from a medical professional (e.g., nutritionist). Consider group support such as fitness groups or diet programs. " + info_link,
        "Note": disclaimer
    },
    "Obesity Type_II": {
        "Food/Drink": "Develop a personalised meal and diet plan with a professional.",
        "Exercise": "Gradually increase physical activity – likely regular and low-impact exercise supervised and recommended by a specialist.",
        "Other": "Monitor blood pressure and glucose levels. Strongly recommended to consult a medical specialist. " + info_link,
        "Note": disclaimer
    },
    "Obesity Type_III": {
        "Food/Drink": "Consult a medical professional to develop a medically approved diet plan – avoid crash diets.",
        "Exercise": "Consult a specialist to develop a personalised plan tailored to your physical and physiological abilities.",
        "Other": "Urgent medical/nutritional supervision is required. " + info_link,
        "Note": disclaimer
    }
}

def generate_recommendations(prediction_result: int) -> dict:
    label = lables.get(prediction_result)
    if label is None:
        return {
            "Food/Drink": "No specific recommendations available.",
            "Exercise": "No specific recommendations available.",
            "Other": "No specific recommendations available. " + info_link,
            "Note": disclaimer
        }
    return recomendation.get(label, {})