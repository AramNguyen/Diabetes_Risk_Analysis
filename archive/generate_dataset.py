import numpy as np
import pandas as pd
from scipy.stats import truncnorm, skewnorm
from scipy.special import expit

def generate_data(n=15000, seed=42):
    np.random.seed(seed)
    
    # 1. Patient ID
    patient_id = np.arange(1, n + 1)
    
    # 2. Age (Skewed to older but bounded 18-95)
    # Most diabetes datasets are skewed older
    age = np.clip(np.round(skewnorm.rvs(a=-2, loc=55, scale=15, size=n)), 18, 95).astype(int)
    
    # 3. Gender
    gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.51, 0.48, 0.01], size=n)
    
    # 4. City
    cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Kolkata', 'Surat', 'Pune', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna']
    city_probs = np.array([0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02])
    city_probs /= city_probs.sum()
    city = np.random.choice(cities, p=city_probs, size=n)
    
    # 5. Income Bracket (correlated with city)
    # Tier 1 cities have higher income
    tier1 = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Pune']
    is_tier1 = np.isin(city, tier1)
    income_probs_t1 = [0.15, 0.50, 0.35] # Low, Middle, High
    income_probs_t2 = [0.35, 0.50, 0.15]
    income_bracket = np.where(
        is_tier1, 
        np.random.choice(['Low', 'Middle', 'High'], p=income_probs_t1, size=n),
        np.random.choice(['Low', 'Middle', 'High'], p=income_probs_t2, size=n)
    )
    
    # 6. Physical Activity (Correlated with age and income)
    # Older people and high income tend to be more sedentary in India
    activity_score = 0.5 * (age < 40) - 0.5 * (age > 60) - 0.3 * (income_bracket == 'High') + np.random.normal(0, 1, n)
    activity_bins = np.percentile(activity_score, [33, 66])
    physical_activity = np.where(activity_score < activity_bins[0], 'Sedentary',
                        np.where(activity_score < activity_bins[1], 'Moderate', 'Active'))
    
    # 7. Diet Type
    diet = np.random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan', 'Pescatarian'], p=[0.40, 0.55, 0.02, 0.03], size=n)
    
    # 8. Stress Level (1-10)
    # Higher in Tier 1 cities and High income
    stress_base = 5 + 1.5 * is_tier1 + 1.0 * (income_bracket == 'High') - 0.5 * (physical_activity == 'Active')
    stress_level = np.clip(np.round(stress_base + np.random.normal(0, 2, n)), 1, 10).astype(int)
    
    # 9. Sleep (Integer/half-hour clustering)
    # Normal distribution doesn't cluster on 6, 7, 8. We force it.
    sleep_base = 8.5 - 0.2 * stress_level + np.random.normal(0, 1, n)
    sleep_base = np.clip(sleep_base, 3, 12)
    # Apply digit preference to sleep (clustering on exact hours or half hours)
    rnd = np.random.rand(n)
    hours_sleep = np.where(rnd < 0.6, np.round(sleep_base), 
                  np.where(rnd < 0.8, np.round(sleep_base*2)/2, sleep_base))
    # Round to 1 decimal
    hours_sleep = np.round(hours_sleep, 1)

    # 10. Family History
    family_hist = np.random.choice(['Yes', 'No'], p=[0.35, 0.65], size=n)
    
    # 11. BMI (Correlated with activity, diet, age)
    # Log-normal for long right tail
    bmi_log_mean = 3.1 + 0.1 * (physical_activity == 'Sedentary') - 0.1 * (physical_activity == 'Active') + 0.05 * (age > 45)
    bmi = np.exp(np.random.normal(bmi_log_mean, 0.15))
    bmi = np.clip(bmi, 15, 60) # Some outliers up to 60
    
    # 12. Waist Circumference (Strongly correlated with BMI, heteroskedastic noise)
    # Variance increases with BMI
    waist = 40 + 2.5 * bmi + np.random.normal(0, 0.5 + 0.1 * bmi, n)
    # Males tend to have larger waist for same BMI
    waist += 5 * (gender == 'Male')
    waist = np.round(waist, 1)
    bmi = np.round(bmi, 1)
    
    # 13. Fasting Blood Sugar
    fbs_base = 70 + 0.5 * age + 1.5 * bmi + 10 * (family_hist == 'Yes') + 2 * stress_level
    fbs = fbs_base + np.random.normal(0, 15, n)
    # Add non-linear exponential tail for severe diabetics
    diabetic_tail = np.random.exponential(30, n) * (fbs > 120)
    fbs += diabetic_tail
    fbs = np.clip(np.round(fbs), 60, 400).astype(int)
    
    # 14. HbA1c (Correlated with FBS, heteroskedastic)
    hba1c = 3.5 + 0.02 * fbs + np.random.normal(0, 0.002 * fbs, n)
    hba1c = np.clip(np.round(hba1c, 1), 3.5, 15.0)
    
    # 15. Blood Pressure
    bp_sys_base = 90 + 0.5 * age + 0.8 * bmi + 1.5 * stress_level
    bp_sys = bp_sys_base + np.random.normal(0, 10, n)
    
    bp_dia_base = 60 + 0.2 * age + 0.5 * bmi + 1.0 * stress_level
    bp_dia = bp_dia_base + np.random.normal(0, 7, n)
    
    # Digit preference function
    def apply_digit_preference(x):
        x = np.round(x).astype(int)
        r = np.random.rand(len(x))
        # 50% rounded to nearest 10, 30% to nearest even number, 20% left alone
        mask_10 = r < 0.5
        mask_even = (r >= 0.5) & (r < 0.8)
        x[mask_10] = np.round(x[mask_10] / 10) * 10
        x[mask_even] = np.round(x[mask_even] / 2) * 2
        return x

    bp_sys = apply_digit_preference(bp_sys)
    bp_sys = np.clip(bp_sys, 80, 250)
    
    bp_dia = apply_digit_preference(bp_dia)
    bp_dia = np.clip(bp_dia, 40, 140)

    # 16. Smoking and Alcohol
    smoking = np.random.choice(['Never', 'Current', 'Former'], p=[0.7, 0.2, 0.1], size=n)
    
    # Alcohol with MNAR (Missing Not At Random)
    alcohol_base = np.random.choice(['Never', 'Occasional', 'Regular'], p=[0.5, 0.4, 0.1], size=n)
    
    # 17. Target (Diabetes Risk)
    # Log odds with interactions and irreducible noise
    # Interaction: BMI * Age (older obese people have much higher risk)
    # Interaction: HbA1c * FBS
    log_odds = -12 + 0.03*age + 0.05*bmi + 0.5*(family_hist=='Yes') + 0.8*hba1c + 0.005*fbs
    log_odds += 0.001 * (age * bmi) + 0.005 * (hba1c * fbs)
    log_odds += np.random.normal(0, 1.5, n)  # Irreducible noise to prevent 99% LR accuracy
    prob = expit(log_odds)
    
    # We want roughly 60% Low, 25% Moderate, 15% High
    # Since prob is somewhat calibrated, we can use percentiles
    p_33 = np.percentile(prob, 60)
    p_66 = np.percentile(prob, 85)
    
    risk = np.where(prob < p_33, 'Low',
           np.where(prob < p_66, 'Moderate', 'High'))
           
    # Assemble DataFrame
    df = pd.DataFrame({
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'city': city,
        'bmi': bmi,
        'family_history_diabetes': family_hist,
        'physical_activity_level': physical_activity,
        'diet_type': diet,
        'smoking_status': smoking,
        'alcohol_consumption': alcohol_base,
        'hours_sleep_per_night': hours_sleep,
        'stress_level': stress_level,
        'fasting_blood_sugar': fbs,
        'hba1c_level': hba1c,
        'blood_pressure_systolic': bp_sys,
        'blood_pressure_diastolic': bp_dia,
        'waist_circumference_cm': waist,
        'income_bracket': income_bracket,
        'diabetes_risk': risk
    })
    
    # 18. Inject Missingness
    # Missing completely at random (MCAR) for smoking and income
    df.loc[np.random.rand(n) < 0.03, 'smoking_status'] = np.nan
    df.loc[np.random.rand(n) < 0.03, 'income_bracket'] = np.nan
    
    # Missing not at random (MNAR) for alcohol
    # More missing for females (cultural bias) and older people
    missing_prob_alcohol = 0.1 + 0.3 * (df['gender'] == 'Female') + 0.2 * (df['age'] > 60)
    missing_prob_alcohol = np.clip(missing_prob_alcohol, 0, 1)
    df.loc[np.random.rand(n) < missing_prob_alcohol, 'alcohol_consumption'] = np.nan
    
    return df

if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_data(n=15000, seed=42)
    output_file = "diabetes_risk_india_2026_rebuilt.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
