import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_fake_job_posts(num_records=5000):
    data = []
    
    # Lists for realistic data
    industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 
                 'Retail', 'Consulting', 'Marketing', 'Real Estate', 'Transportation']
    
    functions = ['Information Technology', 'Engineering', 'Sales', 'Marketing', 
                'Operations', 'Finance', 'Human Resources', 'Customer Service', 
                'Research', 'Management']
    
    employment_types = ['Full-time', 'Part-time', 'Contract', 'Internship', 'Freelance']
    
    experience_levels = ['Entry level', 'Mid level', 'Senior level', 'Executive level', 'Not specified']
    
    education_levels = ['High School', 'Bachelor\'s Degree', 'Master\'s Degree', 
                       'PhD', 'Not specified']
    
    locations = ['New York, NY', 'San Francisco, CA', 'London, UK', 'Berlin, Germany', 
                'Tokyo, Japan', 'Sydney, Australia', 'Toronto, Canada', 'Mumbai, India',
                'Singapore', 'Remote']
    
    # Fake job post templates
    fake_descriptions = [
        "We are looking for a talented {title} to join our team. This is an exciting opportunity to work on cutting-edge projects.",
        "Join our growing company as a {title}. We offer competitive salary and great benefits.",
        "Exciting {title} position available. Apply now for immediate consideration.",
        "We need a skilled {title} with experience in {industry}. Great pay and benefits.",
        "Looking for a motivated {title} to help us expand our business in {location}."
    ]
    
    real_descriptions = [
        "We are seeking a qualified {title} to join our established team. The ideal candidate will have experience in {function} and a degree in {education}.",
        "Our company is hiring a {title} for our {location} office. This role involves {function} responsibilities and requires {experience} experience.",
        "Join our reputable organization as a {title}. We offer comprehensive benefits, professional development opportunities, and a collaborative work environment.",
        "We are looking for a dedicated {title} with strong skills in {function}. The position offers competitive compensation and career growth potential.",
        "Established company seeking experienced {title} for {industry} role. Must have relevant qualifications and proven track record."
    ]
    
    for i in range(num_records):
        job_id = f"JOB_{i+1:05d}"
        
        # Decide if fake or real (roughly 20% fake)
        fraudulent = 1 if random.random() < 0.2 else 0
        
        # Generate data based on fraudulent status
        if fraudulent:
            title = fake.job()
            company = fake.company()
            description = random.choice(fake_descriptions).format(
                title=title.lower(), industry=random.choice(industries).lower(),
                location=random.choice(locations).lower()
            )
            requirements = f"Experience in {random.choice(functions)}. {fake.sentence()}"
            salary = f"${random.randint(50000, 200000)} - ${random.randint(80000, 300000)}"
            has_company_logo = random.choice([0, 1])
            has_questions = random.choice([0, 1])
            telecommuting = random.choice([0, 1])
        else:
            title = fake.job()
            company = fake.company()
            description = random.choice(real_descriptions).format(
                title=title.lower(), function=random.choice(functions).lower(),
                education=random.choice(education_levels).lower(),
                location=random.choice(locations), experience=random.choice(experience_levels).lower(),
                industry=random.choice(industries).lower()
            )
            requirements = f"Bachelor's degree in relevant field. {random.randint(1, 5)} years of experience in {random.choice(functions)}. {fake.sentence()}"
            salary = f"${random.randint(40000, 150000)} - ${random.randint(60000, 200000)}"
            has_company_logo = 1  # Real jobs more likely to have logo
            has_questions = 1  # Real jobs more likely to have questions
            telecommuting = random.choice([0, 1])
        
        location = random.choice(locations)
        employment_type = random.choice(employment_types)
        industry = random.choice(industries)
        function = random.choice(functions)
        required_experience = random.choice(experience_levels)
        required_education = random.choice(education_levels)
        
        # Generate posted date (last 2 years)
        days_ago = random.randint(0, 730)
        posted_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        data.append({
            'job_id': job_id,
            'title': title,
            'company': company,
            'location': location,
            'description': description,
            'requirements': requirements,
            'salary': salary,
            'employment_type': employment_type,
            'posted_date': posted_date,
            'has_company_logo': has_company_logo,
            'has_questions': has_questions,
            'telecommuting': telecommuting,
            'industry': industry,
            'function': function,
            'required_experience': required_experience,
            'required_education': required_education,
            'fraudulent': fraudulent
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_fake_job_posts(5000)
    df.to_csv('data/fake_job_posts.csv', index=False)
    print("Generated 5000 fake job posts and saved to data/fake_job_posts.csv")
    print(f"Distribution: {df['fraudulent'].value_counts()}")