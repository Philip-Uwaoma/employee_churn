#import streamlit as st
#st.title('🎈 App Name')
#st.write('Hello world!')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime
from openai import AzureOpenAI

# Azure OpenAI Client Setup
client = AzureOpenAI(
    api_key="e258b92d9bc14f209e75dd961be0afbb",
    api_version="2024-02-01",
    azure_endpoint="https://evasnapnetopenai.openai.azure.com/",
)

# Function to fetch and process CSV data
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY/main/employee_data.csv"  
    df = pd.read_csv(url)
    # Process the DataFrame
    filtered_df = process_dataframe(df)
    return filtered_df

# Functions for data processing
def process_dataframe(df):
    df['full_name'] = df['First Name'] + ' ' + df['Last Name']
    current_year = datetime.now().year
    df['Age'] = current_year - pd.to_datetime(df['Date Of Birth']).dt.year
    df['tenure_at_company'] = current_year - pd.to_datetime(df['Hire Date']).dt.year
    df['tenure_at_company'] = df['tenure_at_company'].apply(lambda x: '<1' if x < 1 else x)
    required_columns = [
        'Gender', 'Marital Status', 'Department', 'Designation', 'Grade', 'Employment Type',
        'Work Model', 'Number Of Days Per Week', 'Leave Length', 'Payroll Type', 'Montly Gross',
        'Salary Frequency', 'Age', 'tenure_at_company'
    ]
    return df.dropna(subset=required_columns)

def transform_dataframe(df):
    # Create the 'full_name' column by concatenating 'First Name' and 'Last Name'
    
    
    # Create the text column with the required format
    df['details'] = df.apply(lambda row: f"Gender: {row['Gender']}, Marital Status: {row['Marital Status']}, Department: {row['Department']}, "
                                         f"Designation: {row['Designation']}, Grade: {row['Grade']}, Employment Type: {row['Employment Type']}, "
                                         f"Work Model: {row['Work Model']}, Number of Days Per Week: {row['Number Of Days Per Week']}, "
                                         f"Leave Length: {row['Leave Length']}, Payroll Type: {row['Payroll Type']}, Montly Gross: {row['Montly Gross']}Naira, "
                                         f"Salary Frequency: {row['Salary Frequency']}, Age: {row['Age']}, tenure_at_company: {row['tenure_at_company']}", axis=1)
    
    # Create the new dataframe with the required columns
    new_df = df[['full_name', 'Work Status', 'details']]
    
    return new_df

'''def few_shot_prediction(employee_data):
    prompt = f"""
    Predict churn likelihood for the following employee:
    Employee data: {employee_data}
    """
    response = client.chat.completions.create(
        model="gpt-4o-new",
        messages=[
            {"role": "system", "content": "You are a machine learning model predicting employee churn."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.1
    )
    content = response.choices[0].message.content.strip()
    # Parse response
    try:
        prediction_percentage = float(content.split("Likelihood of Churn:")[1].split("%")[0].strip())
        summary = content.split("Summary:")[1].split("Key Features Analysis:")[0].strip()
        feature_analysis = content.split("Key Features Analysis:")[1].strip()
        if prediction_percentage < 25:
            prediction_label = "Not likely to churn"
            color = "green"
        elif 25 <= prediction_percentage <= 50:
            prediction_label = "Less likely to churn"
            color = "lightgreen"
        elif 50 < prediction_percentage <= 75:
            prediction_label = "Likely to churn"
            color = "yellow"
        else:
            prediction_label = "Very likely to churn"
            color = "red"
    except:
        prediction_percentage, prediction_label, summary, feature_analysis = "Error", "Error", "Parsing Error", ""
        color = "grey"
    return prediction_percentage, prediction_label, summary, feature_analysis, color'''


def few_shot_prediction(employee_data):
    # Few-shot examples for churn prediction
    few_shot_prompt = """
    Observe the employee data provided and identify relationships between variables like a machine learning algorithm. 
    Predict the likelihood of churn for the employee as a percentage titled "Likelihood of Churn:" (between 0% and 100%)
    
    Additionally, provide:
    - A categorization (titled "Category:") categorized as:
        - "Not likely to churn" (if prediction is less than 25%),
        - "Less likely to churn" (if prediction is 25%-50%),
        - "Likely to churn" (if prediction is 50%-75%),
        - "Very likely to churn" (if prediction is above 75%).
    - A brief summary (titled "Summary:") explaining the prediction.
    - An analysis of key features (titled "Key Features Analysis:") in the format:
      "feature: positive relationship (or negative relationship): reason".
      
    Employee data: {employee_data}
    """.format(employee_data=employee_data)

    # OpenAI API chat message structure
    messages = [
        {"role": "system", "content": "You are a helpful assistant that simulates the behavior of a machine learning model."},
        {"role": "user", "content": few_shot_prompt}
    ]

    # Call to OpenAI Chat Completions API
    response = client.chat.completions.create(
        model="gpt-4o-new",  # Ensure this matches your model deployment
        messages=messages,
        max_tokens=700,
        temperature=0.1
    )

    # Extract the content from the response
    content = response.choices[0].message.content.strip()
    print(content)

    # Initialize default return values
    prediction_percentage = 0
    prediction_label = "Prediction not available"
    summary = "Summary not found in response."
    feature_analysis = "Feature analysis not found in response."

    # Parse the response to extract the churn prediction percentage, summary, and feature analysis
    try:
        # Extract the prediction percentage
        percentage_start = content.find("Likelihood of Churn:")
        percentage_end = content.find("%", percentage_start)
        if percentage_start != -1 and percentage_end != -1:
            prediction_percentage = float(content[percentage_start + len("Likelihood of Churn:"):percentage_end].strip())

        # Determine the prediction label based on the prediction percentage
        if prediction_percentage < 25:
            prediction_label = "Not likely to churn"
            color = "green"
        elif 25 <= prediction_percentage <= 50:
            prediction_label = "Less likely to churn"
            color = "lightgreen"
        elif 50 < prediction_percentage <= 75:
            prediction_label = "Likely to churn"
            color = "yellow"
        else:
            prediction_label = "Very likely to churn"
            color = "red"

        # Extract the summary
        summary_start = content.find("Summary:")
        analysis_start = content.find("Key Features Analysis:")
        if summary_start != -1 and analysis_start != -1:
            summary = content[summary_start + len("Summary:"):analysis_start].strip()

        # Extract the feature analysis
        analysis_start = content.find("Key Features Analysis:")
        if analysis_start != -1:
            feature_analysis = content[analysis_start + len("Key Features Analysis:"):].strip()

    except Exception as e:
        # Handle any parsing errors
        prediction_percentage = "Prediction percentage parsing error"
        prediction_label = "Prediction label parsing error"
        summary = "Summary parsing error"
        feature_analysis = "Feature analysis parsing error"

    return prediction_percentage, prediction_label, summary, feature_analysis, color


# Main Streamlit app
st.title("EMPLOYEE CHURN PREDICTION")
st.write("Welcome! This is a basic webpage to display employee churn predictions for individuals and companies. "
         "Select an individual to see the person's prediction or select a company and department to see the company's/department's prediction.")

# Load data
filtered_df = load_and_process_data()

# Sidebar for navigation
option = st.sidebar.selectbox("Choose Prediction Type", ["Individual Prediction", "Company Prediction"])

if option == "Individual Prediction":
    selected_name = st.sidebar.selectbox("Select Employee Name", filtered_df['full_name'].unique())
    employee_data = filtered_df[filtered_df['full_name'] == selected_name].iloc[0]
    row = transform_dataframe(employee_data)
    full_name = row['full_name']
    work_status = row['Work Status']
    details = row['details']
    #details = employee_data.to_dict()
    prediction_percentage, prediction_label, summary, feature_analysis, color = few_shot_prediction(details)

    # Display Prediction Results
    st.markdown(f"**{selected_name}**")
    st.write(f"Company: {employee_data['Company Name']} / Department: {employee_data['Department']} / Status: {employee_data['Work Status']}")
    st.write(f"### Prediction: {prediction_percentage}%")
    fig, ax = plt.subplots()
    wedges, _ = ax.pie([prediction_percentage, 100 - prediction_percentage], startangle=90, colors=[color, "lightgrey"], wedgeprops=dict(width=0.3))
    plt.title(prediction_label)
    st.pyplot(fig)

    st.write("### Summary")
    st.write(summary)

    st.write("### FEATURE ANALYSIS")
    analysis_table = [line.split(": ") for line in feature_analysis.split("\n") if ": " in line]
    feature_df = pd.DataFrame(analysis_table, columns=["Feature", "Relationship & Reason"])
    st.table(feature_df)

elif option == "Company Prediction":
    selected_company = st.sidebar.selectbox("Select Company", filtered_df['Company Name'].unique())
    company_df = filtered_df[filtered_df['Company Name'] == selected_company]
    departments = ["All"] + list(company_df['Department'].unique())
    selected_department = st.sidebar.selectbox("Select Department", departments)

    if selected_department == "All":
        department_df = company_df
    else:
        department_df = company_df[company_df['Department'] == selected_department]

    prediction_percentages = []
    names = []
    for _, row in department_df.iterrows():
        row = transform_dataframe(employee_data)
        #full_name = row['full_name']
        #work_status = row['Work Status']
        details = row['details']
        prediction_percentage, _, _, _, _ = few_shot_prediction(details)
        prediction_percentages.append(prediction_percentage)
        names.append(row['full_name'])

    avg_prediction = np.mean(prediction_percentages)
    st.markdown(f"**{selected_department} - {selected_company}**")
    st.write(f"### Average Churn Prediction: {avg_prediction:.2f}%")

    # Bar Chart
    chart_data = pd.DataFrame({"Employee Name": names, "Churn Prediction": prediction_percentages})
    chart_data = chart_data.sort_values(by="Churn Prediction", ascending=False)
    st.bar_chart(chart_data.set_index("Employee Name"))

    # Clickable Names
    selected_name = st.selectbox("View Individual Prediction", names)
    employee_data = department_df[department_df['full_name'] == selected_name].iloc[0]
    row = transform_dataframe(employee_data)
    #full_name = row['full_name']
    #work_status = row['Work Status']
    details = row['details']
    #details = employee_data.to_dict()
    prediction_percentage, prediction_label, summary, feature_analysis, color = few_shot_prediction(details)

    # Display Results
    st.markdown(f"**{selected_name}**")
    st.write(f"Company: {employee_data['Company Name']} / Department: {employee_data['Department']} / Status: {employee_data['Work Status']}")
    st.write(f"### Prediction: {prediction_percentage}%")
    fig, ax = plt.subplots()
    wedges, _ = ax.pie([prediction_percentage, 100 - prediction_percentage], startangle=90, colors=[color, "lightgrey"], wedgeprops=dict(width=0.3))
    plt.title(prediction_label)
    st.pyplot(fig)

    st.write("### Summary")
    st.write(summary)

    st.write("### FEATURE ANALYSIS")
    analysis_table = [line.split(": ") for line in feature_analysis.split("\n") if ": " in line]
    feature_df = pd.DataFrame(analysis_table, columns=["Feature", "Relationship & Reason"])
    st.table(feature_df)
