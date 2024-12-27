
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import scipy.stats as stats
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
# Permutation feature importance


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
from sklearn.compose import ColumnTransformer
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time 
import random


#---- Code for app ------------
st.set_page_config(layout='wide')


# Helper functions:
def calculate_multiclass_metrics(y_true, y_pred):
    """
    Calculate metrics for multiclass classification using string labels.

    Parameters:
    - y_true: Ground truth (actual labels).
    - y_pred: Predicted labels.

    Returns:
    - DataFrame with metrics calculated for each class.
    """
    # Ensure both inputs are lists (if not already)
    y_true = list(y_true)
    y_pred = list(y_pred)

    # Get unique class labels
    class_labels = sorted(set(y_true))  # All possible unique classes
    
    # Initialize a dictionary to store metrics
    metrics_data = {
        "Mental Health Status": [],
        "Sensitivity (Recall)": [],
        "Precision": [],
        "F1-Score": [],
        "Accuracy": [],
    }

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # Calculate metrics for each class
    for i, class_name in enumerate(class_labels):
        tp = cm[i, i]  # True Positives
        fn = cm[i, :].sum() - tp  # False Negatives
        fp = cm[:, i].sum() - tp  # False Positives
        tn = cm.sum() - (tp + fn + fp)  # True Negatives

        # Sensitivity (Recall)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Append metrics
        metrics_data["Mental Health Status"].append(class_name)
        metrics_data["Sensitivity (Recall)"].append(recall)
        metrics_data["Precision"].append(precision)
        metrics_data["F1-Score"].append(f1)
        metrics_data["Accuracy"].append(overall_accuracy)

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df


# Load the model:
def custom_transform(df):
  columns_to_drop = ['ID',
 'FEELING_NERVOUS',
 'TROUBLE_IN_CONCENTRATION',
 'HAVING_TROUBLE_IN_SLEEPING',
 'HAVING_TROUBLE_WITH_WORK',
 'HOPELESSNESS',
 'ANGER',
 'CHANGE_IN_EATING',
 'SUICIDAL_THOUGHT',
 'SOCIAL_MEDIA_ADDICTION',
 'MATERIAL_POSSESSIONS',
 'INTROVERT',
 'FEELING_NEGATIVE',
 'TROUBLE_CONCENTRATING',
 'BLAMMING_YOURSELF']
  #df = df.copy()
  return df.drop(columns_to_drop,axis=1, errors='ignore')

#num_col_ = df.select_dtypes(include=['int64','float64']).columns.columns.to_list() # uncomment if numerical features are present
cat_col_ = ['PANIC', 'AVOIDS_PEOPLE_OR_ACTIVITIES', 'BREATHING_RAPIDLY', 'SWEATING',
       'FEELING_TIRED', 'CLOSE_FRIEND', 'POPPING_UP_STRESSFUL_MEMORY',
       'OVER_REACT', 'HAVING_NIGHTMARES', 'WEIGHT_GAIN']


# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean'))  # Impute missing numeric values with mean
# ]) # uncomment if numerical features are present

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values with 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False,))  # One-hot encode categorical variables
])

# Create the custom transformer to drop columns
custom_transformer = FunctionTransformer(custom_transform)


# Create the full pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, cat_col_)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform='pandas')

model = RandomForestClassifier(n_estimators=400,
                                            random_state=42,
                                            class_weight='balanced',
                                            max_depth=3)

# Create the full pipeline

full_pipeline = Pipeline(steps=[
    ('custom_transform', custom_transformer),
    ('preprocessor', preprocessor),
    ('model',model)
])

# Split data
df_tr = pd.read_csv('/Users/krisghimire/Desktop/mental_health_strem_app/Data/train.csv')
df_tr.columns = [col.upper().replace('.','_').strip() for col in df_tr.columns]
x_Train,x_Test,y_Train,y_Test = train_test_split(df_tr.drop(columns=['DISORDER']),df_tr['DISORDER'],test_size=0.3,random_state=1)

# Load the data preprocessor and model
model_ = joblib.load("/Users/krisghimire/Desktop/mental_health_strem_app/mental_health_pred_model.pkl")

# Custom CSS for Helvetica font
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Helvetica:wght@300;400;700&display=swap');
    
#     html, body, [class*="css"]  {
#         font-family: 'Helvetica', sans-serif;
#     }

#     .centered-title {
#         font-size: 36px;
#         font-weight: bold;
#         text-align: center;
#     }
#     </style>
#     """, unsafe_allow_html=True)


# # Custom CSS for styling tabs and background
# st.markdown(
#     """
#     <style>
#     /* General styling for the app */
#     body {
#         background-color: #f0f2f6; /* Light gray background for the app */
#     }

#     /* Styling for tabs */
#     div[data-testid="stHorizontalBlock"] {
#         background: linear-gradient(to right, #4facfe, #00f2fe);
#         border-radius: 10px;
#         padding: 10px;
#         color: white;
#     }

#     div[data-testid="stHorizontalBlock"] > div {
#         font-size: 18px;
#         font-weight: bold;
#     }

#     /* Styling for individual tabs */
#     div[data-testid="stHorizontalBlock"] > div > div[role="tab"] {
#         background-color: #ffffff; /* White background for unselected tabs */
#         color: #333; /* Dark text color for unselected tabs */
#         padding: 10px 20px;
#         border-radius: 5px;
#         margin-right: 5px;
#         font-size: 16px;
#         cursor: pointer;
#     }

#     /* Styling for the active (selected) tab */
#     div[data-testid="stHorizontalBlock"] > div > div[aria-selected="true"] {
#         background-color: #1f77b4; /* Blue background for selected tab */
#         color: white; /* White text for selected tab */
#     }

#     /* Add some padding between the tab content */
#     section[tabindex="0"] {
#         padding: 20px;
#         border-radius: 10px;
#         background-color: white;
#         box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

col1,col2 = st.columns([2,1])
with col1:
    st.header('Know Your Mental Health Status')
    #st.markdown('<p class="centered-title">Know Your Mental Health Status</p>', unsafe_allow_html=True)
    st.write("""
             Machine learning has revolutionized the field of mental health by enabling the early detection 
             and prediction of mental health crises, thereby improving patient outcomes and reducing the burden 
             on healthcare systems. By analyzing electronic health records (EHRs), machine learning models can 
             continuously monitor patients and predict the risk of a mental health crisis up to 28 days in advance
             
             This Ai models can identify individuals at increased risk of various mental health disorders, 
             including depressive episodes, manic episodes, suicidal ideation, and anxiety crises, allowing
             for timely interventions and a shift from reactive to preventive care strategies 
             """)
with col2:
    #Display the image in the second column
    image = Image.open("/Users/krisghimire/Desktop/mental_health_strem_app/images/julienmh.jpg")
    st.image(image) #, use_container_width=True
 
# Add logo:
st.sidebar.image("/Users/krisghimire/Desktop/mental_health_strem_app/images/logo2.png", width=90)
#st.sidebar.write('Mental Health Ai')
st.sidebar.markdown('<p style="font-size:20px; font-weight:bold;">MansikAi - Mental Health Assistance</p>', unsafe_allow_html=True)

# with st.sidebar.container():
#     # Insert logo on the sidebar
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.image("/Users/krisghimire/Desktop/mental_health_strem_app/images/logo2.png", width=100)
#     with col2:
#         st.write("MansikAi: Mental Health Assistance")

    
#st.sidebar.header('Upload New Data To Make Prediction')
upload_file = st.sidebar.file_uploader('Upload .CSV File',type=['csv'])


# if st.sidebar.button('Clear'):
#     uploaded_file = None
#     st.experimental_rerun() # will rerun the script immediately.


# Create tabs
tabs = st.tabs(['About', 'Model Metrics', 'Bath Prediction', 'Real Time Prediction', 'Business Impact','Ask MAi'])
#st.dataframe(upload_file)
with tabs[0]:
    st.subheader('Current Discussion and Insight On Mental Health')
    # Put Line and Bar chart of mental Health status
    st.write()
    
    ## Add some charts and figures 
    
with tabs[1]:
    st.subheader('Ai Model Performance Metrics')
    y_pred = model_.predict(x_Test)
    y_pred_proba = model_.predict_proba(x_Test)
    y_scores = model_.predict_proba(x_Test)
    class_names= model_.classes_
    y_onehot = pd.get_dummies(y_Test, columns=class_names,dtype=int)
    #y_onehot
    col1,col2,col3 = st.columns(3)
    with col1:
        #st.write('Metric Table')
        mt_df = calculate_multiclass_metrics(y_Test, y_pred)
        mt_df

    with col2:
        #st.write('Confusion Metric')
        
        cm = confusion_matrix(y_Test, y_pred)
        cm = confusion_matrix(y_Test, y_pred)
        fig6 = px.imshow(cm,text_auto=True,
                        labels=dict(x="Model Predicted Status", y="Actual True Status", color="Productivity"),
                        x=['Anxiety', 'Depression', 'Loneliness', 'Normal', 'Stress'],
                        y=['Anxiety', 'Depression', 'Loneliness', 'Normal', 'Stress'],
                        color_continuous_scale=["#43aa8b", "#00a8e8"]
                        )
        fig6.update_coloraxes(showscale=False)
        fig6.update_layout(width=400, height=500,
                           
                           title={
                            'text': "Confusion Metric",
                            'y':0.9,# up and down
                            'x':0.5, # left and right
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font_size': 25,
                            'font_family': 'Arial'
                            }
                           
                           )
        st.plotly_chart(fig6)
                
    with col3:
        #st.write('AUC-ROC Curve')
        
        fig7 = go.Figure()
        fig7.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig7.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig7.update_layout(
            xaxis=dict(
                title=dict(
                    text='False Positive Rate'
                ),
                constrain='domain'
            ),
            yaxis=dict(
                title=dict(
                    text='True Positive Rate'
                ),
                scaleanchor='x',
                scaleratio=1
            ),
            width=450, height=500,
            
            
            title={
                            'text': "AUC-ROC Curve",
                            'y':0.9,# up and down
                            'x':0.5, # left and right
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font_size': 25,
                            'font_family': 'Arial'
                            }
            
        )
        st.plotly_chart(fig7)

    

with tabs[2]:
    st.subheader('Make Prediction On New Patients')
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        df.columns = [col.upper().replace('.','_').strip() for col in df.columns]
        st.write('Preview Of The New Patient Data')
        st.dataframe(df.head())
        
        # Ensure that the model is run only after clicking the button
        if st.button('Predict Mental Health Status'):
            try:
                #Predict using the loaded model
                class_names= model_.classes_
                prediction = model_.predict(df)
                prediction_prob = model_.predict_proba(df)
                prob_df = pd.DataFrame(prediction_prob,columns=class_names,index=df.index)
                pred_output = df.copy()
                pred_output_ = pd.concat([pred_output,prob_df],axis=1)
                pred_output_['Mental Status'] = pred_output_[class_names].idxmax(axis=1)
                pred_output_['Probability Score'] = pred_output_[class_names].max(axis=1)
                st.write('Prediction Report')
                st.dataframe(pred_output_.head(5))

                # --- PLOT BAR CHART
                # ---- Create two columns 
                col3,col4 = st.columns(2)
                with col3:
                    df_status = pred_output_.groupby(by='Mental Status')['ID'].count().reset_index()
                    fig1 = px.bar(df_status,
                      x='Mental Status',
                      y='ID',
                     text='ID',
                     title="Mental Health Status Counts",
                     labels={'Mental Status': 'Status', 'ID':'Number Of Patients'},
                     color='Mental Status',
                     color_discrete_sequence=["#55dde0", "#f26419", "#2f4858", "#43aa8b", "#00a8e8"],
                     template= 'plotly_white')
                    fig1.update_layout(
                            title={
                            'text': "Distribution Of Predicted Mental Health Status",
                            'y':0.9,# up and down
                            'x':0.5, # left and right
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font_size': 25,
                            'font_family': 'Arial'
                            },
                            xaxis_title="Mental Status",
                            yaxis_title="Count of Patients",
                            plot_bgcolor="white",
                            font=dict(
                            family="Arial, sans-serif",
                            color="black"),
                            xaxis=dict(tickfont=dict(size=17)),
                            yaxis=dict(tickfont=dict(size=17)),)
                    st.plotly_chart(fig1)   
                with col4: 
                    df_status['IS_NORM'] = df_status['Mental Status'].apply(lambda x: 'Normal' if x == 'Normal' else 'Mental Issues')
                    df_is_norm = df_status.groupby('IS_NORM')['ID'].sum().reset_index()
                    
                    fig2 = px.pie(
                                  df_is_norm,
                                  names='IS_NORM',
                                  values='ID',
                                  title='Distribution Of Patients Normal vs Illesnn',
                                  hole=0.4,
                                  color='IS_NORM',
                                  color_discrete_map={'Normal':'green','Mental Issues':'red'})
                    # Customize the Layout
                    fig2.update_layout(
                        title=dict(
                            text='Distribution of Patients: Normal vs Mental Issues',
                            x=0.5,
                            xanchor='center',
                            font=dict(size=25, family='Arial')),
                        font=dict(size=15, family='Arial'),
                        plot_bgcolor='white') 
                    st.plotly_chart(fig2)
                
            except Exception as e:
               st.error(f'An errr occured while making prediction{str(e)}')
               
        


with tabs[3]:
    #st.subheader('Real Time Prediction Of Mental Status')
    #--- Write code to predict real time prediction
    # Factors to ask as input questions
    input_factors = ['PANIC', 'AVOIDS_PEOPLE_OR_ACTIVITIES', 'BREATHING_RAPIDLY',
                 'SWEATING', 'FEELING_TIRED', 'CLOSE_FRIEND',
                 'POPPING_UP_STRESSFUL_MEMORY', 'OVER_REACT',
                 'HAVING_NIGHTMARES', 'WEIGHT_GAIN']
    st.title("Make Prediction Of Your Status")
    # Function to create Yes/No buttons with colors
    # Explicit Questions
    # Organize into 2 columns and 5 rows
    col1, col2 = st.columns(2)

    with col1:
        panic = st.radio("Do you tend to panic often?", options=['yes', 'no'], horizontal=True)
        breathing_rapidly = st.radio("Are you breathing rapidly?", options=['yes', 'no'], horizontal=True)
        feeling_tired = st.radio("Are you feeling tired frequently?", options=['yes', 'no'], horizontal=True)
        stressful_memory = st.radio("Do you continuously feel like popping up stressful memories?", options=['yes', 'no'], horizontal=True)
        bad_nightmares = st.radio("Do you often have bad nightmares?", options=['yes', 'no'], horizontal=True)

    with col2:
        avoids_people = st.radio("Do you tend to avoid people or activities?", options=['yes', 'no'], horizontal=True)
        sweating = st.radio("Are you feeling sweaty?", options=['yes', 'no'], horizontal=True)
        close_friend = st.radio("Do you have a close friend that you talk to often?", options=['yes', 'no'], horizontal=True)
        over_react = st.radio("Do you tend to overreact to things?", options=['yes', 'no'], horizontal=True)
        weight_gain = st.radio("Do you feel weight gain?", options=['yes', 'no'], horizontal=True)


    # Collect Inputs into a list
    user_responses = [
        panic, avoids_people, breathing_rapidly, sweating, feeling_tired,
        close_friend, stressful_memory, over_react, bad_nightmares, weight_gain
    ]
    user_resp_dic = {factor:response for factor,response in zip(input_factors,user_responses)}
    input_user_df = pd.DataFrame([user_resp_dic])
    #input_user_df

    #Submit button
    if st.button("Submit"):
        try:
            with st.spinner('Predicting mental health status...'):
                time.sleep(3)
                # Predict using the loaded model
                prediction_ur = model_.predict(input_user_df)
                prediction_proba_ur = model_.predict_proba(input_user_df)
                prediction_proba_ur = prediction_proba_ur.flatten()
                
                # Decode prediction label
                predicted_label = prediction_ur[0]  # Assuming the model returns the label directly
                predicted_prob = np.max(prediction_proba_ur)  # Get the highest probability
                #st.write(prediction_ur)
                #st.write(prediction_proba_ur)
                class_names = model_.classes_
                #st.write(class_names)

                # Display Results
                st.success(f"**Predicted Mental Health Status:** {predicted_label}")
                st.info(f"**Prediction Probability Score:** {predicted_prob:.2f}")

                # Display probability for all classes
                
                df_pred_classes_prob = pd.DataFrame({'Class': class_names, 'Probability':prediction_proba_ur})
                df_pred_classes_prob = df_pred_classes_prob.sort_values(by='Probability', ascending=False)
                # Colors for the 5 classes
                colors_ = ["#55dde0", "#f26419", "#2f4858", "#43aa8b", "#00a8e8"]
                
                # Highlight the top prediction
                top_prediction = df_pred_classes_prob.loc[df_pred_classes_prob['Probability'].idxmax(), 'Class']
                
                # Plot Horizontal Bar Chart
                fig3 = px.bar(
                    df_pred_classes_prob,
                    x='Probability',
                    y='Class',
                    orientation='h',
                    text=df_pred_classes_prob['Probability'].apply(lambda x: f"{x:.1%}"),  # Annotate with percentages
                    title=f"Predicted Mental Health Status : ({top_prediction})",
                    labels={'Class': 'Predicted Class', 'Probability': 'Predicted Probability'},
                    color='Class',
                    color_discrete_sequence=colors_  # Custom color scheme
                )
                # Update Layout
                fig3.update_layout(
                    title=dict(
                        x=0.2,  # Centered title
                        font=dict(size=16, family="Arial")
                    ),
                    xaxis=dict(title="Probability", tickformat=".0%", showgrid=True),
                    yaxis=dict(title="Predicted Class"),
                    plot_bgcolor='white',
                    height=400,
                    width=600,
                    legend=dict(
                    x=1.1,  # Move legend to the right
                    y=0.5,  # Center legend vertically
                    xanchor='left',
                    yanchor='middle',
                    font=dict(size=12)
        )
                )
                # Add text annotations for clarity
                fig3.update_traces(textposition='inside')
                

                st.plotly_chart(fig3)
            
        except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    
with tabs[4]:
    df_metric = pd.read_csv('/Users/krisghimire/Desktop/mental_health_strem_app/Data/prediction_cost_savings.csv')
    df_metric['Date'] = pd.to_datetime(df_metric['Date']) 
    df_metric['Short_Date'] = df_metric['Date'].dt.strftime('%m-%y')  # Format to MM-YY
    st.subheader('Model ROI Metrics')
    # Change the below code later
    col1, col2, col3,col4,col5,col6 = st.columns(6)
    col1.metric("Patient Encountered", "522", "12")
    col2.metric("Predicted Positive", "322", "+18%")
    col3.metric("Care Provided", "16%", "4%")
    col4.metric('Cost Saved', '$23,342','$12000')
    col5.metric('Throughput','40%','13%')
    col6.metric('Time Saved','25 Min/Patient','15 Min')
    
    col5,col6 = st.columns(2)
    with col5:
        # Ensure 'Date' column is already in the format "YYYY-MM-DD"
        #df_metric['Short_Date'] = df_metric['Date'].dt.strftime('%m-%y')  # Format to MM-YY

        # Generate your figure
        fig4 = px.line(df_metric,
                    x='Short_Date', y='Prediction_Count',
                    labels={'Short_Date': 'Prediction Month', 'Prediction_Count': 'Total Predictions'},
                    title='Prediction Made Over The Months',
                    height=500,
                    width=600,
                    markers=True,
                    template='plotly_white'
                    )

        # Update layout to ensure all short month-year values appear on the x-axis
        fig4.update_layout(
            title={
                'text': "Prediction Made Over The Months <br><sub>2023 - 2024</sub>",
                'y': 0.9,
                'x': 0.5,
                "font": dict(family='Arial', size=20, color='gray'),
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                tickmode='array',  # Show all ticks
                tickvals=df_metric['Short_Date'],  # Use the formatted month-year column
                ticktext=df_metric['Short_Date'],  # Display short month-year values
                title='Month-Year',
                tickangle=45  # Rotate x-axis labels for better visibility
            ),
            yaxis=dict(
                title="Total Predictions"
            )
        )
        st.plotly_chart(fig4)
    with col6:
        
        # Generate your figure
        fig5 = px.line(df_metric,
                    x='Short_Date', y='Cost_Saving',
                    labels={'Short_Date': 'Prediction Month', 'Cost_Saving': 'Cost Saving'},
                    title='Cost Saving Made Over The Months',
                    height=500,
                    width=600,
                    
                    markers=True,
                    template='plotly_white'
                    )

        # Update layout to ensure all short month-year values appear on the x-axis
        fig5.update_layout(
            title={
                'text': "Cost Saving Made Over The Months <br><sub>2023 - 2024</sub>",
                'y': 0.9,
                'x': 0.5,
                "font": dict(family='Arial', size=20, color='gray'),
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                tickmode='array',  # Show all ticks
                tickvals=df_metric['Short_Date'],  # Use the formatted month-year column
                ticktext=df_metric['Short_Date'],  # Display short month-year values
                title='Month-Year',
                tickangle=45  # Rotate x-axis labels for better visibility
            ),
            yaxis=dict(
                title="Cost $"
            )
        )
        fig5.update_traces(line=dict(color='green'))
        fig5.update_traces(marker=dict(size=9))
        st.plotly_chart(fig5) 


with tabs[5]:
    st.header('Ask Me Mental Health Related Question')
    #st.chat_input()
    # prompt = st.chat_input("Ask me about mental health ..")
    # if prompt:
    #     st.write(f"User has sent the following prompt: {prompt}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
    # Streamed response emulator
    def response_generator():
        response = random.choice(
            [
                "Hello there! How can I assist you today?",
                "Hi, human! Is there anything I can help you with?",
                "Do you need help?",
            ]
        )
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})




#--------------------------------------------------------------------------------------------------------------
#-------------------------------------------- ----- Model code - ------------
#--------------------------------------------------------------------------------------------------------------


#full_pipeline




# sample_data = {
#     'PANIC': 'yes',
#     'AVOIDS_PEOPLE_OR_ACTIVITIES': 'no',
#     'BREATHING_RAPIDLY': 'no',
#     'SWEATING': 'yes',
#     'FEELING_TIRED': 'yes',
#     'CLOSE_FRIEND': 'no',
#     'POPPING_UP_STRESSFUL_MEMORY': 'yes',
#     'OVER_REACT': 'no',
#     'HAVING_NIGHTMARES': 'yes',
#     'WEIGHT_GAIN': 'no'
# }
# sample_df = pd.DataFrame([sample_data])

# pred = model.predict(sample_df)
# pred_prob = model.predict_proba(sample_df)
# print(pred)
# print('-------------')
# print(pred_prob)
