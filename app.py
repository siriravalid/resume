import streamlit as st

# Function to display the resume
def display_resume():
    # Set page configuration
    st.set_page_config(
        page_title="Siri Ravali's Resume",
        page_icon=":bookmark:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Define colors
    primary_color = "#2C3E50"  # Dark Blue
    accent_color = "#9B59B6"   # Purple
    background_color = "#ECF0F1"  # Light Gray
    text_color = "#2C3E50"     # Dark Blue

    # Apply custom CSS
    st.markdown(f"""
        <style>
            .stApp {{
                background-color: {background_color};
            }}
            .title {{
                color: {primary_color};
                font-size: 36px;
                font-weight: bold;
            }}
            .header {{
                color: {accent_color};
                font-size: 24px;
                font-weight: bold;
            }}
            .subheader {{
                color: {primary_color};
                font-size: 20px;
                font-weight: bold;
            }}
            .content {{
                color: {text_color};
                font-size: 16px;
            }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title">Siri Ravali\'s Resume</p>', unsafe_allow_html=True)

    st.markdown('<p class="header">Contact Information</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">Email: <a href="mailto:siri.ravali31@gmail.com" style="color: #3498DB;">siri.ravali31@gmail.com</a></p>', unsafe_allow_html=True)
    st.markdown('<p class="content">Phone: <a href="tel:+14753018354" style="color: #3498DB;">+1 4753018354</a></p>', unsafe_allow_html=True)
    st.markdown('<p class="content">LinkedIn: <a href="https://www.linkedin.com/in/siriravali/" style="color: #3498DB;">LinkedIn Profile</a></p>', unsafe_allow_html=True)

    st.markdown('<p class="header">Skills</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Machine Learning and Deep Learning Frameworks and Libraries</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Generative Adversarial Networks (GANs), Transformer Networks, TensorFlow, PyTorch, Keras, Scikit-learn, XGBoost, Streamlit, Hugging Face</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="subheader">Development Tools and Platforms</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Heroku, FastAPI, Git, NumPy, Pandas, Matplotlib</p>', unsafe_allow_html=True)

    st.markdown('<p class="subheader">Machine Learning Algorithms</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Linear Regression, Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Naive Bayes, K-Means Clustering, Gradient Boosting Machines (GBM), Neural Networks</p>', unsafe_allow_html=True)

    st.markdown('<p class="subheader">Generative AI Tools</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• GPT-3.5 Turbo, GPT-4, Ollama, Gemini Pro, Generative Adversarial Networks (GAN)</p>', unsafe_allow_html=True)

    st.markdown('<p class="subheader">Cloud Services</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Azure (Azure Machine Learning, Azure Databricks), AWS (AWS SageMaker, AWS Lambda), GCP (Google Cloud Machine Learning Engine, Google BigQuery)</p>', unsafe_allow_html=True)

    st.markdown('<p class="subheader">Technical Skills</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Cloud Computing, CI/CD, Unit Testing, OOP, Distributed Systems, Backend, Full-Stack</p>', unsafe_allow_html=True)

    st.markdown('<p class="header">Experience</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Accenture.co (Nov 2016 - Nov 2019)</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Designed and deployed deep learning models using TensorFlow and PyTorch for image and text processing tasks, achieving a 15% improvement in model accuracy.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Developed and maintained scalable APIs using FastAPI and Heroku for serving machine learning models in production environments.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Implemented end-to-end machine learning pipelines including data preprocessing, feature engineering, and model evaluation using Scikit-learn and XGBoost.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Created interactive web applications using Streamlit to visualize model predictions and analysis results for stakeholders.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Built predictive models using algorithms such as Random Forest, SVM, and Gradient Boosting Machines (GBM) for business intelligence and decision support.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Conducted data analysis and visualization with Pandas and Matplotlib to derive actionable insights from large datasets.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Collaborated with cross-functional teams to integrate machine learning solutions into existing systems, enhancing data-driven decision-making processes.</p>', unsafe_allow_html=True)

    st.markdown('<p class="subheader">Swanktek (Dec 2019 - July 2022)</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Designed and deployed deep learning models using TensorFlow and PyTorch for image and text processing tasks, achieving a 15% improvement in model accuracy.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Developed and maintained scalable APIs using FastAPI and Heroku for serving machine learning models in production environments.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Implemented end-to-end machine learning pipelines including data preprocessing, feature engineering, and model evaluation using Scikit-learn and XGBoost.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Created interactive web applications using Streamlit to visualize model predictions and analysis results for stakeholders.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Built predictive models using algorithms such as Random Forest, SVM, and Gradient Boosting Machines (GBM) for business intelligence and decision support.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Conducted data analysis and visualization with Pandas and Matplotlib to derive actionable insights from large datasets.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Collaborated with cross-functional teams to integrate machine learning solutions into existing systems, enhancing data-driven decision-making processes.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Leveraged AWS and Azure cloud services to deploy and scale machine learning models efficiently.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Implemented CI/CD pipelines for automated deployment and monitoring of machine learning models using Azure DevOps.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Utilized GCP\'s BigQuery for handling large-scale data analysis and processing tasks.</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Integrated machine learning solutions with cloud-based databases and storage solutions for seamless data access and retrieval.</p>', unsafe_allow_html=True)

    st.markdown('<p class="header">Education</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• B.Tech from Chaitanya Bharathi Institute of Technology (2012 - 2016)</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Master’s from University of New Haven (August 2022 - May 2024)</p>', unsafe_allow_html=True)

    st.markdown('<p class="header">Projects</p>', unsafe_allow_html=True)
    st.markdown('<p class="content">• Fake News Detector - Deployed as a public URL: <a href="https://workingsiri-db4cl99taypqbnfvtnwtxa.streamlit.app/" style="color: #3498DB;">Fake News Detector</a></p>', unsafe_allow_html=True)

if __name__ == "__main__":
    display_resume()
