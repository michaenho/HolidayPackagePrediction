# Holiday Package Prediction

###### Author: Michaen Ho

#

## Project Overview

"Trips & Travel.Com" company aims to expand its customer base by introducing a new Wellness Tourism Package. Wellness Tourism is defined as travel that allows the traveler to maintain, enhance, or kick-start a healthy lifestyle, and support or increase one's sense of well-being. The company plans to utilize existing customer and marketing data to efficiently target potential customers, reducing marketing costs and increasing the likelihood of package purchases.

## Instructions for Setting Up the Environment and Running the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/michaenho/HolidayPackagePrediction.git
   cd HolidayPackagePrediction
   
   ```

2. **Install Python Dependencies**:  
   This project uses Python 3.11+. It's recommended to set up a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the Project**:  
   Once dependencies are installed, you can execute the script and open up the prediction website using:
   ```bash
   cd src
   streamlit run main.py
   ```


## Instructions for Building and Running the Docker Container(s)

1. **Build the Docker Image**:  
   Ensure Docker is installed and running on your machine. Then, in the project root directory, build the Docker image:
   ```bash
   docker build -t streamlit-app .

   ```

2. **Run the Docker Container**:  
   Use the following command to run the container.
   ```bash
   docker run -d -p 8501:8501 --name streamlit-container streamlit-app

   ```

3. **Access the Application**:  
   Once the container is running, you can access the prediction webpage by opening a web browser and go to 'http://localhost:8501'.



## Holiday Package Prediction Tool

This Streamlit-based web application predicts whether a customer will purchase a holiday package based on various input features. Users provide details such as product type, marital status, age, designation, passport status, income, and the number of follow-ups. The tool then uses a trained machine learning model to predict if the customer will buy the package, displaying the result on the webpage.

![Plot](/Plots/Webpage.png)

For more details, please refer to the full [project wiki](https://github.com/michaenho/HolidayPackagePrediction/wiki).

