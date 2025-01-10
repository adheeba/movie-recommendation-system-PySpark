# movie-recommendation-system-PySpark
Creating a movie recommendation system with pyspark that an user can interact with.
The tool will take as input the user Id information and recommend new shows in a user friendly manner.

## Usage
### Streamlit webApp Interface
The project has a web GUI Interface for normal users to use.
[![webapp](data\images\streamlit_app.jpeg)]
## Installation
Follow the section below to install it locally in your machine.
### Installing in local machine
Following are the steps to install this project.
1. Download the github repo to your machine using the below command
    ```bash 
    git clone git@github.com:adheeba/movie-recommendation-system-PySpark.git
    ```
2. create a virtual environemnt using the following commands
    ```bash 
    virtualenv env
    ```
3. Activate the virtual environment. The command to activate virtual environment varies with OS
    Below is the command to activate it in windows
    ```bash
    .\env\Scripts\activate
    ```
4. Inside the virtual environmnet use the `requirements.txt` file inside this project to install dependencies
    ```bash
    pip install -r requirements.txt
    ```
Now you have all the setup to run this project.
### Run the project locally
1. Navigate to streamlit folder and run the below command to start streamlit.
    ```bash
    streamlit run stream_lit.py
    ```
2. Run the below command to regenerate the model again.
    ```bash
    python src/MovieRecommendation.py
    ```
## Contributors
The project is contributed by Adheeba Thahsin
## Timeline
The time line for this project is 5 days.
