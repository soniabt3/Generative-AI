import openai
import ast
import re
import pandas as pd
import json


def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''
    
    delimiter = "####"
    example_user_req = {'House Type': 'Apartment','Availability': 'Yes','Location': 'Whitefield','Bedrooms': '2','Carpet Area': '1500','Budget': '15000000'}
    
    system_message = f"""

    You are an intelligent real estate agent in Bangalore and your goal is to find the best house for a user.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('House Type','Availability','Location','Bedrooms','Carpet Area','Budget') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this {{'House Type': 'values','Availability': 'values','Location': 'values','Bedrooms': 'values','Carpet Area': 'values','Budget': 'values'}}
    The value for 'bedrooms', 'carpet area' and 'budget' should be a numerical value extracted from the user's response. 
    The values currently in the dictionary are only representative values. 
    
    {delimiter}Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
    - The value of 'House Type' should be either 'apartment' or 'stand alone house' based on the requirement of the user.
    - The value of 'Availability' should either 'Yes' or 'No' based on the requirement of the user.
    - The value of 'Location' should be as mentioned by the user.
    - The value for 'Bedrooms', 'Carpet Area' and 'Budget' should be a numerical value extracted from the user's response.
    - 'Budget' value needs to be greater than or equal to 1500000 INR. If the user says less than that, please mention that there are no houses in that range.
    - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    {delimiter} Thought 1: Ask a question to understand the user's profile and requirements. \n
    Keep asking relevant questions to comprehend their needs.
    You are trying to fill the values of all the keys ('House Type','Availability','Location','Bedrooms','Carpet Area','Budget') in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys. 
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, proceed to the next step. Otherwise, rephrase the question to capture their profile. \n{delimiter}

    {delimiter}Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step. 
    Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same. 
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

    {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary. 
    If you are not confident about any of the values, ask clarifying questions. {delimiter}

    Follow the above chain of thoughts and only output the final updated python dictionary. \n


    {delimiter} Here is a sample conversation between the user and assistant:
    User: "Hi, I am looking for an apartment."
    Assistant: "Great! We have a lot of apartment options for you. May we know how many bedrooms would you require?"
    User: "I am looking for a 2 bhk"
    Assistant: "Do you have a minimum carpet size requirement in your 2 bhk?"
    User: "I would like it to be at least 1500 sq ft"
    Assistant: "Thank you for providing that information. What area would you be prefer us finding you an apartment in?"
    User: "Whitefield"
    Assistant: "Thank you for the information. Would you like to move in soon or are you willing to wait for a project that is under development?"
    User: "Would like to move in immediately"
    Assistant:"Could you kindly let me know your budget for the house? This will help me find options that fit within your price range while meeting the specified requirements."
    User: "my max budget is 80 lakhs inr"
    Assistant: "{example_user_req}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements. Do not start with Assistant: "
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation



def get_chat_model_completions(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
        max_tokens = 300
    )
    return response.choices[0].message["content"]



def moderation_check(user_input):
    response = openai.Moderation.create(input=user_input)
    moderation_output = response["results"][0]
    if moderation_output["flagged"] == True:
        return "Flagged"
    else:
        return "Not Flagged"


    
def intent_confirmation_layer(response_assistant):
    delimiter = "####"
    prompt = f"""
    You are an experienced real estate agent who has an eye for detail.
    You are provided an input. You need to evaluate if the input has the following keys: 'House Type','Availability','Location','Bedrooms','Carpet Area','Budget'
    Next you need to evaluate if the keys have the the values filled correctly.
    The value of 'House Type' should be either 'apartment' or 'stand alone house' based on the response of the user.
    The value of 'Availability' should either 'Yes' or 'No' based on the response of the user.
    The value of 'Location' should be as mentioned by the user.
    The value for 'Bedrooms', 'Carpet Area' and 'Budget' should be a numerical value extracted from the user's response.
    The value for the key 'budget' needs to contain a number with currency.
    Output a string 'Yes' if the input contains the dictionary with the values correctly filled for all keys.
    Otherwise out the string 'No'.

    Here is the input: {response_assistant}
    Only output a one-word string - Yes/No.
    """

    conversation = [{"role": "system", "content": prompt}]
    
    confirmation = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=conversation,
                                temperature=0, # this is the degree of randomness of the model's output
                                max_tokens = 300
                            )


    return confirmation.choices[0].message["content"]




def dictionary_present(response):
    delimiter = "####"
    user_req = {'House Type': 'Apartment','Availability': 'Yes','Location': 'Whitefield','Bedrooms': '2','Carpet Area': '1500','Budget': '15000000'}
    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract and return only the python dictionary from the input.
            The output should match the format as {user_req}.
            The output should contain the exact keys and values as present in the input.

            Here are some sample input output pairs for better understanding:
            {delimiter}
            input: - House Type: apartment - Availability: yes - Location: Whitefield - Bedrooms: 2 - Carpet Area: 1500 - Budget: 1,50,00,000 INR
            output: {{'House Type': 'Apartment','Availability': 'Yes','Location': 'Whitefield','Bedrooms': '2','Carpet Area': '1500','Budget': '15000000'}}

            input: {{'House Type':     'apartment', 'Availability':     'yes', 'Location':    'whitefield', 'Bedrooms': '2', 'Carpet Area': '1500', 'Budget': '1,50,00,000'}}
            output: {{'House Type': 'Apartment','Availability': 'Yes','Location': 'Whitefield','Bedrooms': '2','Carpet Area': '1500','Budget': '15000000'}}

            input: Here is your user profile 'House Type': 'Apartment','Availability': 'YES','Location': 'whitefield','Bedrooms': '2','Carpet Area': '1,500','Budget': '1.5 cr INR'
            output: {{'House Type': 'Apartment','Availability': 'Yes','Location': 'Whitefield','Bedrooms': '2','Carpet Area': '1500','Budget': '15000000'}}
            {delimiter}

            Here is the input {response}

            """
            
    conversation = [{"role": "system", "content": prompt}]
    
    response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=conversation,
                                temperature=0, # this is the degree of randomness of the model's output
                                max_tokens = 300
                            )
    
    return response.choices[0].message["content"]



def extract_dictionary_from_string(string):
    regex_pattern = r"\{[^{}]+\}"

    dictionary_matches = re.findall(regex_pattern, string)

    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        dictionary_string = dictionary_string.lower()

        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        dictionary = ast.literal_eval(dictionary_string)
    return dictionary




def compare_houses_with_user(user_req_string):
    house_df= pd.read_csv('blr_housing_data.csv')
    user_requirements = extract_dictionary_from_string(user_req_string)
    budget = int(user_requirements.get('budget', '0'))
    house_type = user_requirements.get('house type', '0').lower()
    

    filtered_houses = house_df.copy()
    filtered_houses['price'] = filtered_houses['price'].astype(int)
    filtered_houses = filtered_houses[filtered_houses['price'] <= budget].copy()
    filtered_houses = filtered_houses[filtered_houses['house_type'] == house_type.lower()].copy()
    
    # Create 'Score' column in the DataFrame and initialize to 0
    filtered_houses['Score'] = 0
    for index, row in filtered_houses.iterrows():
        score = 0
        
        if user_requirements.get('availability', '0').lower() == "yes" and row["availability"].lower() == "ready to move":
            score += 1
        elif user_requirements.get('availability', '0').lower() == "no":
            score += 1
            
        if row["location"].lower() == user_requirements.get('location', '0').lower():
            score += 2
            
        bedrooms = int(row['size'].split()[0])
        if bedrooms >= int(user_requirements.get('bedrooms', '0')):
            score += 1
            
        carpet_area = row['total_sqft']
        if carpet_area >= int(user_requirements.get('carpet area', '0')):
            score += 1

        filtered_houses.loc[index, 'Score'] = score

    # Sort the houses by score in descending order and return the top 5 matches
    top_houses = filtered_houses.sort_values('Score', ascending=False).head()
    # print(top_houses)
    return top_houses.to_json(orient='records')




def recommendation_validation(house_recommendation):
    data = json.loads(house_recommendation)
    data1 = []
    for i in range(len(data)):
        if data[i]['Score'] >= 2:
            data1.append(data[i])

    return data1




def initialize_conv_reco(house_matches):
    system_message = f"""
    You are an intelligent real estate broker in Bangalore, India and you are tasked with the objective to \
    find a suitable house for the client from this list: {house_matches}.\
    You should keep the user requirements in mind while answering the questions.\

    Start with a brief summary of each house in the following format, in decreasing order of price of houses and change line after each house detail:
    1. <Housing Society> : <Major specifications of the house>, <Price in Rs>, <Mention the Agent Name and Contact Number for the given property>
    2. <Housing Society> : <Major specifications of the house>, <Price in Rs>, <Mention the Agent Name and Contact Number for the given property>

    Please add a disclaimer in the end that every house may not contain all user specifications but we tried to provide you with the best matches among the available properties.
    """
    conversation = [{"role": "system", "content": system_message }]
    return conversation