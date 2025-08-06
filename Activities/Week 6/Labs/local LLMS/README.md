# **Instructions for the Lab**

**Objective**: In this lab, you will create a simple email writing assistant using Ollama API (via their Python SDK. The assistant will take input from the user, such as the subject of the email, recipient's name, and additional information, and generate a professional email using a large language model.

### Code Template

Below is the code template with key areas left as placeholders. Your goal is to complete these sections.

```python
from ollama import chat

model = 'mollm2:135m'

# Function to generate an email
def generate_email(subject, recipient_name, additional_info):
    # Step 1: Create the prompt to guide the AI in generating the email
    prompt = f"Write a professional email to {recipient_name} with the subject '{subject}'. Include the following information: {additional_info}"
    
    # Step 2: Call the OpenAI API to generate the email
    # Hint: Use the 'chat' endpoint to create a completion based on the prompt
    response = chat(
        model= #YOURCODE, 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    
    # Step 3: Extract and return the generated email content from the API response
    return #YOURCODE # <-- Complete this line to get the email content

# Function to start the email writing assistant
def email_writing_assistant():
    print("Welcome to the Email Writing Assistant!\n")
    
    # Step 4: Gather user input for the email subject, recipient name, and additional information
    subject = #YOURCODE
    recipient_name = #YOURCODE
    additional_info = #YOURCODE
    
    # Step 5: Call the 'generate_email' function and display the generated email
    email = generate_email(subject, recipient_name, additional_info)
    print(f"Email:\n{email}")

# Start the email writing assistant
email_writing_assistant()

```

### Prompt Challenge

Modify the template to include different email tones (e.g., formal, friendly, urgent). Add an extra input where the user can specify a tone, and adjust the prompt accordingly.

1. Update `email_writing_assistant` function to also capture tone. You can name the variable `input_tone`
2. Update `generate_email` function to take an additional parameter `input_tone`
3. Update the messages list to include the tone. Here is an example:

```python
messages=[
            {"role": "system", "content": f"You are a helpful assistant. Always reply in a {input_tone} tone "},
            {"role": "user", "content": prompt},
        ],


```

Test your the code.

### Prompt Bonus Challenge

Modify the prompt to include an additional input parameter `critical_date`

```python
f"Write a professional email to {recipient_name} with the subject '{subject}' and '{critical_date}'. Include the following information: {additional_info}"
```



