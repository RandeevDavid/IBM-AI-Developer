from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Stored in the form [input_1, output_1, input_2, output_2, …]
conversation_history = []

# The following shows the basic interaction with a chatbot. 
# The while loop that follows is the same idea except it allows for a conversation
"""

# Transformers library expects to recieve converstaion history with 
# each element seperated by a newline character (\n)
history_string = "\n".join(conversation_history)

input_text ="hello, how are you doing?"

# Tokenization of inputs (prompt and chat history)
inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
print("INPUTS : ", inputs)

# Vocab or vocabulary files contain a list of all the tokens that a model can recognize and process. They map each token 
# to a unique index, which helps the model convert text into numerical data.
tokenizer.pretrained_vocab_files_map # Used for accessing vocab files doesn't actually do anything in the code 

# Generate the output based on inputs
outputs = model.generate(**inputs)
print("OUTPUT : ", outputs)

# Decode the output to get the response in plaintext and not tokens (Detokenization or reconstruction)
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print("RESPONSE : ", response)

# Add the input and the response to the coversation history for next iteration
conversation_history.append(input_text)
conversation_history.append(response)
print("HISTORY", conversation_history)

"""

# Loop for running a conversation
while True:

    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)