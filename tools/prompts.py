import json

with open('prompts.json', 'r') as file:
    prompts = json.load(file)

guideline_prompt = prompts["guideline_prompt"]
which_type_prompt = prompts["which_type_prompt"]
pre_prompt = prompts["pre_prompt"]
pos_prompt = prompts["pos_prompt"]
bi_prompt  = prompts["bi_prompt"]
mul_prompt = prompts["mul_prompt"]
hm_prompt  = prompts["hm_prompt"]




# guideline_prompt = "You are my assistant. Your task is to carefully read and strictly follow the instructions I provide to answer my questions. My input consists of two parts:\n 1. Instructions: This section outlines specific guidelines for responding to my questions.\n 2. Question: This section contains the question I need you to answer.\n You must adhere strictly to the guidelines in the Instructions section without any deviation. Ensure your responses are accurate, relevant, and aligned with the provided instructions."
# which_type_prompt = "You are tasked with classifying the question I provide into one of three categories, regardless of any missing or unclear information. The categories are:\n 1. Yes/No question: Output \"1\" if the question requires a yes or no answer.\n 2. How many question: Output \"2\" if the question asks for a numerical quantity or count.\n 3. Multiple choice question: Output \"3\" if the question provides multiple answer options.\n Strictly follow these guidelines and output only the corresponding number (1, 2, or 3) based on the question's type."
# pre_prompt = "Before proceeding to the question, I will provide you with premises, which are statements expressed in natural language. Each premise is prefixed with an index, for example:\n Premise 1: If a Python code is well-tested, then the project is optimized.\n Premise 2: If a Python code does not follow PEP 8 standards, then it is not well-tested.\n You must answer the question based on these premises, strictly adhering to them. You must provide the response in the following JSON format: {\"answer\": \"your answer here\", \"idx\": [list of premise indexes used], \"explanation\": \"your explanation based on the premises\"}. In which:\n"
# pos_prompt = "\"explanation\": Provide a clear and concise explanation for your answer, detailing the reasoning process. You must reference the premises used by their indexes (e.g., Premise 1, Premise 2) in the explanation. Ensure all referenced premises are clearly identified by their indexes.\n \"idx\": A list of integers (e.g., [1, 2]) representing the indexes of the premises used in the explanation. Ensure strict consistency: every index in idx must correspond to a premise referenced in the explanation, and every premise referenced in the explanation must have its index included in idx.\n"
# bi_prompt  = "{}\"answer\": Because this is a Yes/No/Uncertain question, you should output one of the following based on the given premises:\n 1. \"Yes\" if the question can be proven true.\n 2. \"No\" if the question can be proven false.\n 3. \"Uncertain\" if the question cannot be proven true or false, meaning no conclusion can be drawn.\n{}".format(pre_prompt, pos_prompt)
# mul_prompt = "{}\"answer\": Because this is an multiple-choice question, you should output the correct option (e.g., \"A\", \"B\", \"C\", \"D\") based on the question and the given premises.\n{}".format(pre_prompt, pos_prompt)
# hm_prompt  = "{}\"answer\": Because this is a how many question, you need to output a single integer based on the question and the given premises.\n{}".format(pre_prompt, pos_prompt)
