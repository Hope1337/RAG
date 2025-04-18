from .prompts import guideline_prompt, which_type_prompt

def generate_full_response(input_text, tokenizer, model):
    messages = [
        {"role": "system", "content": guideline_prompt},
        {"role": "user", "content": input_text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text)
    print()
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = model_inputs.input_ids

    outputs = model.generate(
        input_ids,
        max_new_tokens=5000,  
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True
    )

    # Decode toàn bộ chuỗi kết quả
    generated_sequence = outputs.sequences[0]
    full_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    
    # Chỉ lấy phần nội dung được sinh ra (bỏ phần input prompt)
    input_text_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    return full_text[input_text_len:].strip()

def create_prompt(instructions, question):
    return "1. Instructions:\n {}\n2. Question:\n {}".format(instructions, question)

def which_type(question, tokenizer, model):
    instructions = which_type_prompt
    text = create_prompt(instructions, question)
    return generate_full_response(text, tokenizer, model)