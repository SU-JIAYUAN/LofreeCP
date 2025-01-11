import torch
from tensor_parallel import TensorParallelPreTrainedModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from collections import defaultdict

# Device configuration
device_ids = [1, 2, 4]
devices = [f"cuda:{i}" for i in device_ids]


# Set how many samples (calibration + val + test) you want to generate
# We do not divide samples into different sets in this script.
num_samples = 2000 

num_shots = 32
sampling_num = 20

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_name)
model = TensorParallelPreTrainedModel(model, devices)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Load dataset and filter
dataset = load_dataset("trivia_qa.py", 'rc')['train']
filtered_data = [(ex['question'], ex['answer']['value']) for ex in dataset if len(str(ex['answer']['value']).split()) <= 10]
questions, answers = zip(*filtered_data)

# Generate few-shot prompt
def few_shot(num_shots):
    return "Please answer the following questions.\n" + "\n".join(
        f"{questions[i]}\n{answers[i]}" for i in range(num_shots)
    )

# Locate response positions
def answer_pos_locate(question, text):
    start = text.index(question)
    end = start + len(question) - 1
    newline = text.find('\n', end + 2)
    newline = newline if newline != -1 else text.find('</s>', end + 2)
    return [end, newline]

# Convert list of lists to frequency dictionaries
def list_of_lists_to_frequency_dicts(list_of_lists):
    return [
        dict(sorted(defaultdict(int, ((e, sub_list.count(e)) for e in sub_list)).items(), key=lambda x: x[1], reverse=True))
        for sub_list in list_of_lists
    ]

# Generate text
def generate_text(prompt, max_new_tokens=10, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(devices[0])
    outputs = model.generate(
        inputs.input_ids,
        do_sample=True,
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )
    scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    return outputs, scores

# Generate responses
prompt = few_shot(num_shots)
generated_responses = []

for i in range(num_samples):
    full_prompt = prompt + questions[i + num_shots]
    response_dict = defaultdict(int)

    for _ in range(sampling_num):
        response, _ = generate_text(full_prompt)
        text = tokenizer.decode(response.sequences[0])
        end_pos, newline_pos = answer_pos_locate(questions[i + num_shots], text)
        response_text = text[end_pos + 2:newline_pos].lower()
        response_dict[response_text] += 1

    print(dict(response_dict))
    generated_responses.append(dict(response_dict))

# Save results
with open("generation.txt", "w") as file:
    file.write(str(generated_responses) + "\n")

with open("answers.txt", "w") as file2:
    file2.write(str(answers[num_shots:num_samples + num_shots]) + "\n")
