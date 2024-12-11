import torch
from tensor_parallel import TensorParallelPreTrainedModel
import time
import math
from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer
from datasets import load_dataset
from collections import defaultdict
import json
import numpy as np
from gensim.test.utils import common_texts, datapath
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import pynvml
import torch


device_ids = [1,2,3]
device_name = [f"cuda:{device_ids[0]}", f"cuda:{device_ids[1]}", f"cuda:{device_ids[2]}"]

# For example, we set calibration and test size as 200.
calibration_size = 100
test_size = 200

# define the number of shot, sampling counts, alpha
kshot = 32
num_return_sequences = 1
sampling_num = 20
alpha = 0.4



model_name = "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_name)#, torch_dtype=torch.float16)
model = TensorParallelPreTrainedModel(model, device_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
similarity_model = FastText(sentences=common_texts, vector_size=200, min_count=1)

dataset = load_dataset("trivia_qa.py",'rc')
train_dataset = dataset['train']


questions = []
answers = []
generated_answers = []
correct_answers = []



test_questions = []
test_answers = []
test_correct_answers =[]

divide_bar = 0
num = 0
for example in train_dataset:
    divide_bar += 1
    # For example, we get 20000 samples.
    if divide_bar < 20000:
        answer = str(example['answer']['value'])
        question = str(example['question'])
        words = answer.split()
        if len(words) <= 10:
            num += 1
            if num % 2 == 0:
                questions.append(question)
                answers.append(answer)
            else:
                test_questions.append(question)
                test_answers.append(answer) 


def few_shot(kshot):
    prompt = "Please answer the following questions.\n"
    for i in range(kshot):
        prompt += f"{questions[i]}\n{answers[i]}\n"
    return prompt



def answer_pos_locate(question, text):
    start_position = text.index(question)
    end_position = start_position + len(question) - 1
    newline_position = text.find('\n', end_position + 2)
    if newline_position == -1: 
        newline_position = text.find('</s>', end_position + 2)
    return [end_position, newline_position]


def list_of_lists_to_frequency_dicts(list_of_lists):
    '''
    Convert the list to sorted dics showing the frequencies of generated answers
    '''
    frequency_dicts = []
    for sub_list in list_of_lists:
        element_frequency = defaultdict(int)
        for element in sub_list:
            element_frequency[element] += 1
        sorted_frequency = dict(sorted(element_frequency.items(), key=lambda item: item[1], reverse=True))
        frequency_dicts.append(sorted_frequency)
    return frequency_dicts




def generate_text(
        prompt,
        do_sample = True,
        max_new_tokens = 10,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
    ):
    inputs = tokenizer(prompt, return_tensors="pt").to(device_name[0])
    attention_mask = inputs['attention_mask']

    outputs = model.generate(
        inputs.input_ids,
        do_sample=do_sample,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=return_dict_in_generate,
        output_scores=output_scores
    )

    transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)

    return outputs, transition_scores


prompt = few_shot(kshot)


generated_responses = []
for i in range(calibration_size):
    prompt_question = questions[i + kshot]
    prompt_question = prompt + prompt_question

    response_dict = {}

    for sampling_index in range(sampling_num):
        response, score = generate_text(prompt_question)
        text = tokenizer.decode(response.sequences[0])
        positions = answer_pos_locate(questions[i + kshot], text)
        end_position, newline_position = positions[0], positions[1]
        
        # Extract the response text
        response_text = text[end_position + 2:newline_position].lower()

        # Update the response count in the dictionary
        if response_text in response_dict:
            response_dict[response_text] += 1
        else:
            response_dict[response_text] = 1
    print(response_dict)
    generated_responses.append(response_dict)

with open("generation.txt", "w") as file:
    file.write(str(generated_responses))
    file.write("\n")

with open("answers.txt", "w") as file2:
    file2.write(str(answers[kshot:calibration_size+kshot]))
    file2.write("\n")


# test_generated_answers = []
# test_scores = []

# with open("generation_test_7b_32shot.txt", "a") as file:
#     for k in range(test_size):
#         test_generated_text = []
#         test_prompt_question = test_questions[k]
#         test_prompt_question = prompt + test_prompt_question

#         for sampling_index in range(sampling_num):
#             response, score = generate_text(prompt_question)
#             text = tokenizer.decode(response.sequences[0])
#             positions = answer_pos_locate(questions[i + kshot], text)
#             end_position, newline_position = positions[0], positions[1]
#             test_generated_text.append({'text': text[end_position + 2:newline_position].lower(), 'score': score})
#             print(k, {'text': text[end_position + 2:newline_position].lower(), 'score': score})
        
#         file.write(str(test_generated_text))
#         file.write("\n")
