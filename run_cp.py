import torch
import time
import math
from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer
from datasets import load_dataset
from collections import defaultdict
import json
import string
import numpy as np
from gensim.test.utils import common_texts, datapath
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import ast
import random


best_av_size = 100
best_params = [100, 100]


# define the number of shot, sampling counts, alpha
kshot = 32
sampling_num = 20
predefined_error_rate = 0.3 # pre-defined error rate
results = []

similarity_model = FastText(sentences=common_texts, vector_size=200, min_count=1)

def list_of_lists_to_frequency_dicts(list_of_lists):
    frequency_dicts = []
    for sub_list in list_of_lists:
        element_frequency = defaultdict(int)
        for element in sub_list:
            element_frequency[element] += 1
        sorted_frequency = dict(sorted(element_frequency.items(), key=lambda item: item[1], reverse=True))
        frequency_dicts.append(sorted_frequency)
    return frequency_dicts




def new_CP_score(dict_of_freq, weight):
    dict_of_score = dict_of_freq.copy()
    total_frequency = sum(dict_of_freq.values())

    numerator = 0
    for key, value in dict_of_score.items():
        numerator += - value / total_frequency * math.log(value / total_frequency)
    if total_frequency == 1 or total_frequency == 0:
        total_frequency = 2
    normalized_entropy = numerator / math.log(total_frequency) 

    rank_1_response = ""
    for rank, (key, value) in enumerate(dict_of_score.items()):
        if rank == 1:
            rank_1_response = key
            dict_of_score[key] = 10 - value / total_frequency * 10 + normalized_entropy / 2 * weight
        else:
            dict_of_score[key] = 10 - value / total_frequency * 10 + normalized_entropy / 2 * weight
            dict_of_score[key] -= similarity_model.wv.similarity(key, rank_1_response) * weight_2


    return dict_of_score, normalized_entropy

def calculate_quantile_pos(n, alpha):
    result = np.ceil((n + 1) * (1 - alpha)) / n
    return result


def remove_punctuation(input_string):
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)

def remove_articles(input_string):
    articles = ['a', 'an', 'the']
    words = input_string.split()
    result = ' '.join(word for word in words if word.lower() not in articles)
    return result

def remove_duplicate_whitespace(input_string):
    return ' '.join(input_string.split())

def process_list_of_strings(input_list):
    return [remove_duplicate_whitespace(remove_articles(remove_punctuation(item.lower())))
            for item in input_list]

def process_list_of_dicts(input_list_of_dicts):
    processed_list_of_dicts = []

    for dictionary in input_list_of_dicts:
        processed_dict = {}
        for key, value in dictionary.items():
            processed_key = remove_duplicate_whitespace(remove_articles(remove_punctuation(key.lower())))

            if processed_key in processed_dict:
                processed_dict[processed_key] += value
            else:
                processed_dict[processed_key] = value

        processed_list_of_dicts.append(processed_dict)

    return processed_list_of_dicts

weights = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
weights_2 = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
for weight in weights:
    for weight_2 in weights_2:
        print(f"weight: {weight}, weight_2: {weight_2}")
        cali_answer = []
        test_answer = []
        correct_answers =[]
        generation = []


        with open('answers.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            correct_answers = ast.literal_eval(content)



        with open('generation.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            generation = ast.literal_eval(content)
            

        correct_answers = process_list_of_strings(correct_answers)
        generation = process_list_of_dicts(generation)


        combined_data = list(zip(correct_answers, generation))


        total_correct_sum = 0
        total_size_sum = 0
        total_val_size_sum = 0
        size_distribution_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        coverage_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        repeat_count = 50


        # for example, split_num = 3, which is used to divide samples into cali, val, test. 
        # You can also use other ways to divide samples.
        split_num = 3

        for i in range(repeat_count):
            random.shuffle(combined_data)
            correct_answers, generated_answers_freq = zip(*combined_data)
            nonconformity_scores = []


            test_set = []
            test_correct_answers = []
            val_set = []
            val_correct_answers = []
            num_cali = 0

            for index, dict_of_freq in enumerate(generated_answers_freq):
                if index % split_num == 0: # cali
                    result, NE = new_CP_score(dict_of_freq, weight)
                    num_cali += 1

                    if correct_answers[index] not in [answer for answer in list(generated_answers_freq[index].keys())]:
                        nonconformity_score = 20

                    else:                
                        nonconformity_score = result[correct_answers[index]]


                    nonconformity_scores.append(nonconformity_score)

                elif index % split_num == 1: # val
                    val_set.append(generated_answers_freq[index])
                    val_correct_answers.append(correct_answers[index])
                else: # test
                    test_set.append(generated_answers_freq[index])
                    test_correct_answers.append(correct_answers[index])




            quantile_pos = calculate_quantile_pos(num_cali, predefined_error_rate) * 100

            # print(calculate_quantile_pos(num_cali, predefined_error_rate), quantile_pos)

            sorted_nonconformity_scores = sorted(nonconformity_scores, reverse=True)
            # print(sorted_nonconformity_scores)

            quantile_value = np.percentile(sorted_nonconformity_scores, quantile_pos)

            predicted_answers_val = []
            predicted_answers = []


            for index, dict_of_freq in enumerate(val_set):
                result, NE = new_CP_score(dict_of_freq, weight)
                predicted_answer = {key: score for key, score in result.items() if score <= quantile_value}
                predicted_answers_val.append(list(predicted_answer.keys()))
            
            for index, dict_of_freq in enumerate(test_set):
                result, NE = new_CP_score(dict_of_freq, weight)
                predicted_answer = {key: score for key, score in result.items() if score <= quantile_value}
                predicted_answers.append(list(predicted_answer.keys()))

            total_val_size = 0
            total_size = 0
            total_val_size = 0
            size_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            conditional_coverage = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            total_question = 0
            total_val_question = 0
            total_correct = 0

            for k, sublist in enumerate(predicted_answers_val):
                val_sublist_len = len(sublist)
                total_val_question += 1
                total_val_size += val_sublist_len
            total_val_size_sum += total_val_size

            for k, sublist in enumerate(predicted_answers):
                sublist_len = len(sublist)
                total_question += 1
                if test_correct_answers[k] in sublist:
                    total_correct += 1
                    conditional_coverage[sublist_len] += 1

                total_size += len(sublist)

                size_distribution[len(sublist)] += 1

            total_size_sum += total_size
            total_correct_sum += total_correct
            for i in range(len(size_distribution)):
                size_distribution_sum[i] += size_distribution[i]
                coverage_sum[i] += conditional_coverage[i]

        print("BEGIN===============================================================")
        print("average size ", total_size_sum / repeat_count / total_question)
        print("distribution: ", [x / repeat_count for x in size_distribution_sum])
        print("conditional cov: ", [x / repeat_count for x in coverage_sum])
        print("accuracy: ", total_correct_sum / total_question / repeat_count)
        # print(total_correct_sum)
        # print(total_question)
        # print(repeat_count)
        print("OVER===============================================================")
        if total_val_size_sum / repeat_count / total_val_question < best_av_size:
            best_av_size = total_val_size_sum / repeat_count / total_val_question
            best_params[0] =  weight
            best_params[1] =  weight_2
        results.append([total_size_sum / repeat_count / total_question, total_correct_sum / total_question / repeat_count])

print(best_av_size)
print(best_params)
print(results)
