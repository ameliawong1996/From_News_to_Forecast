# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accuracy metric."""

import datasets
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np
import evaluate
import json
from pathlib import Path
from datetime import datetime

import re  # Import the regular expression module

# Function to extract only the numeric part of the string
def extract_numeric(value):
    match = re.match(r"([-+]?\d*\.?\d+)", value)
    return match.group(0) if match else None

def split_list_by_markers(input_list, start_marker=1, end_marker=2):
    # 保存结果的列表
    result = []
    # 临时列表用于存储当前分割的片段
    temp = []
    # 标记是否开始收集元素
    collecting = False

    for element in input_list:
        if element == start_marker:
            # 遇到起始标记，开始新的片段
            collecting = True
            temp = [element]
        elif collecting:
            # 如果正在收集，则添加到临时列表
            temp.append(element)
            if element == end_marker:
                # 遇到结束标记，结束这个片段的收集
                collecting = False
                result.append(temp)
                temp = []
    
    if temp:
        result.append(temp)

    # 返回结果
    return result

def split_list_by_number(lst, number, n):
    result = []
    i = 0
    # 遍历列表的索引和值
    while i < len(lst):
        # 如果找到了数字1
        if lst[i] == number:
            # 截取从当前元素开始的n个元素
            sub_list = lst[i:i+n+1]
            # 添加到结果中
            result.append(sub_list)
            # 移动索引到子列表的末尾
            i += n
        else:
            i += 1
    return result

def extract_numbers_and_turn_str(text):
    # 正则表达式匹配数字序列，包括小数点和逗号
    pattern = r'\d+\.\d+|\d+'
    # 找到所有匹配的数字
    numbers = re.findall(pattern, text)
    # 连接这些数字为一个字符串，用逗号分隔
    numbers_text = ','.join(numbers)
    return numbers_text


def remove_values_from_list(the_list, value):
    return [element for element in the_list if element != value]


def parse_instruction_input_output(text):
    # 使用正则表达式提取所需部分
    instruction_match = re.search(r'### Instruction:\n(.+?)\n\n###', text, re.DOTALL)
    input_match = re.search(r'### Input:\n(.+?)\n\n###', text, re.DOTALL)
    response_match = re.search(r'### Response:\n(.+)', text, re.DOTALL)

    # 创建字典
    result_dict = {
        "instruction": instruction_match.group(1).strip() if instruction_match else '',
        "input": input_match.group(1).strip() if input_match else '',
        "output": response_match.group(1).strip() if response_match else ''
    }

    return result_dict

def split_list_by_number(lst, number, n):
    result = []
    i = 0
    # 遍历列表的索引和值
    while i < len(lst):
        # 如果找到了数字1
        if lst[i] == number:
            # 截取从当前元素开始的n个元素
            sub_list = lst[i:i+n+1]
            # 添加到结果中
            result.append(sub_list)
            # 移动索引到子列表的末尾
            i += n
        else:
            i += 1
    return result

def append_to_json_file(file_path, new_data):
    # 检查文件是否存在并且不为空
    if Path(file_path).is_file() and Path(file_path).stat().st_size > 0:
        # 读取现有数据
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # 根据你的数据结构，这里可能需要自定义如何更新数据
            # 例如，如果数据是一个字典列表，可以这样追加新的字典
            if isinstance(data, list):
                data.append(new_data)
            elif isinstance(data, dict):
                # 这里需要知道如何更新字典，例如使用一个新的键
                key = new_data.get('key') or 'new_key'
                data[key] = new_data
            else:
                raise ValueError("JSON root must be either a list or a dictionary")
    else:
        # 如果文件不存在或为空，创建一个新列表或字典
        # 这里以列表为例
        data = [new_data]

    # 重新写入JSON文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



_DESCRIPTION = """
Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
 Where:
TP: True positive
TN: True negative
FP: False positive
FN: False negative
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
    sample_weight (`list` of `float`): Sample weights Defaults to None.

Returns:
    accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.

Examples:

    Example 1-A simple example
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
        >>> print(results)
        {'accuracy': 0.5}

    Example 2-The same as Example 1, except with `normalize` set to `False`.
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
        >>> print(results)
        {'accuracy': 3.0}

    Example 3-The same as Example 1, except with `sample_weight` set.
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
        >>> print(results)
        {'accuracy': 0.8778625954198473}
"""


_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Accuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(self, predictions, references, tokenizer, save_path='', normalize=True, sample_weight=None):
        try:
            if predictions[0] == -100:
                predictions, references = references, predictions

            predictions = remove_values_from_list(predictions, -100)
            references = remove_values_from_list(references, -100)
            references_slices = split_list_by_markers(references)
            predictions_slices = split_list_by_markers(predictions, 13291, 338)

            references_text_json = []
            predictions_text_json = []

            if len(references_slices) != len(predictions_slices):
                return {
                "MSE": float(0),
                "RMSE": float(0),
                "MAE": float(0),
                "MAPE": float(0),
                "R^2": float(0),
                "ExplainedVariance": float(0),
                "HitRate": float(0),
            }

            for i in range(len(references_slices)):
                references_text_json.append(parse_instruction_input_output(tokenizer.decode(references_slices[i])))

            for i in range(len(references_slices)):
                dict_temp = {'instruction': references_text_json[i]['instruction'],
                            'input': references_text_json[i]['input'],
                            'output': extract_numbers_and_turn_str(tokenizer.decode(predictions_slices[i]))}
                predictions_text_json.append(dict_temp)

            predicted_values = []
            actual_values = []
            print('download the results to ', './eval_intermid/' + f'eval-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json')
            with open('./eval_intermid/' + f'eval-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json', 'w', encoding='utf-8') as file:
                json.dump(predictions_text_json, file, ensure_ascii=False, indent=4)

            for pred, act in zip(predictions_text_json, references_text_json):
                # Use the extract_numeric function to only get numeric part
                pred_numbers = [float(extract_numeric(value)) for value in pred['output'].split(',') if extract_numeric(value)]
                act_numbers = [float(extract_numeric(value)) for value in act['output'].split(',') if extract_numeric(value)]
                
                if len(act_numbers) < len(pred_numbers):
                    pred_numbers = pred_numbers[:len(act_numbers)]
                elif len(act_numbers) > len(pred_numbers) and len(pred_numbers) > 0:
                    add_on = [pred_numbers[-1]] * (len(act_numbers) - len(pred_numbers))
                    pred_numbers = pred_numbers + add_on
                elif len(act_numbers) == len(pred_numbers):
                    pass
                else:
                    pred_numbers = [0.0] * len(act_numbers)

                predicted_values.extend(pred_numbers)
                actual_values.extend(act_numbers)

            mse = mean_squared_error(actual_values, predicted_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_values, predicted_values)
            mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
            r2 = r2_score(actual_values, predicted_values)
            explained_variance = explained_variance_score(actual_values, predicted_values)
            hits = np.sum(np.sign(np.array(actual_values[1:]) - np.array(actual_values[:-1])) == np.sign(np.array(predicted_values[1:]) - np.array(predicted_values[:-1])))
            hit_rate = hits / (len(actual_values) - 1)

            return {
                "MSE": float(mse),
                "RMSE": float(rmse),
                "MAE": float(mae),
                "MAPE": float(mape),
                "R^2": float(r2),
                "ExplainedVariance": float(explained_variance),
                "HitRate": float(hit_rate),
            }
        except:
            return {
                "MSE": float(0),
                "RMSE": float(0),
                "MAE": float(0),
                "MAPE": float(0),
                "R^2": float(0),
                "ExplainedVariance": float(0),
                "HitRate": float(0),
            }
