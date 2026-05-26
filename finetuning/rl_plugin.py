import asyncio
import re
from typing import List
import numpy as np
from collections import Counter
import logging
from openai import OpenAI
import json

from swift.llm import InferRequest
from swift.plugin import ORM, orms
from swift.llm import InferClient
from swift.llm import InferEngine, RequestConfig

import random


JUDGE_MODEL = "local"
MODEL_PATH = "PATH_TO_MODEL"

winrate_query = """
请你以一个教师的身份来判断下面两道题目的质量，请你综合考虑问题的设计、数值的设计、题目描述的流畅度做出判断。不要求生成的题目中包含题目答案或者解答提示信息。
请你公平的比较长度较长的题目和长度较短的题目，长度较长的题目不一定质量更好。
题目一: 
#QUESTION1#
题目二: 
#QUESTION2#
请严格按下面的格式回复：
<result>题目一更好/题目二更好/质量相近</result>
"""

solvable_query = """
你是一位命题教师，请你首先尝试求解下面这道由AI生成的题目，然后判断:题目是否可解，是否存在设计错误
输入的题目是:#INPUT_QUESTION#
请严格按如下格式输出:
<solution>求解过程</solution>
<check_step>分析过程</check_step>
<check_result>题目可解/题目存在少量错误/题目存在错误</check_result>
"""

cov_query = """请你判断题目是否考察了教师要求的知识点，请注意不要求题目与教师要求的知识点完全相同，但不可以完全不相关，以True或False回答
教师要求考查的知识点是#REQUIRED_KNOWLEDGE#
题目为#INPUT_QUESTION#

请严格按如下格式输出:
<result>{True/False}</result>
"""

check_solve_query = """请你判断模型的题目求解过程是否正确，从下面两个角度来判断：
1. 模型的解题过程是否包含错误
2. 模型在解题过程中是否添加了题目中没有的信息
若均无错误，请回答True，否则请回答False
解题过程为:
#INPUT_SOLUTION#

请严格按如下格式输出:
<result>{True/False}</result>
"""

context_query = """请你对下面输入的这道改编的题目进行打分，判断其改编质量。改编要求与人们的日常生活情景相结合，请你打分时综合考虑题目的合理性、情景的贴合度、题目描述的流畅度这三个方面。
输入的题目是:
#INPUT_QUESTION#

请你给出一个范围在1-5的分数，并严格按如下格式输出:
<result>{分数}</result>
"""


def get_clean_token_len(text):
    text = (
        text.replace(" ", "")
        .replace("\n", "")
        .replace("[", "")
        .replace("]", "")
        .replace("-", "")
    )
    text = re.sub(r"\`\`\`.*?\`\`\`", "", text, re.S)
    text = re.sub(r"\*\*.*?\*\*", "", text, re.S)
    return len(text.encode("utf-8"))


def tag_find(text, tags):
    results = []
    for tag in tags:
        findings = re.findall(f"<{tag}>.*?</{tag}>", text, re.S)
        results.append(re.sub(r"<.*?>", "", findings[0]) if len(findings) == 1 else "")
    return results


class UniReward(ORM):
    def __call__(self, completions, task_type, **kwargs) -> List[float]:
        rewards = []
        for content, tp in zip(completions, task_type):
            reward = 0
            # 模型回复格式正确性检查
            if tp == "generate":
                question, design_steps = tag_find(content, ["question", "design_steps"])
                for item in [question, design_steps]:
                    if item == "":
                        reward -= 0.4
            elif tp == "solve":
                solution, knowledge = tag_find(content, ["solution", "knowledge"])
                for item in [solution, knowledge]:
                    if item == "":
                        reward -= 0.4
            elif tp == "check":
                analysis, result, advice, refined_question = tag_find(
                    content, ["analysis", "result", "advice", "refined_question"]
                )
                for item in [analysis, result, advice, refined_question]:
                    if item == "":
                        reward -= 0.25
                if result == "无需修改" and refined_question != "None":
                    reward -= 0.25
            elif tp == "diff":
                modified_steps, modified_question = tag_find(
                    content, ["modified_steps", "modified_question"]
                )
                for item in [modified_steps, modified_question]:
                    if item == "":
                        reward -= 0.3
                    else:
                        token_len = get_clean_token_len(modified_question)
                        reward = (token_len - 500) / 1000
                        if reward > 0.2:
                            reward = 0.2
            elif tp == "context":
                modified_question, _ = tag_find(content, ["modified_question", "_"])
                if modified_question == "":
                    reward -= 0.3
            elif tp == "score":
                modified_solution, _ = tag_find(content, ["modified_solution", "_"])
                if modified_solution == "":
                    reward -= 0.3
            else:
                pass
            rewards.append(reward)
        return rewards


class LLMJudge(ORM):
    def __init__(self):
        # self.args = input_args
        self.client = OpenAI(
            api_key="YOUR_API_KEY",
            base_url="YOUR_BASE_URL")

        if JUDGE_MODEL == "local":
            model_path = MODEL_PATH
            self.orm_model = None
            if model_path is None:
                self.orm_model = None
            elif model_path in orms:
                self.orm_model = orms[model_path]()
            else:
                from swift.llm import PtEngine
                self.orm_model = PtEngine(model_path, max_model_len=2048)

    def get_response(self, messages, model="qwen", request_config=None):
        if JUDGE_MODEL == "local_only" or model == "qwen":
            infer_requests = InferRequest(messages=messages)
            model = self.orm_model
            infer_func = model.infer if isinstance(model, InferEngine) else model.__call__
            stop_tokens = ['<|im_end|>', '<|endoftext|>', '</check_result>', '</result>']
            request_config = RequestConfig(max_tokens=1024, stop=stop_tokens)
            responses = infer_func([infer_requests], request_config=request_config)
            return responses
        else:
            responses = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return responses

    def __call__(self, completions, task_type, knowledge, gpt_question, **kwargs) -> List[float]:
        prm_infer_requests = []
        request_config = kwargs.get('request_config')
        messages = [
            {"role": "user", "content": ""},
        ]
        error_flag = False
        for content, tp, know, gpt in zip(completions, task_type, knowledge, gpt_question):
            if tp == "generate":
                question, design_steps = tag_find(content, ["question", "design_steps"])
                for item in [question, design_steps]:
                    if item == "":
                        error_flag = True
                if random.random() < 0.6:
                    messages[0]["content"] = winrate_query.replace(
                        "#QUESTION1#", question
                    ).replace(
                        "#QUESTION2#", gpt
                    )
                elif random.random() > 0.5:
                    messages[0]["content"] = solvable_query.replace(
                        "#INPUT_QUESTION#", question
                    )
                else:
                    messages[0]["content"] = cov_query.replace(
                        "#INPUT_QUESTION#", question
                    ).replace(
                        "#REQUIRED_KNOWLEDGE#", know
                    )

            elif tp == "solve":
                messages[0]["content"] = check_solve_query.replace(
                    "#INPUT_SOLUTION#", content
                )
            elif "check" in tp:
                analysis, result, advice, refined_question = tag_find(
                    content, ["analysis", "result", "advice", "refined_question"]
                )
                for item in [analysis, result, advice, refined_question]:
                    if item == "":
                        error_flag = True

                if len(refined_question) < 15 or result == "无需修改":
                    refined_question = str(gpt)

                if tp == "check_knowledge":
                    messages[0]["content"] = cov_query.replace(
                        "#INPUT_QUESTION#", refined_question
                    ).replace(
                        "#REQUIRED_KNOWLEDGE#", know
                    )
                elif tp == "check_solvable":
                    messages[0]["content"] = solvable_query.replace(
                        "#INPUT_QUESTION#", refined_question
                    )
                elif tp == "check_text":
                    if random.random() < 0.5:
                        messages[0]["content"] = winrate_query.replace(
                            "#QUESTION1#", refined_question
                        ).replace(
                        "#QUESTION2#", gpt
                        )
                    else:
                        messages[0]["content"] = solvable_query.replace(
                        "#INPUT_QUESTION#", refined_question
                        )

            elif tp == "diff":
                modified_steps, modified_question = tag_find(
                    content, ["modified_steps", "modified_question"]
                )
                for item in [modified_steps, modified_question]:
                    if item == "":
                        error_flag = True

                if random.random() < 0.5:
                    messages[0]["content"] = winrate_query.replace(
                        "#QUESTION1#", modified_steps
                    ).replace(
                        "#QUESTION2#", gpt
                    )
                else:
                    messages[0]["content"] = solvable_query.replace(
                        "#INPUT_QUESTION#", modified_question
                    )

            elif tp == "context":
                modified_question, _ = tag_find(content, ["modified_question", "_"])
                for item in [modified_question]:
                    if item == "":
                        error_flag = True
                messages[0]["content"] = winrate_query.replace(
                    "#QUESTION1#", modified_question
                ).replace(
                    "#QUESTION2#", gpt
                )
            elif tp == "score":
                prm_infer_requests.append(None)
                continue
            else:
                prm_infer_requests.append(None)
                continue

            if error_flag:
                prm_infer_requests.append(None)
                error_flag = False
                continue
            else:
                prm_infer_requests.append(json.loads(json.dumps(messages)))

        assert len(prm_infer_requests) == len(completions)

        rewards = []
        for req in prm_infer_requests:
            if req is None:
                rewards.append(-0.2)
                continue
            response = self.get_response(req, request_config=request_config)
            if JUDGE_MODEL == "local":
                response_str = str(response[0].choices[0].message.content)

            # Solvability Reward or Solution Correctness Reward or Knowledge Coverage Reward
            if "True" in response_str:
                rewards.append(0.5)
            elif "False" in response_str:
                rewards.append(-0.5)
            elif "<check_result>" in response_str and "<solution>" in response_str:
                # Solvability Reward + Complexity Reward
                reward = 0
                solution, check_reuslt = tag_find(
                    response_str, ["solution", "check_result"]
                )
                if "题目可解" in check_reuslt:
                    reward += 0.5
                elif "题目存在少量错误" in check_reuslt:
                    reward -= 0.2
                elif "题目存在错误" in check_reuslt:
                    reward -= 0.5

                comp = get_clean_token_len(solution)
                if comp > 400:
                    reward += 0.1
                elif comp < 50:
                    reward -= 0.1
                rewards.append(reward)
            
            # Winrate Reward
            elif "题目一更好" in response_str:
                rewards.append(0.5)

            elif "质量相近" in response_str:
                rewards.append(0.1)

            elif "题目二更好" in response_str:
                rewards.append(-0.3)
            elif "<result>" in response_str:
                content = tag_find(response_str, ["result"])[0]
                try:
                    if content == "":
                        findings = re.findall("\d", response_str)
                        if findings == []:
                            rewards.append(0.0)
                            continue
                        else:
                            content = findings[-1]
                    rewards.append((float(content) - 6) / 10)
                except Exception as e:
                    rewards.append(0)
            else:
                print("Failed to match task type")
                print("req\n", req)
                print("Response Str\n", response_str)
                rewards.append(0)
        
        assert len(rewards) == len(completions)
        return rewards




orms["external_uni"] = UniReward
orms["external_llm_judge"] = LLMJudge
