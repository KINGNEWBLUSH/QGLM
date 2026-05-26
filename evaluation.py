from utils import generage_agents
from utils.tools import load_jsonl
import re


REFERENCE_QUESTIONS = load_jsonl("PATH_TO_REFERENCE_QUESTIONS.jsonl")

CHECK_SOLVABLE_PROMPT = """你是一位命题教师，请你首先尝试求解下面这道由AI生成的题目，然后判断:题目是否可解，是否存在设计错误
输入的题目是:{input_quesiton}
请按如下格式输出:
<solution>求解过程</solution>
<check_step>分析过程</check_step>
<check_result>题目正确/题目存在错误</check_result>
"""


def win_rate(recordsA, recordsB, reference):
    recordsB_dict = {}
    for i, record in enumerate(recordsB):
        recordsB_dict[int(record["id"])] = record
    reference_dict = {}
    for i, record in enumerate(reference):
        reference_dict[int(record["id"])] = record

    stor = []
    for i, record in enumerate(recordsA):
        try:
            question1 = record["question"]["question"]
        except Exception:
            question1 = record["question"]

        try:
            question2 = recordsB_dict[int(record["id"])]["question"]
        except Exception:
            continue

        compare_prompt = """请你以一个教师的身份来判断下面两道题目的质量，请你综合考虑问题的设计、数值的设计、题目描述的流畅度做出判断。不要求生成的题目中包含题目答案或者解答提示信息。
请你公平的比较长度较长的题目和长度较短的题目，长度较长的题目不一定质量更好。
题目一: 
{question1}
题目二: 
{question2}
请按下面的格式回复：
<result>题目一更好/题目二更好/质量相近</result>
"""
        try:
            stor.append(
                {
                    "query": compare_prompt.format(
                        question1=question1,
                        question2=question2,
                        reference=reference_dict[int(record["id"])]["question"][
                            "reference"
                        ],
                    ),
                    "id": int(record["id"]),
                }
            )
        except Exception as e:
            print(e)
    return stor


def solvability(records):
    stor = []
    for record in records:
        try:
            question = record["question"]["question"]
        except Exception:
            question = record["question"]

        check_prompt = CHECK_SOLVABLE_PROMPT.format(input_quesiton=question)
        stor.append({"query":check_prompt, "question":question, "id":int(record["id"])})
    return stor


def knowledge_coverage(records, target):
    stor = []
    for i, record in enumerate(records):
        generate_prompt = """请你判断题目是否考察了教师要求的知识点，请注意不要求题目与教师要求的知识点完全相同，但不可以完全不相关，以True或False回答
        教师要求考查的知识点是{required_knowledge}
        题目为{question}
        请以True或False回答
        """
        try:
            stor.append({"query":generate_prompt.format(required_knowledge=target[int(record["id"])]["question"]["knowledge"], question=record["question"]["question"]), "id":int(record["id"])})
        except Exception:
            stor.append({"query":generate_prompt.format(required_knowledge=target[int(record["id"])]["question"]["knowledge"], question=record["question"]), "id":int(record["id"])})

    return stor


def RQD(sbert, record, reference):
    # print(reference['knowledge'])
    exemplar_problems = generage_agents.reference_get_k(
        reference["knowledge"], reference_data=generage_agents.reference_data, k=10
    )
    for question in REFERENCE_QUESTIONS:
        if question in exemplar_problems:
            exemplar_problems.remove(question)
    # print("exemplar_problems", exemplar_problems)
    simi_scores = sbert.set_sim(record["question"], exemplar_problems)
    # print(simi_scores)
    simi_scores = simi_scores.tolist()[0]
    max_score_index = simi_scores.index(max(simi_scores))
    max_score_reference_question = exemplar_problems[max_score_index]
    return max(simi_scores), max_score_reference_question


def reasoning_length(records):
    length = []
    for record in records:
        try:
            text = record['question']['solution']
        except Exception:
            text = record['solution']

        text = text.replace(' ', '').replace('\n', '').replace('[', '').replace(']', '').replace('-', '')
        text = re.sub(r'\`\`\`.*?\`\`\`', '', text, re.S)
        text = re.sub(r'\*\*.*?\*\*', '', text, re.S)
        length.append(len(text.encode('utf-8')))
    
    length.sort()
    return sum(length) / len(length)