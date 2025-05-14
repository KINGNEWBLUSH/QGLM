import random
import sys

from utils.generage_agents import solver
from utils.tools import tag_find
sys.path.append('./utils')
from llm_service import llm_service
import re
import traceback


KNOWLEDGE_POINTS = ['集合', '不等式', '复数', '平面向量', '不等式', '排列组合', '对数函数', '指数函数', '三角函数', '等差数列', '等比数列', '导数', '概率']
DIFF_CHANGE_METHOD = ["通过增加题目的计算复杂度来增加难度，对原题目的题干进行扩充使得解答需要额外的计算，对题目的数值进行调整使得计算难度加大\n注意若对数值进行调整新的数值应符合实际情景，如物品的个数应为整数、与日常生活相关的数值不应设置的过大或过小",
"""
通过改变题目的条件来调整难度，通过将题目考查的情景泛化来增加难度
选取原题目中的一个给定条件，将其变为变化的条件，使得题目的解题过程变得复杂。
示例:
原题目为:'已知点 \( A\left(\\frac{\sqrt{3}}{2}, \\frac{1}{2}\\right) \)，点 \( P \) 和 \( Q \) 在圆 \( x^2 + y^2 = 5 \) 上运动。同时，点 \( A \)、\( P \)、\( Q \) 保持 \( AP \) 垂直于 \( AQ \) 的关系。求 \( PQ \) 的最大值。'
<modified_steps>
题目涉及到了点 \( A \) 的坐标，点 \( P \) 和 \( Q \) 在圆上运动的条件，我选择将将A变为动点来增加题目难度。
具体来说，可以使得A也在另一圆上运动，如圆 \( x^2 + y^2 = 1 \)，将题目改编为:'给定点 \( A \) 在圆 \( x^2 + y^2 = 1 \) 上移动，点 \( P \) 和 \( Q \) 在圆 \( x^2 + y^2 = 5 \) 上移动。如果点 \( A \)、\( P \)、\( Q \) 在它们的运动中始终保持 \( AP \) 垂直于 \( AQ \)，则 \( PQ \) 的最大值是多少？'
</modified_steps>
<modified_question>
给定点 \( A \) 在圆 \( x^2 + y^2 = 1 \) 上移动，点 \( P \) 和 \( Q \) 在圆 \( x^2 + y^2 = 5 \) 上移动。如果点 \( A \)、\( P \)、\( Q \) 在它们的运动中始终保持 \( AP \) 垂直于 \( AQ \)，则 \( PQ \) 的最大值是多少？
</modified_question>
""",
"""
通过引入新的概念，来增加题目难度。
首先选择一个全新的数学知识点，后将原本题目的条件与新的知识点相结合来增加题目的难度。
需要注意改编后的题目应当与原题目的解题过程相似，且数值设计应符合生活常识
可选知识点:['集合', '不等式', '复数', '向量', '不等式', '排列组合', '函数', '数列', '导数', '概率']
示例:
原题目为:'若 3sinx + cosx = a, x∈ (- π^3 ，π^2), 求实数a的取值范围'
<modified_steps>
选择的知识点为'集合'
为了将原题目与新知识点结合，我将原题目中的'3sinx + cosx'进行集合化处理, 设置集合P={y|y =3sinx + cosx, x∈ (- π^3 ，π^2)}
'集合'的常见内容为:集合的交集或并集
因此我设置集合Q = {a}, 考虑其与集合P的并集, 假定 P \cup Q=P, 将题目改编为'已知集合P={y|y =3sinx + cosx, x∈ (- π^3 ，π^2)}, Q = {a}，若 P \cup Q=P, 求实数a的取值范围'
</modified_steps>
<modified_question>
已知集合P={y|y =3sinx + cosx, x∈ (- π^3 ，π^2)}, Q = {a}，若 P \cup Q=P, 求实数a的取值范围
</modified_question>
"""
]

DIFF_CHANGE_PROMPT = """\
请你对输入的题目进行更改，增加题目的难度。
修改难度使用的方法为：{diff_change_method}
请按如下格式输出
<modified_steps>
**question modify steps**
</modified_steps>
<modified_question>
**content of modified question**
</modified_question>
请使用中文输出
原题目为：{question}
"""

def diff_change(llm, inputs, method=-1):
    if method == -1:
        method = random.randint(0, 2)
    diff_change_method = DIFF_CHANGE_METHOD[method]

    messages = [{"role": "user", "content": DIFF_CHANGE_PROMPT.format(
        diff_change_method=diff_change_method,
        question=inputs['question'],
    )}]
    response = llm.get_response(messages)
    if(response['status'] != 'SUCCESS'):
        print("Error in changing the question")
        print(response['response'])
        return {"status": "FAILED", "data": response['response']}
    try:
        print(response['response'])
        modified_steps, modified_question = tag_find(response['response'], ['modified_steps', 'modified_question'])
        inputs['diff_change_method'] = diff_change_method
        inputs['modified_steps'] = modified_steps
        inputs['question'] = modified_question
    except Exception as e:
        print("Error in generating question, generation format error")
        findings = re.findall(r"<modified_question>.*?</modified_question>", response['response'], re.S)
        modified_question = findings[0].replace("**content of modified question**", "")
        modified_question = re.sub(r"<.*?>", "", modified_question)
        inputs['diff_change_method'] = diff_change_method
        inputs['modified_steps'] = ""
        inputs['question'] = modified_question
        return {"status": "SUCCESS", "data": inputs}
    return {"status": "SUCCESS", "data": inputs}

def context_change(llm, inputs, context="生活中的应用"):
    context_change_prompt = f"""
请你在保证题目解题过程不变的情况下，更改原题目的情景为不同场景，修改题目的情景为'{context}'，使得题目与生活中的应用相结合。
原题目为：{inputs['question']}
请按如下格式输出
<modified_question>
**content of modified question**
</modified_question>
"""
    messages = [{"role": "user", "content": context_change_prompt}]
    response = llm.get_response(messages)
    if(response['status'] != 'SUCCESS'):
        print("Error in changing the context")
        print(response['response'])
        return {"status": "FAILED", "data": response['response']}

    modified_question = tag_find(response['response'], ['modified_question'])
    inputs['modified_question'] = modified_question
        
    return {"status": "SUCCESS", "data": inputs}


def grading_standards(llm, inputs, total_score=None):
    grading_prompt = f"""
请你根据题目的解题步骤给出步骤对应的分值。请你完全复制原题目的解题步骤，并在其后添加步骤对应的分数。越关键的步骤分值越高，不重要的步骤可以没有分值。
{"题目的总分是"+str(total_score)+"分" if total_score is not None else ""}
题目的解题步骤是：
{inputs['solution']}
评分标准示例：
content of step1(1分)
content of step2(0分)
content of step3(2分)
请按如下格式输出：
<modified_solution>
**content of grading standards**
</modified_solution>
"""
    messages = [{"role": "user", "content": grading_prompt}]
    response = llm.get_response(messages)
    if(response['status'] != 'SUCCESS'):
        print("Error in forking the question")
        print(response['response'])
        return {"status": "FAILED", "data": response['response']}

    modified_solution = tag_find(response['response'], ['modified_solution'])
    inputs['modified_solution'] = modified_solution
    return {"status": "SUCCESS", "data": inputs}
