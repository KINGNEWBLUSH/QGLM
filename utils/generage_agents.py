import copy
import re
import random
import pandas as pd
import json
import sys

from utils.tools import tag_find
sys.path.append('./utils')
from llm_service import llm_service
import numpy as np
import ast
from sentence_transformers.util import cos_sim
import torch
import traceback

PATH = "data/referenceV3.csv"

def get_knowledge_embedding(knowledge_str):
    knowledge_list = knowledge_str.split(',')
    embedding = []
    for knowledge in knowledge_list:
        embedding.append(knowledge_to_embedding[knowledge])
    return embedding



reference_data = pd.read_csv(PATH)
knowledge_to_embedding = np.load('data/knowledge_embeddingV3.npy', allow_pickle=True).item()
reference_data['know_embedding'] = reference_data['knowledge'].apply(get_knowledge_embedding)
reference_data = reference_data.values

GENERATE_METHODS=[
"""通过交换题目的条件和结论来设计新题目。新题目以源题目中的所求内容作为给定条件，并参考源题目重新进行设计。
例如对于源题：
已知双曲线C：x^2/A^2-y^2/B^2=1(A>0, B>0)的离心率为2, 求点4, 0()到C的渐近线的距离
对条件结论进行交换得新题目：
已知双曲线C：x^2/A^2-y^2/B^2=1(A>0, B>0), 点(4, 0)到C的渐近线的距离为22, 则双曲线C的的离心率为
""",
"""
改变题目题干中的条件来得到新题目。
首先选取题目的某一个条件，然后联想出与其相关的知识点，用这些知识点替换原题条件中的部分内容，从而得到新题目。
例如对于源题：
已知抛物线y^2=2Px(P>0)的准线经过点(-1, 1), 求抛物线焦点坐标
联想出抛物线焦点相关的知识点“弦长”，并改变提问得：
已知抛物线y^2=2Px(P>0)的准线经过点(-1, 1), **求过抛物线焦点Ｆ的最短弦长**
联想出几何中的相切知识，并改变题目条件得:
若抛物线y^2=2Px(P>0)的准线与**圆(x-1)^2+(y+1)^2=4相切**, 求p的值
""",
"""
将题目与生活中常见的情景相结合设计题目
""",
"""
将题目内容进行泛化处理,从而设计新题目。具体来说，将源题条件中的具体数值替换成一般变量，从而使得题目考查更广泛的情景。
例如对于源题：
已知抛物线C：y^2=2x和点P(2, 2), A、B是C上异于点P的两点, 直线PA、PB的\
斜率kPA, kPB满足kPA+kPB=2, 则直线AB过定点()．
A．(1, 0)B．(-1, 0)
C．(0, -1)D．(0, 0)
将直线PA、PB的斜率kPA, kPB关系kPA+kPB=2\
延伸到一般情形kPA+kPB=λ(λ为常数, 且λ!=0), 便得到下面新题目:
P(x0, y0)是抛物线C：y2=2Px(P>0)上一定点, A, B是\
C上异于P的两点, 直线PA, PB的斜率kPA, kPB满足kPA+kPB=\
λ(λ为常数, 且λ!=0), 且直线AB的斜率存在, 则直线AB过定点()．
"""
]

REFINE_PROMPT = """请你按照指示完成以下操作:
1.根据修改意见对原题目进行修改
修改意见是：{requirement}
待修改的题目是：{orinal_question}
2.请优化修改后题目的表达，使其清晰易懂且符合人的思维习惯
请按如下格式输出：
<design_steps>
**the process to design the question**
**the question after modification**
**the question after optimization**
</design_steps>
<question>
**content of output question**
</question>
"""

def generator_DS(llm, knowledge, methods="random"):
    generate_methods = random.choice(GENERATE_METHODS)
    input_knowledge = str(knowledge)
    input_knowledge = input_knowledge.replace(" ", "")
    reference_question = reference_get(input_knowledge)

    generate_prompt = f"""\
请你根据指定的考查知识点以及参考题目生成一道新题目
教师指定的知识点是：{knowledge}
参考题目是：{reference_question}
下面是生成题目的规则：{generate_methods}
请你仔细思考如何设计题目后使用中文按如下格式输出：
[BEGIN]
**content of generated question**
[END]
"""
    return generate_prompt
    

def generator(llm, requirement, input_question_steped={}, reference="", methods="random"):
    """
    Generate questions based on the requirement
    requirement: {
        type: str, the type of the requirement, 'generate' or 'refine'
        knowledge: str, input knowledge
        content: str, the content of the requirement
    }
    """
    # TO BE DONE
    # write the prompt
    # use regex to extrat
    if(methods == "random"):
        generate_methods = random.choice(GENERATE_METHODS)
    elif(methods == "switch"):
        generate_methods = GENERATE_METHODS[0]
    elif(methods == "change"):
        generate_methods = GENERATE_METHODS[1]
    elif(methods == "context"):
        generate_methods = GENERATE_METHODS[2]
    elif(methods == "fanhua"):
        generate_methods = GENERATE_METHODS[3]
    
    question_steped = copy.deepcopy(input_question_steped)
    if(requirement['type'] == 'generate'):
        # get reference question
        input_knowledge = str(requirement["knowledge"])
        input_knowledge = input_knowledge.replace(" ", "")
        if(reference == ""):
            reference_question = reference_get(input_knowledge)
        else:
            reference_question = reference
        question_steped['knowledge'] = requirement["knowledge"]
        question_steped['reference'] = reference_question

        generate_prompt = f"""\
请你根据指定的考查知识点以及参考题目生成一道新题目
教师指定的知识点是：{requirement["knowledge"]}
参考题目是：{reference_question}
下面是生成题目的规则：{generate_methods}
请使用中文按如下格式输出：
<design_steps>
**the process to design the question**
</design_steps>
<question>
**content of generated question**
</question>
"""
        # get and handle response
        messages = [{"role": "user", "content": generate_prompt}]
        print("messages", messages)
        response = llm.get_response(messages)
        print("response", response)
        if(llm.batch_format==True):
            return {"status": "SUCCESS", "data": response}
        if(response['status'] != 'SUCCESS'):
            print("Error in generating question")
            print(response)
            question = "Error in generating question"
            return {"status": "FAILED", "data": question_steped}

        try:
            findings = re.findall(r"<design_steps>.*?</design_steps>", response['response'], re.S)
            design_steps = findings[0].replace("**the process to design the question**", "")
            design_steps = re.sub(r"<.*?>", "", design_steps)
            findings = re.findall(r"<question>.*?</question>", response['response'], re.S)
            question = findings[0].replace("**content of generated question**", "")
            question = re.sub(r"<.*?>", "", question)
            # return the question
            question_steped['design_method'] = generate_methods
            question_steped['design_steps'] = design_steps
            question_steped['question'] = question
        except Exception as e:
            print("Error in generating question, generation format error")
            traceback.print_exc()
            question = "generation format error"
            return {"status": "FAILED", "data": question_steped}
        return {"status": "SUCCESS", "data": question_steped}

    elif(requirement['type'] == 'refine'):
        messages = [{"role": "user", "content": REFINE_PROMPT.format(requirement=requirement['content'], orinal_question=question_steped['question'])}]
        response = llm.get_response(messages)
        if(response['status'] != 'SUCCESS'):
            print("Error in refining question")
            print(response['response'])
            question = "Error in refining question"
            return {"status": "FAILED", "data": question_steped}
        try:
            findings = re.findall(r"<design_steps>.*?</design_steps>", response['response'], re.S)
            design_steps = findings[0].replace("**the process to design the question**", "")
            design_steps = re.sub(r"<.*?>", "", design_steps)
            findings = re.findall(r"<question>.*?</question>", response['response'], re.S)
            question = findings[0].replace("**content of output question**", "")
            question = re.sub(r"<.*?>", "", question)
            question_steped['question'] = question
            return {"status": "SUCCESS", "data": question_steped}
        except Exception as e:
            findings = re.findall(r"<question>.*?</question>", response['response'], re.S)
            question = findings[0].replace("**content of output question**", "")
            question = re.sub(r"<.*?>", "", question)
            question_steped['question'] = question
            return {"status": "SUCCESS", "data": question_steped}
    else:
        return {"status": "FAILED", "data": "invalid requirement type"}

def solver(llm, question_steped):
    """
    Solve the question using chain of thought
    """
    solve_prompt = f"""
你是一个数学专家，请你step by step地解决这个问题。
在求解问题后，请根据解题步骤输出解题过程和涉及的知识点，如'三角函数'、'加减法计算'等。
{question_steped['question']}
请按如下格式输出：
<solution>
**content of solution**
</solution>
<inculude_knowledge>
**knowledge points**
</inculude_knowledge>
"""
    messages = [{"role": "user", "content": solve_prompt}]
    response = llm.get_response(messages, temperature=0)
    if(llm.batch_format==True):
        return {"status": "SUCCESS", "data": response}
    if(response['status'] != 'SUCCESS'):
        print("Error in solving the question")
        print(response['response'])
        question_steped['solution'] = "Error in solving the question"
        return {"status": "FAILED", "data": question_steped}
    
    try:
        findings = re.findall(r"<solution>.*?</solution>", response['response'], re.S)
        solution = findings[0].replace("**content of solution**", "")
        solution = re.sub(r"<.*?>", "", solution)
        question_steped['solution'] = solution
        findings = re.findall(r"<inculude_knowledge>.*?</inculude_knowledge>", response['response'], re.S)
        include_knowledge = findings[0].replace("**knowledge points**", "")
        include_knowledge = re.sub(r"<.*?>", "", include_knowledge)
        question_steped['include_knowledge'] = include_knowledge
    except Exception as e:
        solution = response['response'].replace("**content of solution**", "")
        solution = re.sub(r"<.*?>", "", solution)
        question_steped['solution'] = solution
    question_steped['solution'] = solution
        
    return {"status": "SUCCESS", "data": question_steped}

KNOWLEDGE_CHECK_PROMPT = """请你分析输入的题目是否考查了指定的知识点
如果题目没有考查教师指定的知识点，请给出具体的修改建议，并给出修改后的题目
案例:
input knowledge:出油率与比例计算
input question:小王骑自行车骑了15千米，共消耗了3千卡的能量。那他平均每骑行1千米消耗多少千卡的能量？
<analysis>题目考查了比例计算的知识点，但没有考查出油率的知识点。</analysis>
<result>待修改</result>
<advice>调整题目情景使其涉及出油率，将题目情景由骑车更改为榨油，例如15千克原料能够榨出3kg的油，平均每千克原料能榨出多少千克的油？</advice>
<refined_question>15千克原料能够榨出3kg的油，平均每千克原料能榨出多少千克的油？</refined_question>
请你按如下格式输出：
<analysis>对题目分析的过程</analysis>
<result>无需修改/待修改</result>
<advice>对题目修改的建议(若无需修改则输出'None')</advice>
<refined_question>修改后的题目(若无需修改则输出'None')</refined_question>
用户的输入为:
input knowledge:{knowledge}
input question:{question}
"""

CONDITION_CHECK_PROMPT = """请你分析输入的题目数值、条件设置是否合理，请你检查如下几个方面是否存在问题
1.题目的条件设置是否合理，是否缺少必要的条件导致题目求解时需要手动假设(题目给出的变量不包括在内)，或者是否包含多余的条件没有使用
2.题目的数值设计是否符合常识，如人数、物品的数量应该是整数；如时间、大小、份额的数量设置是否符合生活常识
3.题目给定的条件是否有冲突，如果有，是否应该删除部分的条件
4.是否包含了过多的信息导致题目无需计算
在检查过后，请给出具体的修改建议，并给出修改后的题目
请注意做出的修改应仅局限于题目本身的条件设置和数值设计，不要改变题目的考查知识点和解题思路。
案例一:
input question:'五年级同学栽树800棵，比六年级同学栽树棵数的90%还少200棵，六年级同学栽树多少棵？'
input solution:'六年级同学植树棵树的90%应该为800+200=1000棵。\n所以六年级同学栽树棵数为1000/0.9=1111.11棵。'
<analysis>栽树的棵数应该是整数，需要对题目的数据进行修改，若五年级栽树棵数为Y，(Y+200)/09应为整数，例如Y可以是700。验证:更改后(700+200)/0.9=1000是整数。</analysis>
<result>待修改</result>
<advice>建议将五年级栽树棵数修改为700，使得题目数据合理。</advice>
<refined_question>五年级同学栽树700棵，比六年级同学栽树棵数的90%还少200棵，六年级同学栽树多少棵？</refined_question>
案例二:
input question:'已知食堂原有的大米与又买来的大米的重量比为4:3，假设每天消耗48千克，求食堂原来有大米的总重量和又买来的大米的总重量分别是多少千克？\n'
input solution:'设一共消耗了t天，则消耗总重量为48t千克。则原有大米重量为48t*4/7千克，又买来的大米重量为48t*3/7千克。\n\n所以食堂原来有大米的总重量和又买来的大米的总重量分别是：\n\n原有大米重量：48t*4/7千克\n又买来的大米重量：48t*3/7千克\n'
<analysis>题目中提到的“每天消耗48千克”没有说明是消耗多少天，需要做出假设进行求解，应指定天数信息。</analysis>
<result>待修改</result>
<advice>建议在题目中添加消耗天数的信息，例如增加条件“这些大米可供食堂消耗10天”</advice>
<refined_question>已知食堂原有的大米与又买来的大米的重量比为4:3，这些大米可供食堂消耗10天，假设每天消耗48千克，求食堂原来有大米的总重量和又买来的大米的总重量分别是多少千克？</refined_question>
请你按如下格式输出：
<analysis>对题目分析的过程，并通过具体的计算验证你的修改是否有效</analysis>
<result>无需修改/待修改</result>
<advice>对题目修改的建议(若无需修改则输出'None')</advice>
<refined_question>修改后的题目(若无需修改则输出'None')</refined_question>
用户输入为:
input question：{question}
input solution：{solution}
"""

TEXT_CHECK_PROMPT = """请你对输入题目的叙述进行检查，并判断：
1.题目的叙述是否符合常识
2.题目的语句是否通顺、易读
如果题目不符合常识或语句不通顺，请给出具体的修改建议，并给出修改后的题目
注意，你所作的修改应仅局限于题目本身的叙述，不可以改编题目的数值!
案例一：
input question:'在一个小区内，某超市位于公园的同侧。已知从超市到公园的距离为360米，而从公园到电影院的距离为150米。请问，从超市到电影院的直线距离有多少米？'
<analysis>仅提及了超市和公园的位置关系，缺少电影院的位置，阅读时会产生歧义。</analysis>
<result>待修改</result>
<advice>建议明确电影院的位置关系，如电影院在公园的东侧，超市在公园的西侧等。</advice>
<refined_question>在一个小区内，某超市位于公园的西侧，电影院在公园的东侧。已知从超市到公园的距离为360米，而从公园到电影院的距离为150米。请问，从超市到电影院的直线距离有多少米？</refined_question>
案例二：
input question:'在某座山的特定区域，当海拔高度增加时，气温每小时降低a摄氏度。已知山顶气温为T_top，山脚气温为T_base，其中 T_top = 14.5摄氏度，T_base = 30摄氏度。请问该山峰相对山脚的高度h是多少千米？（假定每千米对应的气温变化为固定值）'
<analysis>在该语境下气温应随高度增加，与小时数无关，题目叙述不符合常识。</analysis>
<result>待修改</result>
<advice>将气温每小时降低a摄氏度改为气温每千米降低a摄氏度</advice>
<refined_question>在某座山的特定区域，当海拔高度增加时，气温每千米降低a摄氏度。已知山顶气温为T_top，山脚气温为T_base，其中 T_top = 14.5摄氏度，T_base = 30摄氏度。请问该山峰相对山脚的高度h是多少千米？（假定每千米对应的气温变化为固定值）</refined_question>
请你按如下格式输出：
<analysis>对题目分析的过程</analysis>
<result>无需修改/待修改</result>
<advice>对题目修改的建议(若无需修改则输出'None')</advice>
<refined_question>修改后的题目(若无需修改则输出'None')</refined_question>
用户的输入为:
input question：{question}
"""

def quality_check(llm, question_steped, phase=0):
    """
    Check the question based on the solution
    """
    
    question_steped['remark_analysis'] = ""
    question_steped['remark'] = "Okay"
    question_steped['remark_text'] = ""
    knowledge_check_prompt = KNOWLEDGE_CHECK_PROMPT.format(knowledge=question_steped['knowledge'], question=question_steped['question'])
    condition_check_prompt = CONDITION_CHECK_PROMPT.format(question=question_steped['question'], solution=question_steped['solution'])
    text_check_prompt = TEXT_CHECK_PROMPT.format(question=question_steped['question'])
    for i, prompt in enumerate([knowledge_check_prompt, condition_check_prompt, text_check_prompt]):
        if(i < phase-1 and phase != 0):
            continue
        print("phase", i+1)
        messages = [{"role": "user", "content": prompt}]
        response = llm.get_response(messages)
        if(response['status'] != 'SUCCESS'):
            print("Error in checking the question")
            print(response['response'])
            question_steped['remark'] = "Error"
            return {"status": "FAILED", "data": question_steped}

        remark = response['response']
        analysis, result, advice, refined_question = tag_find(remark, ['analysis', 'result', 'advice', 'refined_question'])
        remark_analysis = re.findall(r"<analysis>.*?</analysis>", remark, flags=re.S)
        if(len(remark_analysis) > 0):
            question_steped['remark_analysis'] += f"{i+1}." + remark_analysis[0].replace("<analysis>", "").replace("</analysis>", "") + '\n'
        if('待修改' in str(remark)):
            question_steped['remark'] = "Rejected"
            question_steped['remark_text'] = advice.replace('\n', '') + ' '
            question_steped['question'] = refined_question
            return {"status": "SUCCESS", "data": question_steped}

    question_steped['remark'] = "Okay"
    return {"status": "SUCCESS", "data": question_steped}
        

def reference_get(input_knowledge, reference_data=reference_data, k=30):
    """
    Get reference questions based on the knowledge
    """ 
    # "ques_content", "know_name", "know_embedding"
    reference_data = copy.copy(reference_data)
    try:
        input_embedding = knowledge_to_embedding[input_knowledge]
        for i, ele in enumerate(reference_data):
            ele[3] = np.array(ele[3])
            if(ele[2] == "[]"):
                reference_data[i][3] = 0
            else:
                reference_data[i][3] = cos_sim(input_embedding, ele[3])
                reference_data[i][3] = torch.max(reference_data[i][3]).numpy()
        reference_data = sorted(reference_data, key=lambda x: x[3], reverse=True)
        result = random.choice(reference_data[:k])
        return result[0]
    except Exception as e:
        print("Error in getting reference question")
        print("input_knowledge", input_knowledge)
        traceback.print_exc()
        return "Error in getting reference question"


def reference_get_k(input_knowledge, reference_data=reference_data, k=30):
    """
    Get reference questions based on the knowledge
    """ 
    # "ques_content", "know_name", "know_embedding"
    reference_data = copy.copy(reference_data)
    try:
        input_embedding = knowledge_to_embedding[input_knowledge]
        for i, ele in enumerate(reference_data):
            ele[3] = np.array(ele[3])
            if(ele[2] == "[]"):
                reference_data[i][3] = 0
            else:
                reference_data[i][3] = cos_sim(input_embedding, ele[3])
                reference_data[i][3] = torch.max(reference_data[i][3]).numpy()
        reference_data = sorted(reference_data, key=lambda x: x[3], reverse=True)
        reference_data = reference_data[0:k]
        for i, ele in enumerate(reference_data):
            reference_data[i] = list(ele)[0]
        return reference_data[0:k]
    except Exception as e:
        print("Error in getting reference question")
        print("input_knowledge", input_knowledge)
        traceback.print_exc()
        return ["Error in getting reference question"]


def iterate_refine(llm, question, check_start=0):
    for retry in range(check_start, 5):
        print("Solving question")
        response = solver(llm, question)
        status = response["status"]
        question = response["data"]
        print("checking question")
        if(status != "SUCCESS"):
            continue
        response = quality_check(llm, question, retry)
        status = response["status"]
        question = response["data"]
        print(question["remark_text"])
        print(question["question"])
        if(status != "SUCCESS"):
            continue
        elif(question["remark"] == "Okay"):
            break
    return question
