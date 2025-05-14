import json
from torch import rand
from utils.generage_agents import generator, iterate_refine, quality_check, solver
from utils.llm_service import llm_service
from utils.modify_agents import context_change, diff_change, grading_standards
from utils.tools import tag_find



def question_generate(llm, knowledge):
    record = {}
    response = generator(llm, {'type':"generate", "knowledge":knowledge})

    question = json.loads(json.dumps(record["question"]))
    for retry in range(3):
        print("Solving question")
        response = solver(llm, question)
        status = response["status"]
        question = response["data"]

        print("checking question")
        if(status != "SUCCESS"):
            continue
        response = quality_check(llm, question, retry+1)
        status = response["status"]
        question = response["data"]
        print(question["remark_text"])

        if(question["remark"] == "Okay"):
            break
    record["question"] = question
    record["status"] = response["status"]

    return record

def question_modify(llm, question, request=""):
    record = {}
    record["question"] = question
    if request == "diff_change":
        response = diff_change(llm, json.loads(json.dumps(record["question"])))
    elif request == "integrate_senario":
        response = context_change(llm, json.loads(json.dumps(record["question"])))
    elif request == "grading":
        response = grading_standards(llm, json.loads(json.dumps(record["question"])))
    else:
        return None
    record["status"] = response["status"]

    if request == "diff_change":
        response = solver(llm, response["data"])
        checked_question = iterate_refine(llm, response["data"], check_start=2)
        print(checked_question["question"])
        record["question"] = checked_question
    
    return record