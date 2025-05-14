import json
import re

def tag_find(text, tags):
    results = []
    for tag in tags:
        findings = re.findall(f"<{tag}>.*?</{tag}>", text, re.S)
        results.append(re.sub(r"<.*?>", "", findings[0]) if len(findings) > 0 else "")
    return results

def load_jsonl(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records