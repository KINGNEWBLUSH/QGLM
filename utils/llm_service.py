from flask import request
import openai
from openai import OpenAI
import traceback



class llm_service:
    def __init__(self, api_key="", base_url="", model="gpt-4o-mini"):
        openai.api_key = api_key
        self.model = model
        
        openai.api_key = api_key
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def get_response(self, messages, temperature=1, max_tokens=2048):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            try:
                self.update_useage(completion.usage.total_tokens)
            except Exception:
                pass
            return {'status': 'SUCCESS', 'response': completion.choices[0].message.content}
        except Exception as e:
            return {'status': 'FAILED', 'response': traceback.format_exc()}
