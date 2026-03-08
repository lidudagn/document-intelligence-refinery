import json, re
text = """```json\n{"relevant_sections": ["Section 1"]}\n```"""
def _parse(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
    return json.loads(text.strip())
print(_parse(text))
