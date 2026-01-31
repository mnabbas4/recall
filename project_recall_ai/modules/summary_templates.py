
import json
import os

TEMPLATE_PATH = "data/summary_templates.json"


def load_templates():
    if not os.path.exists(TEMPLATE_PATH):
        return {}

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_templates(templates: dict):
    with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)
