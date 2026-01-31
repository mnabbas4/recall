import json
from pathlib import Path

TEMPLATE_FILE = Path("data/summary_templates.json")

DEFAULT_TEMPLATE = {
    "Default": {
        "sections": [
            "Context",
            "Problem",
            "Solution",
            "Lessons Learned"
        ],
        "tone": "simple",
        "length": "short"
    }
}

def load_templates():
    if not TEMPLATE_FILE.exists():
        save_templates(DEFAULT_TEMPLATE)
        return DEFAULT_TEMPLATE

    try:
        data = json.loads(TEMPLATE_FILE.read_text())
        if not data:
            save_templates(DEFAULT_TEMPLATE)
            return DEFAULT_TEMPLATE
        return data
    except Exception:
        save_templates(DEFAULT_TEMPLATE)
        return DEFAULT_TEMPLATE

def save_templates(templates: dict):
    TEMPLATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_FILE.write_text(json.dumps(templates, indent=2))
