import os
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from .utils import safe_filename

class MemoryManager:
    def __init__(self, data_dir='data'):
        self.base = Path(data_dir)
        self.upload_dir = self.base / 'uploads'
        self.mem_dir = self.base / 'memories'
        self.index_fname = self.mem_dir / 'memories_index.json'
        self.index = self._load_index()

    def _load_index(self):
        if self.index_fname.exists():
            return json.loads(self.index_fname.read_text())
        return {}

    def _save_index(self):
        self.index_fname.write_text(json.dumps(self.index, indent=2))

    def list_memories(self):
        return list(self.index.keys())

    def list_memories_full(self):
        return {k:v['memory_path'] for k,v in self.index.items()}

    def save_upload(self, uploaded_file):
        # uploaded_file is a streamlit UploadedFile
        fname = safe_filename(uploaded_file.name)
        dest = self.upload_dir / fname
        with open(dest, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return str(dest)

    def create_or_update_memory(self, name, df: pd.DataFrame, mode='Create new memory file', target_memory=None):
        mid = safe_filename(name)
        mem_path = str(self.mem_dir / f"{mid}.parquet")
        meta = {
            'memory_id': mid,
            'memory_path': mem_path,
            'timestamp': datetime.utcnow().isoformat(),
            'n_records': len(df)
        }
        # write parquet
        df.to_parquet(mem_path, index=False)
        # update index
        self.index[mid] = meta
        self._save_index()
        return meta

    def load_memory_dataframe(self, mem_id):
        meta = self.index.get(mem_id)
        if not meta:
            raise FileNotFoundError('Memory not found')
        return pd.read_parquet(meta['memory_path'])



    def generate_llm_summary(
        self,
        insights: dict,
        query: str,
        template: dict,
        instructions: str
    ) -> str:
    
        client = OpenAI()
    
        prompt = f"""
    You are an expert analyst.
    
    User query:
    {query}
    
    Historical insights:
    - Total matches: {insights['matches']}
    - Applications: {', '.join(insights.get('top_applications', []))}
    - Machine types: {', '.join(insights.get('top_machine_types', []))}
    
    Common problems:
    {chr(10).join(insights.get('common_problems', []))}
    
    Common solutions:
    {chr(10).join(insights.get('common_solutions', []))}
    
    User instructions:
    {instructions}
    
    Required structure:
    Sections: {', '.join(template.get('sections', []))}
    Tone: {template.get('tone')}
    Length: {template.get('length')}
    
    Write a clear, detailed, outcome-driven summary.
    """
    
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
    
        return resp.choices[0].message.content.strip()
