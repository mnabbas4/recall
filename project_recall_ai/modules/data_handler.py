# modules/data_handler.py
import pandas as pd

REQUIRED_COLS = ['COMMESSA', 'CLIENTE', 'ANNO', 'TIPO MACCHINA', 'APPLICAZIONE', 'TIPO PROBLEMA', 'DESCRIZIONE', 'SOLUZIONE LESSON LEARNED', 'DATA INSERIMENTO', 'RCPRD', 'REPORT CANTIERE', 'CONCERNED DEPARTMENTS', 'REPORT RIUNIONE CHIUSURA PROGETTO']

class DataHandler:
    @staticmethod
    def read_and_validate(uploaded_file):
        try:
            name = uploaded_file.name.lower()
            if name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            return None, f"Could not read file: {e}"

        # Try to map columns heuristically (case-insensitive)
        cols = list(df.columns)
        mapping = {}
        lower_map = {c.lower(): c for c in cols}
        for rc in REQUIRED_COLS:
            if rc in cols:
                continue
            if rc.lower() in lower_map:
                mapping[lower_map[rc.lower()]] = rc
            else:
                # try small variations
                for c in cols:
                    if c.strip().lower().startswith(rc.split()[0].lower()):
                        mapping[c] = rc
                        break

        if mapping:
            df = df.rename(columns=mapping)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            return None, f"Missing required columns: {missing}"

        df = df[REQUIRED_COLS].fillna('')
        return df, None
