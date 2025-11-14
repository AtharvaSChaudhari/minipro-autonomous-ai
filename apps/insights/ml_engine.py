from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any


def compute_insights_from_url(csv_url: str, monthly_income: float) -> Dict[str, Any]:
    df = pd.read_csv(csv_url)
    norm = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    rev = {v: k for k, v in norm.items()}

    def pick(syns):
        for s in syns:
            if s in rev:
                return rev[s]
        return None

    date_col = pick([
        'date', 'transaction_date', 'txn_date', 'value_date', 'posting_date', 'timestamp', 'transaction_dt'
    ])
    if not date_col:
        raise ValueError('CSV missing required column: date')

    amount_col = pick(['amount', 'transaction_amount', 'amt', 'value', 'amount_inr', 'inr_amount', 'transaction_amt'])
    debit_col = pick(['debit', 'debit_amount', 'withdrawn', 'withdrawal_amount'])
    credit_col = pick(['credit', 'credit_amount', 'deposit', 'deposited_amount'])
    type_col = pick(['type', 'transaction_type', 'cr_dr', 'crdr', 'dr_cr', 'debit_credit'])
    category_col = pick(['category', 'sub_category', 'spending_category'])
    upi_col = pick(['upi_app', 'upi', 'app', 'app_name', 'upi_app_name', 'upi_platform', 'upi_app_'])

    tx = df.copy()
    tx['date'] = pd.to_datetime(tx[date_col], errors='coerce')
    tx = tx.dropna(subset=['date'])
    tx['month'] = tx['date'].dt.to_period('M').astype(str)

    def to_num(series):
        return pd.to_numeric(series, errors='coerce').fillna(0.0)

    if type_col:
        t = tx[type_col].astype(str).str.lower().str.strip()
        t = t.replace({
            'dr': 'debit', 'd': 'debit', 'debits': 'debit', 'withdrawal': 'debit', 'payment': 'debit', 'paid': 'debit', 'outflow': 'debit',
            'cr': 'credit', 'c': 'credit', 'credits': 'credit', 'deposit': 'credit', 'received': 'credit', 'inflow': 'credit', 'refund': 'credit', 'salary': 'credit'
        })
        tx['type'] = t
    else:
        tx['type'] = ''

    if debit_col or credit_col:
        d_series = to_num(tx[debit_col]) if debit_col else pd.Series(0.0, index=tx.index)
        c_series = to_num(tx[credit_col]) if credit_col else pd.Series(0.0, index=tx.index)
        tx['_debit_v'] = d_series
        tx['_credit_v'] = c_series
        def infer_type(row):
            if row['_debit_v'] > 0 and row['_credit_v'] == 0:
                return 'debit'
            if row['_credit_v'] > 0 and row['_debit_v'] == 0:
                return 'credit'
            if row['_debit_v'] >= row['_credit_v']:
                return 'debit'
            return 'credit'
        tx['amount'] = np.where(d_series > 0, d_series, c_series)
        tx.loc[tx['type'] == '', 'type'] = tx.loc[tx['type'] == ''].apply(infer_type, axis=1)
    elif amount_col:
        a = to_num(tx[amount_col])
        if (tx['type'] == '').any():
            tx.loc[a < 0, 'type'] = 'debit'
            tx.loc[a >= 0, 'type'] = 'credit'
        tx['amount'] = a.abs()
    else:
        raise ValueError('CSV missing required column: amount')

    if category_col:
        tx['category'] = tx[category_col].astype(str)
    if upi_col:
        tx['upi_app'] = tx[upi_col].astype(str)

    spend_df = tx[tx['type'] == 'debit'].copy()

    monthly_spend = spend_df.groupby('month')['amount'].sum().sort_index()
    monthly_savings = monthly_spend.apply(lambda s: float(max(monthly_income - s, 0)))

    categories = {}
    if 'category' in spend_df.columns:
        categories = spend_df.groupby('category')['amount'].sum().sort_values(ascending=False).head(12).to_dict()
    upi_distribution = {}
    if 'upi_app' in spend_df.columns:
        upi_distribution = spend_df.groupby('upi_app')['amount'].sum().sort_values(ascending=False).head(12).to_dict()

    avg_spend = float(monthly_spend.mean()) if len(monthly_spend) else 0.0
    vol = float(monthly_spend.std()) if len(monthly_spend) > 1 else 0.0
    vol_norm = (min(vol / (avg_spend + 1e-6), 1.5) / 1.5) if avg_spend > 0 else 0.0
    savings_rate = (monthly_income - avg_spend) / monthly_income if monthly_income > 0 else 0.0

    income_series = {m: float(monthly_income) for m in monthly_spend.index}
    ratio_series = {m: float((monthly_spend[m] / monthly_income) * 100.0) if monthly_income > 0 else 0.0 for m in monthly_spend.index}
    spike_threshold = avg_spend + vol if len(monthly_spend) > 1 else float('inf')
    spikes = [str(m) for m, v in monthly_spend.to_dict().items() if float(v) > spike_threshold]

    needs_keywords = ['grocery', 'grocer', 'rent', 'utility', 'electric', 'water', 'gas', 'education', 'medical', 'insurance', 'fuel', 'transport']
    if 'category' in spend_df.columns:
        def is_need(cat: str) -> bool:
            c = str(cat).lower()
            return any(k in c for k in needs_keywords)
        wants = spend_df[~spend_df['category'].apply(is_need)]['amount'].sum()
        total_spend = spend_df['amount'].sum()
        discretionary_ratio = float(wants / total_spend) if total_spend > 0 else 0.0
    else:
        discretionary_ratio = 0.5

    risk = 100.0 * (0.6 * (1 - max(0.0, min(savings_rate, 1.0))) + 0.3 * vol_norm + 0.1 * max(0.0, min(discretionary_ratio, 1.0)))
    risk = float(max(0.0, min(risk, 100.0)))
    eligible = bool(savings_rate >= 0.2 and risk <= 50.0)

    insights = {
        'monthly_spendings': {k: float(v) for k, v in monthly_spend.to_dict().items()},
        'monthly_savings': {k: float(v) for k, v in monthly_savings.to_dict().items()},
        'total_savings': float(monthly_savings.sum()),
        'categories': {str(k): float(v) for k, v in categories.items()},
        'upi_distribution': {str(k): float(v) for k, v in upi_distribution.items()},
        'risk_score': risk,
        'loan_eligible': eligible,
        'income_per_month': float(monthly_income),
        'total_spend_180': float(spend_df['amount'].sum()),
        'months_covered': sorted(list(set(monthly_spend.index))),
        'income_series': income_series,
        'expense_to_income_ratio': ratio_series,
        'spending_volatility_std': float(vol),
        'spending_volatility_index': float(vol_norm * 100.0),
        'spending_spikes': spikes,
    }
    return insights
