import os
from typing import Optional
from supabase import create_client, Client

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

_client: Optional[Client] = None

def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError('Supabase credentials are not configured')
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def get_csv_link(bank_name: str, account_number: str, ifsc_code: str) -> Optional[str]:
    client = get_client()
    res = client.table('bank_accounts').select('csv_data_links') \
        .eq('bank_name', bank_name) \
        .eq('account_number', account_number) \
        .eq('ifsc_code', ifsc_code) \
        .maybe_single() \
        .execute()
    data = getattr(res, 'data', None)
    if data and isinstance(data, dict):
        return data.get('csv_data_links')
    return None
