import hashlib

def sha1_hex(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()