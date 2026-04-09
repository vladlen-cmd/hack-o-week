# Dummy decryption function for demo

def decrypt_data(row):
    # Replace with real decryption logic
    return {k: v.replace('ENCRYPTED_', '') if isinstance(v, str) else v for k, v in row.items()}
