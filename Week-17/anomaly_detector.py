# Dummy anomaly highlighting function

def highlight_anomalies(data):
    # Add a new field 'anomaly_highlight' for demo
    for row in data:
        if row.get('status') == 'anomaly' or (isinstance(row.get('score'), str) and row['score'].isdigit() and int(row['score']) > 100):
            row['anomaly_highlight'] = 'YES'
        else:
            row['anomaly_highlight'] = 'NO'
    return data
