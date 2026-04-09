from flask import Flask, send_file, request, jsonify
from encryption_engine import decrypt_data
from anomaly_detector import highlight_anomalies
import pandas as pd
import io
from pathlib import Path

from flask import send_from_directory

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR / 'static'))

# Dummy function to simulate fetching encrypted analyzed data
def get_encrypted_analyzed_data():
    # Replace with actual data source
    return [
        {'user_id': 1, 'score': 'ENCRYPTED_85', 'status': 'ENCRYPTED_normal'},
        {'user_id': 2, 'score': 'ENCRYPTED_120', 'status': 'ENCRYPTED_anomaly'},
    ]

# Dashboard route
@app.route('/')
def dashboard():
    return send_from_directory(app.static_folder, 'index.html')

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# Data API for dashboard
@app.route('/data')
def data_api():
    encrypted_data = get_encrypted_analyzed_data()
    decrypted_data = [decrypt_data(row) for row in encrypted_data]
    highlighted_data = highlight_anomalies(decrypted_data)
    return jsonify(highlighted_data)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/export', methods=['GET'])
def export_data():
    format = request.args.get('format', 'csv')
    # Fetch and decrypt data
    encrypted_data = get_encrypted_analyzed_data()
    decrypted_data = [decrypt_data(row) for row in encrypted_data]
    # Highlight anomalies
    highlighted_data = highlight_anomalies(decrypted_data)
    df = pd.DataFrame(highlighted_data)
    if format == 'csv':
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode()),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='analyzed_data.csv')
    elif format == 'pdf':
        # Simple PDF export using pandas (for demo; use reportlab for advanced)
        import matplotlib.pyplot as plt
        from pandas.plotting import table
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        tab = table(ax, df, loc='center', cellLoc='center')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='pdf')
        buf.seek(0)
        return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name='analyzed_data.pdf')
    else:
        return jsonify({'error': 'Invalid format'}), 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
