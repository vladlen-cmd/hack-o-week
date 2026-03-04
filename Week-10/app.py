from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
from model import SustainabilityTracker
app=Flask(__name__,static_folder='static'); CORS(app); tracker=SustainabilityTracker()
def init():
    try:
        if os.path.exists('model.pkl') and os.path.exists('campus_data.csv'): tracker.load_data();tracker.load_model()
    except Exception as e: print(f"⚠ {e}")
@app.route('/')
def index(): return send_from_directory('static','index.html')
@app.route('/api/historical')
def hist():
    try:
        if tracker.data is None: tracker.load_data()
        d=tracker.data.copy();d['timestamp']=d['timestamp'].astype(str);return jsonify({'success':True,'data':d.to_dict(orient='records')})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
@app.route('/api/predict')
def predict():
    try:
        if tracker.model is None: return jsonify({'success':False,'error':'Not trained'}),400
        return jsonify({'success':True,'prediction':tracker.predict_next(),'current':{k:float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else v for k,v in tracker.data.iloc[-1].to_dict().items() if k!='timestamp'}|{'timestamp':str(tracker.data.iloc[-1]['timestamp'])}})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
@app.route('/api/stats')
def stats():
    try:
        if tracker.model is None: return jsonify({'success':False,'error':'Not trained'}),400
        return jsonify({'success':True,'metrics':tracker.metrics,'feature_importance':tracker.get_feature_importance()})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
@app.route('/api/sustainability')
def sust():
    try: return jsonify({'success':True,'summary':tracker.get_sustainability_summary()})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
if __name__=='__main__':
    print("="*60);print("Campus-Wide Sustainability Tracker");print("="*60)
    init();print("\n✓ http://localhost:5009");app.run(debug=False,port=5009)
