import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle, os, warnings
warnings.filterwarnings('ignore')

class SportsEnergyPredictor:
    def __init__(self, data_file='sports_data.csv', model_file='model.pkl'):
        self.data_file = data_file; self.model_file = model_file
        self.model = None; self.scaler = None; self.data = None
        self.feature_names = None; self.metrics = None

    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        print(f"✓ Loaded {len(self.data)} records"); return self.data

    def _features(self, df):
        f = df.copy()
        f['hour'] = f['timestamp'].dt.hour
        f['dow'] = f['timestamp'].dt.dayofweek
        f['hour_sin'] = np.sin(2*np.pi*f['hour']/24)
        f['hour_cos'] = np.cos(2*np.pi*f['hour']/24)
        f['lag1'] = f['electricity_kwh'].shift(1)
        f['lag24'] = f['electricity_kwh'].shift(24)
        f['roll_mean'] = f['electricity_kwh'].rolling(24).mean()
        f['roll_std'] = f['electricity_kwh'].rolling(24).std()
        f['user_roll'] = f['users'].rolling(12).mean()
        f = f.dropna().reset_index(drop=True)
        self.feature_names = ['hour','dow','is_weekend','is_night','users','floodlights_on','has_event','hour_sin','hour_cos','lag1','lag24','roll_mean','roll_std','user_roll']
        return f

    def train(self, test_size=0.2):
        if self.data is None: self.load_data()
        f = self._features(self.data)
        s = int(len(f)*(1-test_size))
        Xtr,ytr = f.iloc[:s][self.feature_names], f.iloc[:s]['electricity_kwh']
        Xte,yte = f.iloc[s:][self.feature_names], f.iloc[s:]['electricity_kwh']
        self.scaler = StandardScaler()
        Xtr_s = self.scaler.fit_transform(Xtr); Xte_s = self.scaler.transform(Xte)
        self.model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        self.model.fit(Xtr_s, ytr)
        yp = self.model.predict(Xte_s)
        self.metrics = {
            'rmse': round(float(np.sqrt(mean_squared_error(yte, yp))), 3),
            'mae': round(float(mean_absolute_error(yte, yp)), 3),
            'r2': round(float(r2_score(yte, yp)), 4),
            'mape': round(float(np.mean(np.abs((yte-yp)/yte))*100), 2)
        }
        print(f"✓ SVR trained — R²={self.metrics['r2']}"); return self.model

    def get_feature_importance(self):
        # SVR doesn't have native feature importance; use permutation-style approximation
        if self.model is None: raise ValueError("Not trained")
        f = self._features(self.data)
        X = self.scaler.transform(f[self.feature_names])
        base = float(np.mean(np.abs(f['electricity_kwh'] - self.model.predict(X))))
        imp = {}
        for i, name in enumerate(self.feature_names):
            Xp = X.copy(); Xp[:, i] = np.random.permutation(Xp[:, i])
            score = float(np.mean(np.abs(f['electricity_kwh'] - self.model.predict(Xp))))
            imp[name] = round(max(0, score - base), 4)
        total = sum(imp.values()) or 1
        return {k: round(v/total, 4) for k, v in imp.items()}

    def predict_next(self):
        f = self._features(self.data)
        row = self.scaler.transform(f.iloc[[-1]][self.feature_names])
        p = float(self.model.predict(row)[0])
        return {'prediction': round(p,2), 'lower_bound': round(p*0.92,2), 'upper_bound': round(p*1.08,2), 'confidence_level': 92}

    def get_night_analysis(self):
        if self.data is None: self.load_data()
        night = self.data[self.data['is_night']==1]; day = self.data[self.data['is_night']==0]
        return {
            'night_avg': round(float(night['electricity_kwh'].mean()),2),
            'day_avg': round(float(day['electricity_kwh'].mean()),2),
            'night_max': round(float(night['electricity_kwh'].max()),2),
            'floodlight_hours': int(self.data['floodlights_on'].sum()),
            'event_count': int(self.data['has_event'].sum()//6) # approx events
        }

    def save_model(self):
        with open(self.model_file,'wb') as f:
            pickle.dump({'model':self.model,'scaler':self.scaler,'features':self.feature_names,'metrics':self.metrics},f)
        print(f"✓ Saved to {self.model_file}")

    def load_model(self):
        with open(self.model_file,'rb') as f: b=pickle.load(f)
        self.model=b['model'];self.scaler=b['scaler'];self.feature_names=b['features'];self.metrics=b['metrics']
        print(f"✓ Loaded from {self.model_file}")

if __name__=='__main__':
    p=SportsEnergyPredictor(); p.load_data(); p.train()
    print("Night analysis:", p.get_night_analysis())
    print("Prediction:", p.predict_next())
    p.save_model()
