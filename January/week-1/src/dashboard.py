import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class ElectricityDashboard:
    def __init__(self, data_processor, predictor):
        self.data_processor = data_processor
        self.predictor = predictor
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("⚡ Peak Hour Electricity Analysis Dashboard", 
                           className="text-center mb-4 mt-4",
                           style={'color': '#00d9ff', 'font-weight': 'bold'}),
                    html.P("Hourly electricity consumption analysis with moving average smoothing and linear regression predictions",
                          className="text-center text-muted mb-4")
                ])
            ]),
            
            # Metrics Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("RMSE", className="text-center"),
                            html.H2(id="metric-rmse", className="text-center text-info")
                        ])
                    ], className="mb-3")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("MAE", className="text-center"),
                            html.H2(id="metric-mae", className="text-center text-success")
                        ])
                    ], className="mb-3")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("R² Score", className="text-center"),
                            html.H2(id="metric-r2", className="text-center text-warning")
                        ])
                    ], className="mb-3")
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("MAPE", className="text-center"),
                            html.H2(id="metric-mape", className="text-center text-danger")
                        ])
                    ], className="mb-3")
                ], width=3),
            ]),
            
            # Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Label("Smoothing Window (hours):", className="font-weight-bold"),
                            dcc.Slider(
                                id='smoothing-slider',
                                min=6,
                                max=72,
                                step=6,
                                value=24,
                                marks={i: str(i) for i in range(6, 73, 12)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ])
                    ], className="mb-3")
                ], width=12)
            ]),
            
            # Time Series Plot
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("📈 Time Series: Raw vs Smoothed Data")),
                        dbc.CardBody([
                            dcc.Graph(id='timeseries-plot', config={'displayModeBar': False})
                        ])
                    ], className="mb-3")
                ], width=12)
            ]),
            
            # Prediction Plot
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🎯 Actual vs Predicted Load")),
                        dbc.CardBody([
                            dcc.Graph(id='prediction-plot', config={'displayModeBar': False})
                        ])
                    ], className="mb-3")
                ], width=12)
            ]),
            
            # Evening Peaks and Heatmap
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🌆 Evening Peak Trends")),
                        dbc.CardBody([
                            dcc.Graph(id='evening-peaks-plot', config={'displayModeBar': False})
                        ])
                    ], className="mb-3")
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("🔥 Weekly Consumption Heatmap")),
                        dbc.CardBody([
                            dcc.Graph(id='heatmap-plot', config={'displayModeBar': False})
                        ])
                    ], className="mb-3")
                ], width=6)
            ]),
            
            # Feature Importance
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("📊 Feature Importance")),
                        dbc.CardBody([
                            dcc.Graph(id='feature-importance-plot', config={'displayModeBar': False})
                        ])
                    ], className="mb-3")
                ], width=12)
            ]),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P("Peak Hour Electricity Analysis | Powered by Plotly Dash",
                          className="text-center text-muted mb-4")
                ])
            ])
            
        ], fluid=True, style={'backgroundColor': '#0a0e27'})
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('metric-rmse', 'children'),
             Output('metric-mae', 'children'),
             Output('metric-r2', 'children'),
             Output('metric-mape', 'children'),
             Output('timeseries-plot', 'figure'),
             Output('prediction-plot', 'figure'),
             Output('evening-peaks-plot', 'figure'),
             Output('heatmap-plot', 'figure'),
             Output('feature-importance-plot', 'figure')],
            [Input('smoothing-slider', 'value')]
        )
        def update_dashboard(smoothing_window):
            # Reprocess data with new smoothing window
            self.data_processor.apply_moving_average(window=smoothing_window)
            df_features = self.data_processor.extract_features()
            train_df, test_df = self.data_processor.get_train_test_split(df_features)
            
            # Retrain model
            self.predictor.train(train_df)
            metrics = self.predictor.evaluate(test_df)
            
            # Get predictions
            test_df['predicted_load'] = self.predictor.predict(test_df)
            
            # Update metrics
            rmse_text = f"{metrics['rmse']:.1f} MW"
            mae_text = f"{metrics['mae']:.1f} MW"
            r2_text = f"{metrics['r2']:.3f}"
            mape_text = f"{metrics['mape']:.1f}%"
            
            # Create visualizations
            timeseries_fig = self.create_timeseries_plot()
            prediction_fig = self.create_prediction_plot(test_df)
            evening_fig = self.create_evening_peaks_plot()
            heatmap_fig = self.create_heatmap()
            feature_fig = self.create_feature_importance_plot()
            
            return (rmse_text, mae_text, r2_text, mape_text,
                   timeseries_fig, prediction_fig, evening_fig, heatmap_fig, feature_fig)
    
    def create_timeseries_plot(self):
        df = self.data_processor.smoothed_df
        
        # Use last 30 days for better visibility
        df_recent = df.tail(30 * 24)
        
        fig = go.Figure()
        
        # Raw data
        fig.add_trace(go.Scatter(
            x=df_recent['datetime'],
            y=df_recent['load_mw'],
            mode='lines',
            name='Raw Data',
            line=dict(color='rgba(100, 149, 237, 0.3)', width=1),
            hovertemplate='%{y:.1f} MW<extra></extra>'
        ))
        
        # Smoothed data
        fig.add_trace(go.Scatter(
            x=df_recent['datetime'],
            y=df_recent['load_smoothed'],
            mode='lines',
            name='Smoothed (Moving Avg)',
            line=dict(color='#00d9ff', width=2),
            hovertemplate='%{y:.1f} MW<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Date',
            yaxis_title='Load (MW)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig
    
    def create_prediction_plot(self, test_df):
        """Create actual vs predicted plot."""
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=test_df['datetime'],
            y=test_df['load_smoothed'],
            mode='lines',
            name='Actual',
            line=dict(color='#00ff88', width=2),
            hovertemplate='%{y:.1f} MW<extra></extra>'
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=test_df['datetime'],
            y=test_df['predicted_load'],
            mode='lines',
            name='Predicted',
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            hovertemplate='%{y:.1f} MW<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Date',
            yaxis_title='Load (MW)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig
    
    def create_evening_peaks_plot(self):
        df = self.data_processor.smoothed_df
        evening_df = self.data_processor.get_evening_peaks(df)
        
        # Group by date and get max evening load
        evening_df['date'] = evening_df['datetime'].dt.date
        daily_peaks = evening_df.groupby('date')['load_smoothed'].max().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_peaks['date'],
            y=daily_peaks['load_smoothed'],
            mode='lines+markers',
            name='Evening Peak',
            line=dict(color='#ffd700', width=2),
            marker=dict(size=6),
            hovertemplate='%{y:.1f} MW<extra></extra>'
        ))
        
        # Add trend line
        z = np.polyfit(range(len(daily_peaks)), daily_peaks['load_smoothed'], 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=daily_peaks['date'],
            y=p(range(len(daily_peaks))),
            mode='lines',
            name='Trend',
            line=dict(color='#ff6b6b', width=2, dash='dot'),
            hovertemplate='%{y:.1f} MW<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Date',
            yaxis_title='Peak Load (MW)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig
    
    def create_heatmap(self):
        df = self.data_processor.smoothed_df.copy()
        df['hour'] = df['datetime'].dt.hour
        df['day_name'] = df['datetime'].dt.day_name()
        
        # Create pivot table
        pivot = df.pivot_table(
            values='load_smoothed',
            index='day_name',
            columns='hour',
            aggfunc='mean'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([day for day in day_order if day in pivot.index])
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Turbo',
            hovertemplate='Hour: %{x}<br>Day: %{y}<br>Avg Load: %{z:.1f} MW<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week'
        )
        
        return fig
    
    def create_feature_importance_plot(self):
        importance_df = self.predictor.get_feature_importance()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_df['abs_coefficient'],
            y=importance_df['feature'],
            orientation='h',
            marker=dict(
                color=importance_df['abs_coefficient'],
                colorscale='Viridis',
                showscale=False
            ),
            hovertemplate='%{y}: %{x:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Absolute Coefficient',
            yaxis_title='Feature',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def run(self, debug=True, port=8050):
        print(f"\n{'='*60}")
        print(f"🚀 Starting Peak Hour Electricity Dashboard...")
        print(f"{'='*60}")
        print(f"📊 Dashboard URL: http://localhost:{port}")
        print(f"💡 Press Ctrl+C to stop the server")
        print(f"{'='*60}\n")
        
        self.app.run(debug=debug, port=port)
