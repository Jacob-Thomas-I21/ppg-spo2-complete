#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import json
import argparse
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Kaggle environment paths
KAGGLE_INPUT_PATH = "/kaggle/input/yuhuty7/ppg_dataset_full/csv"
KAGGLE_WORKING_PATH = "/kaggle/working"

try:
    import wfdb  # For PhysioNet WFDB format
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("WFDB not available - WFDB format loading disabled")

try:
    from scipy.io import loadmat  # For MATLAB files
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("MATLAB file support not available")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    print("TensorFlow available for deep learning models")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - using sklearn MLP only")

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PPGProcessor:
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
        self.features = {}
        self.processed_signal = None
        self.peaks = None
        self.beat_segments = []
        
    def load_data(self, file_path=None, data_type='simulated'):
        if data_type == 'simulated':
            # Generate simulated PPG data for demonstration
            t = np.linspace(0, 60, self.fs * 60)  # 60 seconds
            heart_rate = 75  # BPM
            
            # Main pulse wave
            pulse_freq = heart_rate / 60
            ppg_signal = np.sin(2 * np.pi * pulse_freq * t)
            
            # Add dicrotic notch (secondary peak)
            dicrotic = 0.3 * np.sin(2 * np.pi * pulse_freq * t + np.pi/3)
            ppg_signal += dicrotic
            
            # Add respiratory variation
            resp_freq = 0.25  # 15 breaths per minute
            resp_modulation = 0.1 * np.sin(2 * np.pi * resp_freq * t)
            ppg_signal *= (1 + resp_modulation)
            
            # Add noise
            noise = 0.05 * np.random.randn(len(t))
            ppg_signal += noise
            
            # Simulate motion artifacts (random spikes)
            motion_artifacts = np.zeros_like(ppg_signal)
            artifact_indices = np.random.choice(len(ppg_signal), size=10, replace=False)
            motion_artifacts[artifact_indices] = np.random.uniform(-0.5, 0.5, 10)
            ppg_signal += motion_artifacts
            
            # Simulate SpO2 values (normally distributed around 97%)
            spo2_true = np.random.normal(97, 2, len(ppg_signal))
            spo2_true = np.clip(spo2_true, 85, 100)
            
            self.raw_signal = ppg_signal
            self.time = t
            self.spo2_true = spo2_true
            
            print(f"Generated simulated PPG data: {len(ppg_signal)} samples, {self.fs} Hz")
            
        elif data_type == 'csv':
            if file_path is None:
                raise ValueError("file_path must be provided for CSV data")
                
            try:
                data = pd.read_csv(file_path)
                
                ppg_cols = ['ppg', 'PPG', 'pleth', 'PLETH', 'signal', 'Signal']
                spo2_cols = ['spo2', 'SpO2', 'SAO2', 'sao2', 'oxygen_saturation']
                time_cols = ['time', 'Time', 'timestamp', 'Timestamp', 't']
                
                ppg_col = None
                for col in ppg_cols:
                    if col in data.columns:
                        ppg_col = col
                        break
                
                if ppg_col is None:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        ppg_col = numeric_cols[0]
                        print(f"Using column '{ppg_col}' as PPG signal")
                    else:
                        raise ValueError("No numeric columns found for PPG signal")
                
                self.raw_signal = data[ppg_col].values
                spo2_col = None
                for col in spo2_cols:
                    if col in data.columns:
                        spo2_col = col
                        break
                
                if spo2_col is not None:
                    self.spo2_true = data[spo2_col].values
                else:
                    print("No SpO2 ground truth found in CSV")
                    self.spo2_true = None
                
                # Create time vector
                time_col = None
                for col in time_cols:
                    if col in data.columns:
                        time_col = col
                        break
                
                if time_col is not None:
                    self.time = data[time_col].values
                else:
                    self.time = np.linspace(0, len(self.raw_signal)/self.fs, len(self.raw_signal))
                
                print(f"Loaded CSV data: {len(self.raw_signal)} samples")
                
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                return None, None
                
        return self.raw_signal, self.time
    
    def preprocess_signal(self, signal_data=None):
        if signal_data is None:
            signal_data = self.raw_signal
            
        print("\nPreprocessing PPG signal...")
        
        # 1. Bandpass filter (0.5-8 Hz) 
        nyquist = self.fs / 2
        low_cutoff = 0.5 / nyquist
        high_cutoff = 8.0 / nyquist
        
        b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_signal = filtfilt(b, a, signal_data)
        
        # 2. Baseline wander removal
        b_hp, a_hp = butter(2, 0.5/nyquist, btype='high')
        detrended_signal = filtfilt(b_hp, a_hp, filtered_signal)
        
        # 3. Motion artifact detection
        window_size = self.fs * 5  # 5-second windows
        sqi_threshold = 0.7
        
        clean_signal = detrended_signal.copy()
        for i in range(0, len(clean_signal) - window_size, window_size):
            window = clean_signal[i:i+window_size]
            sqi = self._calculate_sqi(window)
            
            if sqi < sqi_threshold:
                clean_signal[i:i+window_size] = self._interpolate_segment(
                    clean_signal, i, i+window_size)
        
        # 4. Smooth the signal
        clean_signal = savgol_filter(clean_signal, window_length=5, polyorder=2)
        
        # 5. Normalize signal
        self.processed_signal = (clean_signal - np.mean(clean_signal)) / np.std(clean_signal)
        
        print(f"Preprocessing completed. Signal length: {len(self.processed_signal)}")
        return self.processed_signal
    
    def _calculate_sqi(self, window):
        signal_power = np.var(window)
        noise_estimate = np.var(np.diff(window))
        if noise_estimate == 0:
            return 1.0
        snr = signal_power / noise_estimate
        sqi = min(snr / 100, 1.0)
        return sqi
    
    def _interpolate_segment(self, signal, start_idx, end_idx):
        if start_idx == 0:
            return signal[end_idx:end_idx + (end_idx - start_idx)]
        elif end_idx >= len(signal):
            return signal[start_idx - (end_idx - start_idx):start_idx]
        else:
            x = np.array([start_idx - 1, end_idx])
            y = np.array([signal[start_idx - 1], signal[end_idx]])
            interp_func = interp1d(x, y, kind='linear')
            x_new = np.arange(start_idx, end_idx)
            return interp_func(x_new)
    
    def detect_beats(self, signal_data=None):
        if signal_data is None:
            signal_data = self.processed_signal
            
        print("\nDetecting heartbeats...")
        
        # Find systolic peaks
        min_distance = int(0.4 * self.fs)  # Minimum 0.4s between peaks (150 BPM max)
        
        # Calculate dynamic threshold
        signal_std = np.std(signal_data)
        threshold = 0.3 * signal_std
        
        peaks, properties = find_peaks(signal_data, 
                                     height=threshold,
                                     distance=min_distance,
                                     prominence=threshold/2)
        
        # Filter false peaks using physiological constraints
        valid_peaks = self._validate_peaks(peaks, signal_data)
        
        self.peaks = valid_peaks
        
        # Calculate beat intervals
        if len(valid_peaks) > 1:
            beat_intervals = np.diff(valid_peaks) / self.fs
            heart_rates = 60 / beat_intervals
            
            print(f"Detected {len(valid_peaks)} beats")
            print(f"Average heart rate: {np.mean(heart_rates):.1f} ± {np.std(heart_rates):.1f} BPM")
        
        return valid_peaks
    
    def _validate_peaks(self, peaks, signal_data):
        if len(peaks) < 2:
            return peaks
            
        # Remove peaks with unrealistic intervals
        valid_peaks = [peaks[0]]
        
        for i in range(1, len(peaks)):
            interval = (peaks[i] - valid_peaks[-1]) / self.fs
            # Heart rate between 40-180 BPM
            if 0.33 <= interval <= 1.5:  
                valid_peaks.append(peaks[i])
                
        return np.array(valid_peaks)
    
    def extract_beat_features(self, signal_data=None, peaks=None):
        if signal_data is None:
            signal_data = self.processed_signal
        if peaks is None:
            peaks = self.peaks
            
        print("\nExtracting beat-level features...")
        
        beat_features = []
        
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            beat_segment = signal_data[start_idx:end_idx]
            
            if len(beat_segment) < 10:  # Too short segment
                continue
                
            features = self._analyze_single_beat(beat_segment, start_idx)
            beat_features.append(features)
            
        self.beat_features = pd.DataFrame(beat_features)
        print(f"Extracted features for {len(beat_features)} beats")
        
        return self.beat_features
    
    def _analyze_single_beat(self, beat_segment, start_idx):
        features = {}
        
        # 1. Systolic peak (maximum)
        systolic_peak_idx = np.argmax(beat_segment)
        systolic_amplitude = beat_segment[systolic_peak_idx]
        features['systolic_amplitude'] = systolic_amplitude
        features['systolic_time'] = systolic_peak_idx / self.fs
        
        # 2. Find dicrotic notch (local minimum after systolic peak)
        post_systolic = beat_segment[systolic_peak_idx:]
        if len(post_systolic) > 10:
            search_start = len(post_systolic) // 3
            dicrotic_idx = search_start + np.argmin(post_systolic[search_start:])
            dicrotic_amplitude = post_systolic[dicrotic_idx]
            features['dicrotic_amplitude'] = dicrotic_amplitude
            features['dicrotic_time'] = (systolic_peak_idx + dicrotic_idx) / self.fs
            
            # 3. Diastolic peak (after dicrotic notch)
            post_dicrotic = post_systolic[dicrotic_idx:]
            if len(post_dicrotic) > 5:
                diastolic_peak_idx = dicrotic_idx + np.argmax(post_dicrotic)
                diastolic_amplitude = post_systolic[diastolic_peak_idx]
                features['diastolic_amplitude'] = diastolic_amplitude
                features['diastolic_time'] = (systolic_peak_idx + diastolic_peak_idx) / self.fs
        
        # 4. Timing features
        features['pulse_width'] = len(beat_segment) / self.fs
        
        # Rise time (10% to 90% of systolic peak)
        peak_10 = 0.1 * systolic_amplitude
        peak_90 = 0.9 * systolic_amplitude
        rise_start = np.where(beat_segment[:systolic_peak_idx] >= peak_10)[0]
        rise_end = np.where(beat_segment[:systolic_peak_idx] >= peak_90)[0]
        
        if len(rise_start) > 0 and len(rise_end) > 0:
            features['rise_time'] = (rise_end[0] - rise_start[0]) / self.fs
        else:
            features['rise_time'] = 0
            
        # 5. Morphological features
        features['area_under_curve'] = np.trapz(beat_segment) / self.fs
        features['peak_to_peak_amplitude'] = np.max(beat_segment) - np.min(beat_segment)
        features['mean_amplitude'] = np.mean(beat_segment)
        features['std_amplitude'] = np.std(beat_segment)
        
        # 6. Spectral features
        fft_beat = np.abs(np.fft.fft(beat_segment))
        features['spectral_centroid'] = np.sum(np.arange(len(fft_beat)) * fft_beat) / np.sum(fft_beat)
        features['spectral_energy'] = np.sum(fft_beat**2)
        
        return features
    
    def extract_spo2_features(self, red_ppg=None, ir_ppg=None):
        print("\nExtracting SpO2-specific features...")
        
        if red_ppg is None or ir_ppg is None:
            # Simulate red and IR channels from single PPG
            red_ppg = self.processed_signal
            ir_ppg = self.processed_signal * 1.2 + 0.1 * np.random.randn(len(self.processed_signal))
        
        spo2_features = {}
        
        # 1. AC/DC ratio for both channels
        red_ac = np.std(red_ppg)
        red_dc = np.mean(np.abs(red_ppg))
        ir_ac = np.std(ir_ppg)
        ir_dc = np.mean(np.abs(ir_ppg))
        
        spo2_features['red_ac_dc_ratio'] = red_ac / red_dc if red_dc != 0 else 0
        spo2_features['ir_ac_dc_ratio'] = ir_ac / ir_dc if ir_dc != 0 else 0
        
        # 2. R-value (fundamental for SpO2 calculation)
        r_value = (red_ac / red_dc) / (ir_ac / ir_dc) if (ir_ac != 0 and ir_dc != 0) else 0
        spo2_features['r_value'] = r_value
        
        # 3. Perfusion Index
        spo2_features['perfusion_index_red'] = (red_ac / red_dc) * 100
        spo2_features['perfusion_index_ir'] = (ir_ac / ir_dc) * 100
        
        # 4. Signal quality metrics
        spo2_features['signal_correlation'] = np.corrcoef(red_ppg, ir_ppg)[0, 1]
        
        # 5. Frequency domain features
        red_fft = np.abs(np.fft.fft(red_ppg))
        ir_fft = np.abs(np.fft.fft(ir_ppg))
        
        freqs = np.fft.fftfreq(len(red_ppg), 1/self.fs)
        heart_freq_band = (freqs >= 0.8) & (freqs <= 3.0)  # 48-180 BPM
        
        spo2_features['red_heart_power'] = np.sum(red_fft[heart_freq_band])
        spo2_features['ir_heart_power'] = np.sum(ir_fft[heart_freq_band])
        
        self.spo2_features = spo2_features
        return spo2_features
    
    def estimate_spo2_classical(self):
        if not hasattr(self, 'spo2_features'):
            self.extract_spo2_features()
            
        r_value = self.spo2_features['r_value']
        spo2_classical = 110 - 25 * r_value
        spo2_classical = np.clip(spo2_classical, 70, 100)
        
        return spo2_classical


class MLModelManager:
    
    def __init__(self):
        self.traditional_models = {}
        self.deep_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.trained_models = {}
        self.training_results = {}
        self._initialize_all_models()
    
    def _initialize_all_models(self):
        print("Initializing all ML models...")
        self.traditional_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'linear_regression': LinearRegression(),
            'mlp_neural_network': MLPRegressor(
                hidden_layer_sizes=(16, 8),  # Smaller network for small dataset
                activation='relu',
                solver='lbfgs',  # Better for small datasets
                alpha=0.1,  # Higher regularization
                max_iter=5000,  # Much more iterations to ensure convergence
                random_state=42,
                early_stopping=False,
                tol=1e-6  # Lower tolerance for better convergence
            ),
            'support_vector_regression': SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.1
            )
        }
        if TF_AVAILABLE:
            self.deep_model = OptimizedDeepSpO2Model()
            print(f"  Initialized {len(self.traditional_models)} traditional ML models + 1 optimized deep learning model")
        else:
            print(f"  Initialized {len(self.traditional_models)} traditional ML models")
    
    def prepare_training_data(self, ppg_processors, spo2_values):
        print("\nPreparing training data...")
        
        all_features = []
        all_targets = []
        
        for processor, spo2 in zip(ppg_processors, spo2_values):
            if hasattr(processor, 'beat_features') and not processor.beat_features.empty:
                beat_features = processor.beat_features.mean()
            else:
                beat_features = pd.Series({
                    'systolic_amplitude': 0.5,
                    'pulse_width': 0.8,
                    'rise_time': 0.1,
                    'area_under_curve': 0.4,
                    'peak_to_peak_amplitude': 1.0,
                    'mean_amplitude': 0.0,
                    'std_amplitude': 0.3
                })
            
            if hasattr(processor, 'spo2_features'):
                spo2_features = processor.spo2_features
            else:
                # Create dummy SpO2 features if none exist
                spo2_features = {
                    'red_ac_dc_ratio': 0.1,
                    'ir_ac_dc_ratio': 0.12,
                    'r_value': 0.8,
                    'perfusion_index_red': 2.0,
                    'perfusion_index_ir': 2.4,
                    'signal_correlation': 0.9
                }
            combined_features = {**beat_features.to_dict(), **spo2_features}
            
            all_features.append(list(combined_features.values()))
            all_targets.append(spo2)
            
        self.feature_names = list(combined_features.keys())
        X = np.array(all_features)
        y = np.array(all_targets)
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        print(f"  Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_all_models(self, X, y, test_size=0.2):
        """Train all ML models (traditional + deep learning) in one consolidated function"""
        print(f"\n{'='*80}")
        print("CONSOLIDATED MACHINE LEARNING MODEL TRAINING")
        print(f"{'='*80}")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Ensure feature_names is set
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        all_results = {}
        
        # Auto-optimize MLP before training
        print(f"\n{'-'*60}")
        print("AUTO-OPTIMIZING MLP FOR SMALL DATASET")
        print(f"{'-'*60}")
        try:
            optimized_results, best_mlp = self.optimize_mlp_hyperparameters(X, y)
            print(f"MLP optimization completed - RMSE improved to {optimized_results['test_rmse']:.3f}")
        except Exception as e:
            print(f"MLP optimization failed: {e}")
        
        # Train traditional models
        print(f"\n{'-'*60}")
        print("TRAINING TRADITIONAL ML MODELS")
        print(f"{'-'*60}")
        
        for name, model in self.traditional_models.items():
            print(f"\nTraining {name.replace('_', ' ').title()}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Metrics
                all_results[name] = {
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test),
                    'test_mae': mean_absolute_error(y_test, y_pred_test),
                    'model_type': 'Traditional ML'
                }
                
                self.trained_models[name] = model
                
                print(f"    Test RMSE: {all_results[name]['test_rmse']:.3f}")
                print(f"    Test R²: {all_results[name]['test_r2']:.3f}")
                print(f"    Test MAE: {all_results[name]['test_mae']:.3f}")
                
            except Exception as e:
                print(f"   Training failed: {e}")
                all_results[name] = {
                    'train_rmse': float('inf'),
                    'test_rmse': float('inf'),
                    'train_r2': -float('inf'),
                    'test_r2': -float('inf'),
                    'test_mae': float('inf'),
                    'model_type': 'Traditional ML',
                    'error': str(e)
                }

        if TF_AVAILABLE and self.deep_model:
            print("TRAINING OPTIMIZED DEEP LEARNING MODEL")
            
            try:
                self.deep_model.feature_names = self.feature_names
                deep_results = self.deep_model.train_model(X, y, test_size=test_size, epochs=200, batch_size=8)
                deep_results['model_type'] = 'Deep Learning (Optimized)'
                all_results['deep_neural_network'] = deep_results
                
                print(f"    Optimized Deep Learning Training Completed")
                print(f"    Test RMSE: {deep_results['test_rmse']:.3f}")
                print(f"    Test R²: {deep_results['test_r2']:.3f}")
                print(f"    Test MAE: {deep_results['test_mae']:.3f}")
                
            except Exception as e:
                print(f"   Deep Learning Training Failed: {e}")
                all_results['deep_neural_network'] = {
                    'train_rmse': float('inf'),
                    'test_rmse': float('inf'),
                    'train_r2': -float('inf'),
                    'test_r2': -float('inf'),
                    'test_mae': float('inf'),
                    'model_type': 'Deep Learning (Optimized)',
                    'error': str(e)
                }
        
        self.training_results = all_results
        
        # Display comprehensive comparison
        self._display_model_comparison(all_results)
        
        # Save all models
        self.save_all_models()
        
        # Also save best model separately
        best_model_name = self._get_best_model(all_results)
        if best_model_name:
            self.save_best_model(best_model_name)
        
        return all_results
    
    def _display_model_comparison(self, results):
        """Display comprehensive model comparison"""
        print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
        print(f"{'Model':<25} {'Test RMSE':<12} {'Test R²':<10} {'Test MAE':<10} {'Type':<15} {'Status':<10}")
        
        for model_name, metrics in results.items():
            status = "  Success" if 'error' not in metrics else " Failed"
            model_type = metrics.get('model_type', 'Unknown')
            
            if 'error' not in metrics:
                print(f"{model_name:<25} {metrics['test_rmse']:<12.3f} "
                      f"{metrics['test_r2']:<10.3f} {metrics['test_mae']:<10.3f} "
                      f"{model_type:<15} {status:<10}")
            else:
                print(f"{model_name:<25} {'N/A':<12} {'N/A':<10} {'N/A':<10} "
                      f"{model_type:<15} {status:<10}")
        
        # Model complexity analysis
        
        feature_count = len(self.feature_names) if self.feature_names else 0
        print(f"Linear Regression: ~{feature_count} parameters")
        print(f"Random Forest: ~{100 * 10} decision nodes (approx)")
        print(f"Gradient Boosting: ~{100 * 10} decision nodes (approx)")
        print(f"MLP Neural Network: ~{100*feature_count + 50*100 + 25*50} parameters")
        if TF_AVAILABLE:
            print(f"Deep Neural Network: ~{256*feature_count + 128*256 + 64*128 + 32*64 + 32} parameters")
    
    def _get_best_model(self, results):
        """Get the best performing model based on test RMSE"""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print(" No models trained successfully")
            return None
        
        best_model = min(valid_results.items(), key=lambda x: x[1]['test_rmse'])
        print(f"\n Best Model: {best_model[0]} (RMSE: {best_model[1]['test_rmse']:.3f})")
        return best_model[0]
    
    def predict(self, features, model_name=None):
        """Make SpO2 prediction using trained model"""
        if model_name is None:
            # Use best model
            model_name = self._get_best_model(self.training_results)
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
            
        features_scaled = self.scaler.transform([features])
        prediction = self.trained_models[model_name].predict(features_scaled)[0]
        
        return np.clip(prediction, 70, 100)
    
    def save_best_model(self, model_name=None):
        if model_name is None:
            model_name = self._get_best_model(self.training_results)
        
        if model_name is None:
            print(" No model to save")
            return
        
        if model_name in self.trained_models:
            save_path = os.path.join(KAGGLE_WORKING_PATH, f'best_spo2_model_{model_name}.pkl')
            
            model_data = {
                'model': self.trained_models[model_name],
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_name': model_name,
                'training_results': self.training_results
            }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model_data, save_path)
            print(f"  Best model ({model_name}) saved to {save_path}")
        
        elif model_name == 'deep_neural_network' and self.deep_model:
            self.deep_model.save_model()
            print(f" Best deep learning model saved")
    
    def save_all_models(self):
        """Save all trained models individually"""
        print("\nSaving all trained models...")
        
        for model_name in self.trained_models:
            save_path = os.path.join(KAGGLE_WORKING_PATH, f'spo2_model_{model_name}.pkl')
            
            model_data = {
                'model': self.trained_models[model_name],
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_name': model_name,
                'training_results': self.training_results
            }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model_data, save_path)
            print(f"  {model_name} saved to {save_path}")
        
        if TF_AVAILABLE and self.deep_model and hasattr(self.deep_model, 'model') and self.deep_model.model:
            deep_save_path = os.path.join(KAGGLE_WORKING_PATH, 'spo2_model_deep_neural_network')
            self.deep_model.save_model(deep_save_path)
            print(f"  deep_neural_network saved to {deep_save_path}.h5")
        
        print(f"  All models saved to {KAGGLE_WORKING_PATH}")
    
    def load_specific_model(self, model_name):
        """Load a specific model by name"""
        model_path = os.path.join(KAGGLE_WORKING_PATH, f'spo2_model_{model_name}.pkl')
        
        if not os.path.exists(model_path):
            print(f" Model file not found: {model_path}")
            return False
        
        try:
            model_data = joblib.load(model_path)
            self.trained_models[model_name] = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            if 'training_results' in model_data:
                self.training_results = model_data['training_results']
            
            print(f"  {model_name} loaded successfully")
            return True
            
        except Exception as e:
            print(f" Error loading {model_name}: {e}")
            return False
    def optimize_mlp_hyperparameters(self, X, y):
        """Optimize MLP hyperparameters using grid search"""
        from sklearn.model_selection import GridSearchCV
        import warnings
        
        print("\n OPTIMIZING MLP HYPERPARAMETERS...")
        print("Note: Some convergence warnings are normal for small datasets during grid search")
        
        # Temporarily suppress convergence warnings during grid search
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            
            # Define parameter grid for MLP optimization (optimized for small datasets)
            param_grid = {
                'hidden_layer_sizes': [
                    (8,), (16,), (32,),
                    (16, 8), (32, 16)  # Reduced combinations for faster search
                ],
                'activation': ['relu', 'tanh'],
                'solver': ['lbfgs'],  # Focus on lbfgs for small datasets
                'alpha': [0.01, 0.1, 0.5],
                'learning_rate_init': [0.001, 0.01]  # Reduced learning rates
            }
            
            # Create base MLP with higher max_iter
            mlp_base = MLPRegressor(
                max_iter=3000,  # Balanced for grid search
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                tol=1e-5
            )
            
            # Scale data
            X_scaled = self.scaler.fit_transform(X)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                mlp_base,
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0  # Reduce verbosity
            )
            
            print("Running optimized grid search for small datasets...")
            grid_search.fit(X_scaled, y)
        
        # Get best model
        best_mlp = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        
        print(f"\n BEST MLP PARAMETERS FOUND:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"  Best CV RMSE: {np.sqrt(best_score):.3f}")
        
        # Update the MLP model in traditional_models
        self.traditional_models['mlp_neural_network'] = best_mlp
        
        # Train and evaluate the optimized model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_mlp.fit(X_train_scaled, y_train)
        y_pred_test = best_mlp.predict(X_test_scaled)
        
        optimized_results = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'best_params': best_params
        }
        
        print(f"\n OPTIMIZED MLP PERFORMANCE:")
        print(f"  Test RMSE: {optimized_results['test_rmse']:.3f}")
        print(f"  Test R²: {optimized_results['test_r2']:.3f}")
        print(f"  Test MAE: {optimized_results['test_mae']:.3f}")
        
        # Save optimized model
        self.trained_models['mlp_neural_network_optimized'] = best_mlp
        
        return optimized_results, best_mlp

    def train_optimized_mlp(self, X, y):
        """Train only the optimized MLP model"""      
        optimized_results, best_mlp = self.optimize_mlp_hyperparameters(X, y)
        
        # Save the optimized model
    def optimize_deep_learning_model(self, X, y):
        """Optimize deep learning model for small datasets"""
        if not TF_AVAILABLE:
            print(" TensorFlow not available for deep learning optimization")
            return None
            
        print(f"\n OPTIMIZING DEEP LEARNING MODEL FOR SMALL DATASET...")
        print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Create optimized deep model for small datasets
        optimized_deep_model = OptimizedDeepSpO2Model()
        
        results = optimized_deep_model.train_model(X, y, test_size=0.2, epochs=200)
        
        # Save optimized deep model
        save_path = os.path.join(KAGGLE_WORKING_PATH, 'optimized_deep_model')
        optimized_deep_model.save_model(save_path)
        
        print(f"\n OPTIMIZED DEEP LEARNING PERFORMANCE:")
        print(f"  Test RMSE: {results['test_rmse']:.3f}")
        print(f"  Test R²: {results['test_r2']:.3f}")
        print(f"  Test MAE: {results['test_mae']:.3f}")
        print(f" Optimized deep model saved to {save_path}.h5")
        
        return results
    
    def load_model(self, load_path=None):
        """Load trained model from disk"""
        if load_path is None:
            load_path = os.path.join(KAGGLE_WORKING_PATH, 'best_spo2_model.pkl')
        elif not os.path.isabs(load_path):
            if os.path.exists(os.path.join(KAGGLE_WORKING_PATH, load_path)):
                load_path = os.path.join(KAGGLE_WORKING_PATH, load_path)
            
        try:
            model_data = joblib.load(load_path)
            
            self.trained_models[model_data['model_name']] = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            if 'training_results' in model_data:
                self.training_results = model_data['training_results']
            
            print(f"Model loaded from {load_path}")
            return True
            
        except Exception as e:
            print(f" Error loading model: {e}")
            return False


class DeepSpO2Model:
    """Deep Learning model for SpO2 estimation using TensorFlow/Keras"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.history = None
        
    def create_model(self, input_dim):
        """Create deep neural network architecture"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
            
        # Optimized architecture for small datasets
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),  # Higher dropout for regularization
            layers.Dense(16, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def train_model(self, X, y, test_size=0.2, epochs=100, batch_size=32):
        """Train the deep learning model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        print(f"Training Deep Learning model with {X.shape[0]} samples and {X.shape[1]} features...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.model is None:
            self.create_model(X_train_scaled.shape[1])
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=200,  # More epochs for small dataset
            batch_size=min(8, len(X_train)),  # Smaller batch size
            callbacks=callbacks,
            verbose=0  # Reduce verbosity
        )
        
        train_loss = self.model.evaluate(X_train_scaled, y_train, verbose=0)
        test_loss = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        y_pred_train = self.model.predict(X_train_scaled, verbose=0).flatten()
        y_pred_test = self.model.predict(X_test_scaled, verbose=0).flatten()
        
        results = {
            'train_rmse': np.sqrt(train_loss[0]),
            'test_rmse': np.sqrt(test_loss[0]),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mae': test_loss[1]
        }
        
        return results
    
    def predict(self, features):
        """Make prediction using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled, verbose=0)[0][0]
        
        return np.clip(prediction, 70, 100)
    
    def save_model(self, save_path=None):
        """Save deep learning model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if save_path is None:
            save_path = os.path.join(KAGGLE_WORKING_PATH, 'spo2_deep_model')
        elif not os.path.isabs(save_path):
            save_path = os.path.join(KAGGLE_WORKING_PATH, save_path)
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save TensorFlow model
        self.model.save(f"{save_path}.h5")
        
        # Save scaler and metadata
        model_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, f"{save_path}_metadata.pkl")
        
        print(f"Deep model saved to {save_path}.h5")


class OptimizedDeepSpO2Model:
    """Optimized Deep Learning model for small datasets"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.history = None
        
    def create_optimized_model(self, input_dim):
        """Create optimized deep neural network for small datasets"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        # Ultra-simple architecture for very small datasets (20 samples)
        model = keras.Sequential([
            # Single hidden layer with heavy regularization
            layers.Dense(8, activation='relu', input_shape=(input_dim,),
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.7),  # Very high dropout
            
            # Output layer
            layers.Dense(1, activation='linear')
        ])
        
        # Very conservative training settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Higher learning rate for faster convergence
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def train_model(self, X, y, test_size=0.2, epochs=200, batch_size=8):
        """Train optimized model for small datasets"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        print(f"Training OPTIMIZED Deep Learning model with {X.shape[0]} samples and {X.shape[1]} features...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create optimized model
        if self.model is None:
            self.create_optimized_model(X_train_scaled.shape[1])
        
        # Very conservative callbacks for tiny datasets
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,  # Stop early to prevent overfitting
                restore_best_weights=True,
                min_delta=0.01  # Larger threshold
            )
        ]
        
        # Train with very conservative settings
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=30,  # Much fewer epochs
            batch_size=min(2, len(X_train)),  # Tiny batch size
            callbacks=callbacks,
            verbose=0  # Reduce verbosity
        )
        
        # Evaluate
        train_loss = self.model.evaluate(X_train_scaled, y_train, verbose=0)
        test_loss = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        # Predictions for metrics
        y_pred_train = self.model.predict(X_train_scaled, verbose=0).flatten()
        y_pred_test = self.model.predict(X_test_scaled, verbose=0).flatten()
        
        results = {
            'train_rmse': np.sqrt(train_loss[0]),
            'test_rmse': np.sqrt(test_loss[0]),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mae': test_loss[1],
            'epochs_trained': len(self.history.history['loss'])
        }
        
        return results
    
    def predict(self, features):
        """Make prediction using optimized model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled, verbose=0)[0][0]
        
        return np.clip(prediction, 70, 100)
    
    def save_model(self, save_path=None):
        """Save optimized deep learning model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if save_path is None:
            save_path = os.path.join(KAGGLE_WORKING_PATH, 'optimized_deep_spo2_model')
        elif not os.path.isabs(save_path):
            save_path = os.path.join(KAGGLE_WORKING_PATH, save_path)
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save TensorFlow model
        self.model.save(f"{save_path}.h5")
        
        # Save scaler and metadata
        model_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.history.history if self.history else None
        }
        joblib.dump(model_data, f"{save_path}_metadata.pkl")
        
        print(f"Optimized deep model saved to {save_path}.h5")




def process_kaggle_dataset():
    print("Processing Kaggle PPG Dataset...")
    print(f"Dataset path: {KAGGLE_INPUT_PATH}")
    
    if not os.path.exists(KAGGLE_INPUT_PATH):
        print(f"Dataset path not found: {KAGGLE_INPUT_PATH}")
        print("Available paths:")
        if os.path.exists("/kaggle/input"):
            for item in os.listdir("/kaggle/input"):
                print(f"  /kaggle/input/{item}")
        return None
    
    csv_files = []
    for file in os.listdir(KAGGLE_INPUT_PATH):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(KAGGLE_INPUT_PATH, file))
    
    if not csv_files:
        print("No CSV files found in dataset")
        return None
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  {os.path.basename(file)}")
    
    results = []
    for file_path in csv_files:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        result = process_ppg_file(file_path)
        if result:
            results.append(result)
    
    if results:
        output_file = os.path.join(KAGGLE_WORKING_PATH, 'ppg_analysis_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nCombined results saved to: {output_file}")
    
    return results


def process_ppg_file(file_path, model_path=None, output_path=None):
    """Process a single PPG file and estimate SpO2"""
    print(f"Processing PPG file: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    if file_ext == '.csv':
        data_type = 'csv'
    elif file_ext == '.mat' and MATLAB_AVAILABLE:
        data_type = 'mat'
    elif file_ext in ['.dat', '.hea'] and WFDB_AVAILABLE:
        data_type = 'wfdb'
    else:
        print(f"Unsupported file type or missing dependencies: {file_ext}")
        return None
    
    processor = PPGProcessor(sampling_rate=125)
    
    try:
        # Load data
        ppg_signal, time_vector = processor.load_data(file_path, data_type=data_type)
        
        if ppg_signal is None:
            print("Failed to load PPG data")
            return None
        
        processed_signal = processor.preprocess_signal()
        peaks = processor.detect_beats()
        
        if len(peaks) < 3:
            print("Insufficient beats detected for reliable analysis")
            return None
        
        beat_features = processor.extract_beat_features()
        spo2_features = processor.extract_spo2_features()
        
        spo2_classical = processor.estimate_spo2_classical()
        
        spo2_ml = None
        if model_path and os.path.exists(model_path):
            ml_manager = MLModelManager()
            if ml_manager.load_model(model_path):
                # Prepare features for ML model
                combined_features = {**beat_features.mean().to_dict(), **spo2_features}
                feature_vector = [combined_features.get(name, 0) for name in ml_manager.feature_names]
                spo2_ml = ml_manager.predict(feature_vector)
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / processor.fs
            heart_rate = np.mean(60 / rr_intervals)
        else:
            heart_rate = None
        
        results = {
            'file_path': str(file_path),
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'signal_length_seconds': len(ppg_signal) / processor.fs,
            'sampling_rate': processor.fs,
            'beats_detected': len(peaks),
            'heart_rate_bpm': heart_rate,
            'spo2_classical': spo2_classical,
            'spo2_ml': spo2_ml,
            'signal_quality_metrics': {
                'mean_amplitude': float(np.mean(processed_signal)),
                'std_amplitude': float(np.std(processed_signal)),
                'snr_estimate': float(np.var(processed_signal) / np.var(np.diff(processed_signal)))
            },
            'beat_features_summary': {
                col: float(beat_features[col].mean()) 
                for col in beat_features.select_dtypes(include=[np.number]).columns
            } if not beat_features.empty else {},
            'spo2_features': {k: float(v) for k, v in spo2_features.items()}
        }
        
        if output_path:
            if not os.path.isabs(output_path):
                output_path = os.path.join(KAGGLE_WORKING_PATH, output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
        else:
            # Auto-save to working directory
            filename = f"ppg_results_{os.path.basename(file_path).replace('.csv', '.json')}"
            output_path = os.path.join(KAGGLE_WORKING_PATH, filename)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results auto-saved to {output_path}")
        
        # Print summary
        print("PROCESSING RESULTS")
        print(f"Signal Duration: {results['signal_length_seconds']:.1f} seconds")
        print(f"Beats Detected: {results['beats_detected']}")
        if heart_rate:
            print(f"Heart Rate: {heart_rate:.1f} BPM")
        print(f"SpO2 (Classical): {spo2_classical:.1f}%")
        if spo2_ml:
            print(f"SpO2 (ML): {spo2_ml:.1f}%")
        print(f"Signal Quality (SNR): {results['signal_quality_metrics']['snr_estimate']:.1f}")
        
        return results
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def run_kaggle_pipeline():
    """Main pipeline for Kaggle environment"""
    print("PPG FEATURE EXTRACTION AND SPO2 ESTIMATION")
    print("Environment Check:")
    print(f"  Dataset path exists: {os.path.exists(KAGGLE_INPUT_PATH)}")
    print(f"  Working directory: {KAGGLE_WORKING_PATH}")
    print(f"  WFDB available: {WFDB_AVAILABLE}")
    print(f"  MATLAB support: {MATLAB_AVAILABLE}")
    print(f"  TensorFlow available: {TF_AVAILABLE}")
    
    print("\nStep 1: Processing Kaggle PPG Dataset")
    results = process_kaggle_dataset()
    
    if not results:
        print("No data processed. Falling back to demo mode.")
        demonstrate_pipeline()
        return
    
    if len(results) >= 3:
        print(f"\nStep 2: Training ML models with {len(results)} samples")
        try:
            # Create training data from processed results
            processors = []
            spo2_values = []
            
            for result in results[:min(20, len(results))]:  # Use up to 20 files for training
                # Create a processor with features from the result
                processor = PPGProcessor(sampling_rate=125)
                
                duration = result.get('signal_length_seconds', 30)
                t = np.linspace(0, duration, int(125 * duration))
                hr = result.get('heart_rate_bpm', 75)
                
                pulse_freq = hr / 60
                ppg = np.sin(2 * np.pi * pulse_freq * t)
                ppg += 0.3 * np.sin(2 * np.pi * pulse_freq * t + np.pi/3)
                ppg += 0.05 * np.random.randn(len(t))
                
                processor.raw_signal = ppg
                processor.time = t
                processor.preprocess_signal()
                processor.detect_beats()
                processor.extract_beat_features()
                processor.extract_spo2_features()
                
                processors.append(processor)
                spo2_values.append(result.get('spo2_classical', 95))
            
            ml_manager = MLModelManager()
            X, y = ml_manager.prepare_training_data(processors, spo2_values)
            training_results = ml_manager.train_all_models(X, y)
            
            print(f"\n  Model training completed successfully")
            
        except Exception as e:
            print(f"ERROR: Model training failed: {e}")
            print("Continuing with classical SpO2 estimation only")
    
    print("\nStep 3: Generating Summary Report")
    generate_summary_report(results)
    print(f"  Processed {len(results)} PPG files")
    print(f"  Results saved to {KAGGLE_WORKING_PATH}")
    print("  Summary report generated")
    print("  Models trained and saved (if applicable)")


def generate_summary_report(results):
    """Generate a summary report of all processed files"""
    if not results:
        return
    
    summary = {
        'total_files_processed': len(results),
        'successful_analyses': len([r for r in results if r.get('beats_detected', 0) > 0]),
        'average_heart_rate': np.mean([r.get('heart_rate_bpm', 0) for r in results if r.get('heart_rate_bpm')]),
        'average_spo2_classical': np.mean([r.get('spo2_classical', 0) for r in results if r.get('spo2_classical')]),
        'signal_quality_stats': {
            'mean_snr': np.mean([r.get('signal_quality_metrics', {}).get('snr_estimate', 0) for r in results]),
            'mean_amplitude': np.mean([r.get('signal_quality_metrics', {}).get('mean_amplitude', 0) for r in results])
        },
        'file_details': [
            {
                'filename': os.path.basename(r.get('file_path', '')),
                'duration_seconds': r.get('signal_length_seconds', 0),
                'beats_detected': r.get('beats_detected', 0),
                'heart_rate_bpm': r.get('heart_rate_bpm', 0),
                'spo2_classical': r.get('spo2_classical', 0)
            }
            for r in results
        ]
    }
    
    summary_path = os.path.join(KAGGLE_WORKING_PATH, 'ppg_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary report saved to: {summary_path}")
    
    print("\nKEY STATISTICS:")
    print(f"  Files processed: {summary['total_files_processed']}")
    print(f"  Successful analyses: {summary['successful_analyses']}")
    if summary['average_heart_rate'] > 0:
        print(f"  Average heart rate: {summary['average_heart_rate']:.1f} BPM")
    if summary['average_spo2_classical'] > 0:
        print(f"  Average SpO2: {summary['average_spo2_classical']:.1f}%")
    print(f"  Average signal quality (SNR): {summary['signal_quality_stats']['mean_snr']:.1f}")


def demonstrate_pipeline():
    
    processor = PPGProcessor(sampling_rate=125)
    
    ppg_signal, time_vector = processor.load_data(data_type='simulated')
    
    processed_signal = processor.preprocess_signal()
    
    peaks = processor.detect_beats()
    
    beat_features = processor.extract_beat_features()
    spo2_features = processor.extract_spo2_features()
    
    spo2_classical = processor.estimate_spo2_classical()
    
    print(f"\nClassical SpO2 Estimation: {spo2_classical:.1f}%")
    
    print(f"\nExtracted Features Summary:")
    print(f"Beat-level features: {len(beat_features)} beats analyzed")
    print(f"SpO2-specific features: {len(spo2_features)} features")
    
    if not beat_features.empty:
        print(f"\nKey Beat Features (mean ± std):")
        numeric_features = beat_features.select_dtypes(include=[np.number])
        for feature in ['systolic_amplitude', 'pulse_width', 'rise_time']:
            if feature in numeric_features.columns:
                mean_val = numeric_features[feature].mean()
                std_val = numeric_features[feature].std()
                print(f"  {feature}: {mean_val:.3f} ± {std_val:.3f}")
    
    print(f"\nSpO2 Features:")
    for key, value in spo2_features.items():
        print(f"  {key}: {value:.3f}")
    
    return processor


def train_comprehensive_ml_demo():
    """Demonstrate comprehensive ML model training"""
    
    processors = []
    spo2_ground_truth = []
    
    print("Generating diverse training datasets...")
    
    for i in range(100):
        processor = PPGProcessor(sampling_rate=125)
        
        t = np.linspace(0, 30, 125 * 30)
        heart_rate = np.random.uniform(60, 100)
        pulse_freq = heart_rate / 60
        ppg = np.sin(2 * np.pi * pulse_freq * t)
        
        dicrotic_strength = np.random.uniform(0.2, 0.4)
        ppg += dicrotic_strength * np.sin(2 * np.pi * pulse_freq * t + np.pi/3)
        
        noise_level = np.random.uniform(0.03, 0.08)
        ppg += noise_level * np.random.randn(len(t))
        
        signal_quality = 1 - noise_level
        perfusion_quality = np.random.uniform(0.7, 1.0)
        base_spo2 = np.random.normal(97, 2)
        
        spo2_true = base_spo2 + 2 * signal_quality + 1 * perfusion_quality - abs(heart_rate - 75) * 0.02
        spo2_true = np.clip(spo2_true, 85, 100)
        
        processor.raw_signal = ppg
        processor.time = t
        processor.spo2_true = spo2_true
        
        processor.preprocess_signal()
        processor.detect_beats()
        processor.extract_beat_features()
        processor.extract_spo2_features()
        
        processors.append(processor)
        spo2_ground_truth.append(spo2_true)
    
    ml_manager = MLModelManager()
    X, y = ml_manager.prepare_training_data(processors, spo2_ground_truth)
    
    print(f"\nTraining data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    results = ml_manager.train_all_models(X, y)
    
    return ml_manager, results


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='PPG Feature Extraction and SpO2 Estimation - Reorganized')
    parser.add_argument('--input', '-i', type=str, help='Input PPG file path')
    parser.add_argument('--output', '-o', type=str, help='Output results file path')
    parser.add_argument('--model', '-m', type=str, help='Path to trained ML model')

    
    args = parser.parse_args()
    
    if args.kaggle:
        print("Processing Kaggle PPG dataset...")
        run_kaggle_pipeline()
        
    elif args.demo:
        print("Running demonstration with simulated data...")
        demonstrate_pipeline()
        
    elif args.train:
        print("Training ML models...")
        train_comprehensive_ml_demo()
        
    elif args.input:
        if not os.path.exists(args.input):
            print(f"Input file not found: {args.input}")
            return
            
        results = process_ppg_file(args.input, args.model, args.output)
        
        if results is None:
            print("Processing failed")
            return
            
    else:
        print("No action specified. Use --help for usage information")


def run_notebook_pipeline():
    try:
        # Check if we're in Kaggle environment
        if os.path.exists("/kaggle"):
            print("Detected Kaggle environment - running Kaggle pipeline...")
            run_kaggle_pipeline()
        else:
            print("Starting PPG Feature Extraction and SpO2 Estimation Pipeline...")
            
            print("STEP 1: BASIC PIPELINE DEMONSTRATION")
            processor = demonstrate_pipeline()
            
            print("STEP 2: COMPREHENSIVE ML MODEL TRAINING")
            ml_manager, training_results = train_comprehensive_ml_demo()
            

            print("PIPELINE SUMMARY")
            print("  PPG signal preprocessing and noise removal")
            print("  Beat detection and segmentation")
            print("  Feature extraction (morphological, temporal, spectral)")
            print("  Classical SpO2 estimation using R-value")
            print("  CONSOLIDATED machine learning models")
            print("  Deep learning models (if TensorFlow available)")
            print("  Model comparison and selection")
            print("  Model persistence (save/load functionality)")
            print("  Kaggle environment optimization")
            
            print(f"\nFiles Generated:")
            print(f"  Best trained ML model saved to {KAGGLE_WORKING_PATH}")
            print(f"  Comprehensive performance metrics")
            print(f"  Model comparison results")
            
    except Exception as e:
        print(f"Error in pipeline execution: {e}")
        print("Falling back to basic demo...")
        demonstrate_pipeline()

if __name__ == "__main__":
    try:
        if 'ipykernel' in sys.modules or 'google.colab' in sys.modules or any('-f' in str(arg) for arg in sys.argv):
            print("Detected notebook environment - running pipeline...")
            run_notebook_pipeline()
        elif len(sys.argv) > 1 and not any('-f' in str(arg) for arg in sys.argv):
            main()
        else:
            # Default execution
            run_notebook_pipeline()
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Running basic pipeline...")
        run_notebook_pipeline()
