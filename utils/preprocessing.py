import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler

def find_header_row_and_sep(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for i, line in enumerate(lines[:20]):
            line_lower = line.lower()
            if 'x' in line_lower and 'y' in line_lower:
                sep = ',' 
                if ';' in line: sep = ';'
                if '\t' in line: sep = '\t'
                return i, sep
        return 0, ',' 
    except:
        return 0, ','

def calculate_metrics(df: pd.DataFrame, metrics: set) -> pd.DataFrame:
    if len(df) < 2: return df
    
    x = df['X'].values
    y = df['Y'].values
    
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    
    if 'T' in df.columns:
        t = df['T'].values
        dt = np.diff(t, prepend=t[0])
    else:
        dt = np.zeros_like(dx)

    dt[dt <= 1e-5] = 0.01 

    distance = np.sqrt(dx**2 + dy**2)
    
    speed = np.divide(distance, dt, out=np.zeros_like(distance), where=dt!=0)
    
    speed[np.isinf(speed)] = 0
    speed[np.isnan(speed)] = 0
    
    acceleration = np.gradient(speed)
    
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = (dx * ddy - dy * ddx)
    den = (dx**2 + dy**2)**1.5
    curvature = np.divide(num, den, out=np.zeros_like(num), where=den!=0)

    dp = np.zeros_like(dx)
    if 'Pressure' in df.columns:
        dp = np.diff(df['Pressure'], prepend=df['Pressure'].iloc[0])

    if 'PressureVelocity' in metrics: df['PressureVelocity'] = dp
    if 'Speed' in metrics: df['Speed'] = speed
    if 'Curvature' in metrics: df['Curvature'] = curvature
    if 'Acceleration' in metrics: df['Acceleration'] = acceleration

    return df.fillna(0)

def load_signature(file_path: str, additional_metrics: set=None) -> pd.DataFrame:
    header_row, sep = find_header_row_and_sep(file_path)
    
    try:
        df = pd.read_csv(file_path, header=header_row, sep=sep, engine='python')
    except:
        return pd.DataFrame()

    df.columns = [str(c).strip().replace('"', '').replace("'", "") for c in df.columns]

    rename_map = {
        'x': 'X', 'y': 'Y', 't': 'T', 'p': 'Pressure', 'pressure': 'Pressure', 
        'prs': 'Pressure', 'button': 'Btn', 'btn': 'Btn', 'stroke': 'Stroke'
    }
    new_cols = {c: rename_map.get(c.lower(), c) for c in df.columns}
    df.rename(columns=new_cols, inplace=True)

    if 'X' not in df.columns or 'Y' not in df.columns:
        return pd.DataFrame()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['X', 'Y'], inplace=True)
    df = df[(df['X'] > 0) | (df['Y'] > 0)]
    
    if df.empty: return pd.DataFrame()

    if 'T' not in df.columns:
        df['T'] = np.linspace(0, 1, len(df))
    if 'Pressure' not in df.columns:
        df['Pressure'] = 100

    if additional_metrics:
        df = calculate_metrics(df, additional_metrics)
        
    return df

def normalize_x_y_pressure(x, y, p) -> tuple:
    if len(x) == 0: return (np.array([]), np.array([]), np.array([]))
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    p = np.array(p).reshape(-1, 1)
    x_norm = MinMaxScaler().fit_transform(x).flatten()
    y_norm = MinMaxScaler().fit_transform(y).flatten()
    p_norm = MinMaxScaler().fit_transform(p).flatten()
    return (x_norm, y_norm, p_norm)

def resample_signature(sig, n_points=100) -> np.array:
    if sig.empty: return np.zeros((n_points, 3))
    sig = sig.copy()
    t_original = np.linspace(0, 1, len(sig))
    t_new = np.linspace(0, 1, n_points)
    x_interp = interpolate.interp1d(t_original, sig['X'], kind='linear')(t_new)
    y_interp = interpolate.interp1d(t_original, sig['Y'], kind='linear')(t_new)
    if 'Pressure' in sig.columns:
        p_vals = sig['Pressure']
    else:
        p_vals = np.zeros_like(sig['X'])
    p_interp = interpolate.interp1d(t_original, p_vals, kind='cubic')(t_new)
    return np.column_stack((x_interp, y_interp, p_interp))