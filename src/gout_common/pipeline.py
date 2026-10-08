"""Pipeline del modelo ganador de la memoria (XGBoost) con el preprocesamiento adentro:
LabelEncoder -> OrdinalEncoder (mismos códigos: orden alfabético), imputación de rezagos con
la media de train, StandardScaler ajustado en train. Se loguea entero en MLflow."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from gout_common.features import CAT_COLS, LAG_COLS, NUM_COLS_1, NUM_COLS_2

SEED = 42
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=SEED, n_jobs=-1, verbosity=0)


def build_pipeline(**xgb_overrides) -> Pipeline:
    pre = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_COLS),
        ('num1', 'passthrough', NUM_COLS_1),
        ('lag', SimpleImputer(strategy='mean'), LAG_COLS),
        ('num2', 'passthrough', NUM_COLS_2),
    ])
    return Pipeline([('pre', pre), ('scaler', StandardScaler()),
                     ('xgb', XGBRegressor(**{**XGB_PARAMS, **xgb_overrides}))])


def postprocess(pred) -> np.ndarray:
    """Igual que la memoria: unidades enteras no negativas."""
    return np.clip(np.round(pred), 0, None).astype(int)
