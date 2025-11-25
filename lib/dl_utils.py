"""
Utilidades para modelos de Deep Learning (NeuralForecast)
Autor: Joaquín Mora
Fecha: 2025-11-25
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def preparar_datos_neuralforecast(df_train: pd.DataFrame, 
                                    df_test: pd.DataFrame,
                                    col_fecha: str = 'idSecuencia',
                                    col_producto: str = 'producto',
                                    col_target: str = 'udsVenta') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convierte datasets al formato requerido por NeuralForecast.
    
    NeuralForecast requiere:
    - Columna 'unique_id': identificador de serie temporal (producto)
    - Columna 'ds': timestamp/fecha
    - Columna 'y': variable objetivo
    - Columnas adicionales: variables exógenas
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Dataset de entrenamiento
    df_test : pd.DataFrame
        Dataset de test
    col_fecha : str
        Nombre de columna con fechas
    col_producto : str
        Nombre de columna con identificador de producto
    col_target : str
        Nombre de columna objetivo
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames train y test en formato NeuralForecast
    """
    # Copiar datasets
    train = df_train.copy()
    test = df_test.copy()
    
    # Renombrar columnas obligatorias
    train = train.rename(columns={
        col_producto: 'unique_id',
        col_fecha: 'ds',
        col_target: 'y'
    })
    
    test = test.rename(columns={
        col_producto: 'unique_id',
        col_fecha: 'ds',
        col_target: 'y'
    })
    
    # Ordenar por unique_id y ds
    train = train.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    test = test.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    
    # Asegurar que 'ds' es datetime
    if not pd.api.types.is_datetime64_any_dtype(train['ds']):
        train['ds'] = pd.to_datetime(train['ds'])
    if not pd.api.types.is_datetime64_any_dtype(test['ds']):
        test['ds'] = pd.to_datetime(test['ds'])
    
    print(f"\n✅ Datos preparados para NeuralForecast:")
    print(f"   Train: {train.shape} | Productos: {train['unique_id'].nunique()}")
    print(f"   Test:  {test.shape} | Productos: {test['unique_id'].nunique()}")
    print(f"   Rango train: {train['ds'].min()} a {train['ds'].max()}")
    print(f"   Rango test:  {test['ds'].min()} a {test['ds'].max()}")
    
    return train, test


def crear_dataset_validacion(df_train: pd.DataFrame, 
                             val_size: float = 0.2,
                             col_fecha: str = 'ds',
                             col_producto: str = 'unique_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset de entrenamiento en train y validación temporal.
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Dataset de entrenamiento completo
    val_size : float
        Proporción de datos para validación (0.2 = 20%)
    col_fecha : str
        Nombre de columna con fechas
    col_producto : str
        Nombre de columna con identificador único
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames train y validación
    """
    # Obtener el punto de corte temporal
    fechas_unicas = sorted(df_train[col_fecha].unique())
    n_fechas = len(fechas_unicas)
    n_val = int(n_fechas * val_size)
    fecha_corte = fechas_unicas[-n_val]
    
    # Dividir
    train = df_train[df_train[col_fecha] < fecha_corte].copy()
    val = df_train[df_train[col_fecha] >= fecha_corte].copy()
    
    print(f"\n✅ División train/validación:")
    print(f"   Train: {train.shape} | {train[col_fecha].min()} a {train[col_fecha].max()}")
    print(f"   Val:   {val.shape} | {val[col_fecha].min()} a {val[col_fecha].max()}")
    
    return train, val


def seleccionar_features_exogenas(df: pd.DataFrame, 
                                   excluir: List[str] = None) -> List[str]:
    """
    Identifica features exógenas (excluye identificadores y target).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    excluir : List[str]
        Columnas adicionales a excluir
    
    Returns:
    --------
    List[str]
        Lista de nombres de features exógenas
    """
    # Columnas obligatorias de NeuralForecast
    base_excluir = ['unique_id', 'ds', 'y']
    
    if excluir:
        base_excluir.extend(excluir)
    
    # Filtrar columnas
    features = [col for col in df.columns if col not in base_excluir]
    
    print(f"\n✅ Features exógenas identificadas: {len(features)}")
    print(f"   {features[:5]}{'...' if len(features) > 5 else ''}")
    
    return features


def reconstruir_predicciones(y_hat: pd.DataFrame,
                             df_test_original: pd.DataFrame,
                             modelo_name: str,
                             col_producto: str = 'producto',
                             col_fecha: str = 'idSecuencia') -> pd.DataFrame:
    """
    Reconstruye DataFrame de test con predicciones en formato original.
    
    Parameters:
    -----------
    y_hat : pd.DataFrame
        Predicciones de NeuralForecast (tiene 'unique_id', 'ds', modelo_name)
    df_test_original : pd.DataFrame
        Dataset de test original (para recuperar columnas)
    modelo_name : str
        Nombre del modelo (columna en y_hat)
    col_producto : str
        Nombre de columna producto en dataset original
    col_fecha : str
        Nombre de columna fecha en dataset original
    
    Returns:
    --------
    pd.DataFrame
        Dataset test con predicciones
    """
    # Copiar test original
    test_pred = df_test_original.copy()
    
    # Renombrar columnas en y_hat para merge
    y_hat_merge = y_hat[['unique_id', 'ds', modelo_name]].copy()
    y_hat_merge = y_hat_merge.rename(columns={
        'unique_id': col_producto,
        'ds': col_fecha,
        modelo_name: 'prediccion'
    })
    
    # Asegurar tipos compatibles
    if not pd.api.types.is_datetime64_any_dtype(test_pred[col_fecha]):
        test_pred[col_fecha] = pd.to_datetime(test_pred[col_fecha])
    if not pd.api.types.is_datetime64_any_dtype(y_hat_merge[col_fecha]):
        y_hat_merge[col_fecha] = pd.to_datetime(y_hat_merge[col_fecha])
    
    # Merge
    test_pred = test_pred.merge(
        y_hat_merge,
        on=[col_producto, col_fecha],
        how='left'
    )
    
    print(f"\n✅ Predicciones reconstruidas:")
    print(f"   Shape: {test_pred.shape}")
    print(f"   Predicciones no nulas: {test_pred['prediccion'].notna().sum()}")
    
    return test_pred


def agregar_frecuencia_temporal(df: pd.DataFrame, 
                                col_fecha: str = 'ds',
                                freq: str = 'D') -> pd.DataFrame:
    """
    Asegura que el dataset tiene frecuencia temporal regular.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset con series temporales
    col_fecha : str
        Nombre de columna con fechas
    freq : str
        Frecuencia: 'D' diaria, 'W' semanal, 'M' mensual
    
    Returns:
    --------
    pd.DataFrame
        Dataset con frecuencia regular
    """
    # Asegurar datetime
    if not pd.api.types.is_datetime64_any_dtype(df[col_fecha]):
        df[col_fecha] = pd.to_datetime(df[col_fecha])
    
    # Inferir frecuencia si no se proporciona
    if freq is None:
        freq = pd.infer_freq(df.groupby('unique_id')[col_fecha].first())
        print(f"   Frecuencia inferida: {freq}")
    
    return df


def filtrar_productos_con_datos_suficientes(df: pd.DataFrame,
                                            min_observaciones: int = 30,
                                            col_producto: str = 'unique_id') -> pd.DataFrame:
    """
    Filtra productos con observaciones mínimas para entrenamiento.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    min_observaciones : int
        Número mínimo de observaciones por producto
    col_producto : str
        Nombre de columna producto
    
    Returns:
    --------
    pd.DataFrame
        Dataset filtrado
    """
    conteos = df[col_producto].value_counts()
    productos_validos = conteos[conteos >= min_observaciones].index
    
    df_filtrado = df[df[col_producto].isin(productos_validos)].copy()
    
    n_eliminados = df[col_producto].nunique() - df_filtrado[col_producto].nunique()
    
    print(f"\n✅ Filtrado de productos:")
    print(f"   Productos originales: {df[col_producto].nunique()}")
    print(f"   Productos válidos (>={min_observaciones} obs): {df_filtrado[col_producto].nunique()}")
    print(f"   Productos eliminados: {n_eliminados}")
    
    return df_filtrado
