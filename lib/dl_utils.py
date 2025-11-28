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
                             col_fecha: str = 'idSecuencia',
                             col_target: str = 'udsVenta',
                             scaler_target_path: str = 'datos/scaler_target.pkl') -> pd.DataFrame:
    """
    Reconstruye DataFrame de test con predicciones en formato original.
    Desnormaliza automáticamente las predicciones y el target si existe scaler.
    
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
    col_target : str
        Nombre de columna objetivo (para desnormalizar)
    scaler_target_path : str
        Ruta al archivo scaler_target.pkl (None para no desnormalizar)
    
    Returns:
    --------
    pd.DataFrame
        Dataset test con predicciones desnormalizadas
    """
    import pickle
    import os
    
    # Copiar test original
    test_pred = df_test_original.copy()
    
    # Copiar y_hat y resetear índice si unique_id está como índice
    y_hat_work = y_hat.copy()
    if 'unique_id' in y_hat_work.index.names or (y_hat_work.index.name == 'unique_id'):
        y_hat_work = y_hat_work.reset_index()
    
    # Renombrar columnas en y_hat para merge
    y_hat_merge = y_hat_work[['unique_id', 'ds', modelo_name]].copy()
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
    
    # ============================================================
    # DESNORMALIZACIÓN CON SCALER
    # ============================================================
    if scaler_target_path and os.path.exists(scaler_target_path):
        print(f"\n🔄 Desnormalizando predicciones y target...")
        
        try:
            # Cargar scaler
            with open(scaler_target_path, 'rb') as f:
                scaler_target = pickle.load(f)
            
            # Desnormalizar target (udsVenta)
            if col_target in test_pred.columns:
                test_pred[col_target] = scaler_target.inverse_transform(
                    test_pred[[col_target]]
                ).flatten()
            
            # Desnormalizar predicciones
            mask_valid = test_pred['prediccion'].notna()
            if mask_valid.sum() > 0:
                test_pred.loc[mask_valid, 'prediccion'] = scaler_target.inverse_transform(
                    test_pred.loc[mask_valid, ['prediccion']]
                ).flatten()
            
            print(f"   ✅ Desnormalización completada")
            print(f"   📊 Estadísticas desnormalizadas:")
            print(f"      - {col_target}: min={test_pred[col_target].min():.2f}, "
                  f"max={test_pred[col_target].max():.2f}, "
                  f"mean={test_pred[col_target].mean():.2f}")
            print(f"      - prediccion: min={test_pred['prediccion'].min():.2f}, "
                  f"max={test_pred['prediccion'].max():.2f}, "
                  f"mean={test_pred['prediccion'].mean():.2f}")
            
        except Exception as e:
            print(f"   ⚠️ Error al desnormalizar: {e}")
            print(f"   Las predicciones quedan en escala normalizada")
    else:
        if scaler_target_path:
            print(f"\n⚠️ No se encontró scaler en: {scaler_target_path}")
        print(f"   Las predicciones están en escala normalizada")
    
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


def preparar_variables_estaticas(df_train_nf: pd.DataFrame,
                                 df_test_nf: pd.DataFrame,
                                 stat_exog_list: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extrae variables estáticas (que no cambian en el tiempo) y las separa en un DataFrame aparte.
    Elimina estas columnas de los datasets temporales.
    
    En NeuralForecast, las variables estáticas deben proporcionarse en un DataFrame separado
    con una fila por unique_id, no repetidas en cada timestamp.
    
    Parameters:
    -----------
    df_train_nf : pd.DataFrame
        Dataset de entrenamiento en formato NeuralForecast
    df_test_nf : pd.DataFrame
        Dataset de test en formato NeuralForecast
    stat_exog_list : List[str]
        Lista de nombres de columnas estáticas a extraer
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - df_train_nf sin columnas estáticas
        - df_test_nf sin columnas estáticas
        - static_df: DataFrame con unique_id y variables estáticas
    """
    # Verificar que las columnas existen
    columnas_faltantes = [col for col in stat_exog_list if col not in df_train_nf.columns]
    if columnas_faltantes:
        print(f"\n⚠️ Columnas no encontradas en train: {columnas_faltantes}")
        stat_exog_list = [col for col in stat_exog_list if col in df_train_nf.columns]
    
    if not stat_exog_list:
        print("\n⚠️ No hay columnas estáticas válidas para extraer")
        return df_train_nf, df_test_nf, None
    
    # Crear DataFrame de variables estáticas (una fila por unique_id)
    columnas_static = ['unique_id'] + stat_exog_list
    static_df = df_train_nf[columnas_static].drop_duplicates('unique_id').reset_index(drop=True)
    
    # Eliminar columnas estáticas de los datasets temporales
    df_train_clean = df_train_nf.drop(columns=stat_exog_list)
    df_test_clean = df_test_nf.drop(columns=stat_exog_list)
    
    print(f"\n✅ Variables estáticas extraídas:")
    print(f"   Columnas estáticas: {stat_exog_list}")
    print(f"   Shape static_df: {static_df.shape}")
    print(f"   Productos únicos: {static_df['unique_id'].nunique()}")
    print(f"\n📊 Datasets temporales actualizados:")
    print(f"   Train: {df_train_clean.shape} (eliminadas {len(stat_exog_list)} columnas)")
    print(f"   Test:  {df_test_clean.shape} (eliminadas {len(stat_exog_list)} columnas)")
    
    return df_train_clean, df_test_clean, static_df
