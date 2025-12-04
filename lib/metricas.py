"""
Librería de métricas
Autor: Joaquín Mora
Fecha: 2025-11-21
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calcular_metricas(
    y: pd.Series,
    y_pred: pd.Series,
    algoritmo: str,
    ndetalle: str = "Global",
    cluster=None,
    producto=None,
) -> dict:
    """
    Calcula métricas de evaluación para modelos de series temporales e incluye
    metadatos del experimento para facilitar comparaciones posteriores.

    Parameters:
    -----------
    y : pd.Series
        Valores reales
    y_pred : pd.Series
        Valores predichos
    algoritmo : str
        Nombre del algoritmo/modelo (obligatorio)
    ndetalle : str, default="Global"
        Nivel de detalle del modelo. Posibles valores: "Global", "Cluster", "Producto"
    cluster : Any, optional
        Identificador del cluster (si aplica)
    producto : Any, optional
        Identificador del producto (si aplica)

    Returns:
    --------
    dict
        Diccionario con metadatos (Algoritmo, NDetalle, Cluster, Producto) y métricas calculadas
    """
    y_true = np.array(y)
    y_predicted = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_predicted)
    mse = mean_squared_error(y_true, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_predicted)
    
    # MAPE - evitar división por cero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_predicted[mask]) / y_true[mask])) * 100 if mask.any() else np.inf
    
    # SMAPE - evitar división por cero (0/0)
    denominator = np.abs(y_true) + np.abs(y_predicted)
    mask_smape = denominator != 0
    if mask_smape.any():
        smape = np.mean(2.0 * np.abs(y_predicted[mask_smape] - y_true[mask_smape]) / 
                       denominator[mask_smape]) * 100
    else:
        smape = 0.0  # Si ambos son siempre cero, el error es 0
    
    # RMSSE - evitar división por cero
    if len(y_true) > 1:
        naive_forecast = np.roll(y_true, 1)[1:]
        y_true_scaled = y_true[1:]
        mse_naive = np.mean((y_true_scaled - naive_forecast)**2)
        rmsse = np.sqrt(mse / mse_naive) if mse_naive > 0 else np.inf
    else:
        rmsse = np.inf
    
    # MAE Percentage
    mae_percentage = (mae / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else np.inf
    
    return {
        'Algoritmo': algoritmo,
        'NDetalle': ndetalle,
        'Cluster': cluster,
        'Producto': producto,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'MAPE (%)': round(mape, 2),
        'SMAPE (%)': round(smape, 2),
        'RMSSE': round(rmsse, 4),
        'MAE (%)': round(mae_percentage, 2)
    }


def comparar_metricas(resultados: list, ordenar_por: str = 'RMSE', 
                      ascendente: bool = True) -> pd.DataFrame:
    """
    Crea un DataFrame comparativo de múltiples modelos.
    
    Parameters:
    -----------
    resultados : list
        Lista de diccionarios devueltos por calcular_metricas
    ordenar_por : str, default='RMSE'
        Métrica por la cual ordenar los resultados
    ascendente : bool, default=True
        Si True, ordena de menor a mayor (mejor para MAE, RMSE, MAPE)
        Si False, ordena de mayor a menor (mejor para R2)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con todos los modelos y sus métricas ordenados
    """
    if not resultados:
        return pd.DataFrame()
    
    df_metricas = pd.DataFrame(resultados)
    
    # Verificar que la métrica existe
    if ordenar_por not in df_metricas.columns:
        print(f"⚠️  Métrica '{ordenar_por}' no encontrada. Usando 'RMSE'")
        ordenar_por = 'RMSE'
    
    # Reemplazar inf con NaN para ordenamiento correcto
    df_metricas = df_metricas.replace([np.inf, -np.inf], np.nan)
    
    # Ordenar (NaN van al final)
    df_metricas = df_metricas.sort_values(ordenar_por, ascending=ascendente, na_position='last')
    
    # Resetear índice
    df_metricas = df_metricas.reset_index(drop=True)
    
    return df_metricas


def resumen_metricas(resultados: list) -> None:
    """
    Imprime un resumen formateado de las métricas.
    
    Parameters:
    -----------
    resultados : list
        Lista de diccionarios devueltos por calcular_metricas
    """
    if not resultados:
        print("⚠️  No hay resultados para mostrar")
        return
    
    df = comparar_metricas(resultados)
    
    print("\n" + "="*100)
    print("📊 RESUMEN DE MÉTRICAS")
    print("="*100)
    # Reordenar columnas para visualización clara
    cols = [
        'Algoritmo', 'NDetalle', 'Cluster', 'Producto',
        'MAE', 'RMSE', 'R2', 'MAPE (%)', 'SMAPE (%)', 'RMSSE', 'MAE (%)'
    ]
    cols_existentes = [c for c in cols if c in df.columns]
    print(df[cols_existentes].to_string(index=False))
    print("="*100)
    
    # Identificar mejor modelo (ignorando NaN/inf)
    df_valido = df.dropna(subset=['RMSE'])
    if not df_valido.empty:
        mejor_modelo = df_valido.iloc[0]['Algoritmo']
        mejor_rmse = df_valido.iloc[0]['RMSE']
        print(f"\n🏆 Mejor modelo: {mejor_modelo} (RMSE: {mejor_rmse:.4f})")
    else:
        print("\n⚠️  No hay modelos válidos para comparar")


def agregar_estadisticas_error(df: pd.DataFrame, col_real: str, 
                                col_pred: str) -> pd.DataFrame:
    """
    Agrega columnas de error a un DataFrame con predicciones.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con predicciones
    col_real : str
        Nombre de la columna con valores reales
    col_pred : str
        Nombre de la columna con valores predichos
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con columnas adicionales: error, error_abs, error_pct
    """
    df = df.copy()
    df['error'] = df[col_pred] - df[col_real]
    df['error_abs'] = np.abs(df['error'])
    
    # Error porcentual evitando división por cero
    mask = df[col_real] != 0
    df['error_pct'] = 0.0
    df.loc[mask, 'error_pct'] = (df.loc[mask, 'error'] / df.loc[mask, col_real]) * 100
    
    return df


def resumen_final_modelos(todas_metricas: list) -> None:
    """
    Genera un resumen completo de todas las métricas de modelos con tablas detalladas
    por algoritmo y nivel de detalle.
    
    Parameters:
    -----------
    todas_metricas : list
        Lista de diccionarios con métricas de todos los modelos evaluados.
        Cada diccionario debe contener: Algoritmo, NDetalle, Cluster, Producto y métricas.
    """
    print("\n" + "="*100)
    print("🏆 RESUMEN FINAL - COMPARACIÓN DE TODOS LOS MODELOS")
    print("="*100)

    if todas_metricas:
        # Tabla completa de métricas con metadatos
        print("\n📊 Resumen de todas las métricas (todas las columnas):")
        df_comparacion_final = pd.DataFrame(todas_metricas).copy()

        # Asegurar columnas y ordenar por Algoritmo y NDetalle (y RMSE ascendente dentro)
        columnas = [
            'Algoritmo', 'NDetalle', 'Cluster', 'Producto',
            'MAE', 'MSE', 'RMSE', 'R2', 'MAPE (%)', 'SMAPE (%)', 'RMSSE', 'MAE (%)'
        ]
        cols_exist = [c for c in columnas if c in df_comparacion_final.columns]

        # Reemplazar inf por NaN para ordenar correctamente
        df_comparacion_final = df_comparacion_final.replace([np.inf, -np.inf], np.nan)

        # Orden: Algoritmo, NDetalle, RMSE
        orden_cols = [c for c in ['Algoritmo', 'NDetalle', 'RMSE'] if c in df_comparacion_final.columns]
        df_comparacion_final = df_comparacion_final.sort_values(orden_cols, ascending=[True, True, True])

        print(df_comparacion_final[cols_exist].to_string(index=False))

        # Resúmenes por NDetalle
        print("\n" + "="*100)
        print("📊 ANÁLISIS POR CATEGORÍAS DE MODELOS (NDetalle)")
        print("="*100)

        for nivel in ['Global', 'Cluster', 'Producto']:
            dfn = df_comparacion_final[df_comparacion_final['NDetalle'] == nivel]
            print(f"\n🔹 {nivel.upper()}:")
            if len(dfn) > 0:
                cols_nivel = ['Algoritmo', 'NDetalle', 'Cluster', 'Producto', 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE (%)']
                print(dfn[[c for c in cols_nivel if c in dfn.columns]].to_string(index=False))
                print(f"\n   Promedio RMSE: {dfn['RMSE'].mean():.2f}")
                if nivel == 'Cluster':
                    print(f"   Mejor cluster: {dfn.loc[dfn['RMSE'].idxmin(), 'Cluster']} (RMSE: {dfn['RMSE'].min():.2f})")
                    print(f"   Peor cluster: {dfn.loc[dfn['RMSE'].idxmax(), 'Cluster']} (RMSE: {dfn['RMSE'].max():.2f})")
                if nivel == 'Producto':
                    best_idx = dfn['RMSE'].idxmin()
                    worst_idx = dfn['RMSE'].idxmax()
                    print(f"   Mejor producto: C{dfn.loc[best_idx, 'Cluster']} P{dfn.loc[best_idx, 'Producto']} (RMSE: {dfn['RMSE'].min():.2f})")
                    print(f"   Peor producto: C{dfn.loc[worst_idx, 'Cluster']} P{dfn.loc[worst_idx, 'Producto']} (RMSE: {dfn['RMSE'].max():.2f})")

        # Mejor modelo general (por RMSE)
        print("\n" + "="*100)
        print("🥇 MEJOR MODELO GENERAL (RMSE mínimo)")
        print("="*100)
        mejor_idx = df_comparacion_final['RMSE'].idxmin()
        mejor_modelo = df_comparacion_final.loc[mejor_idx]

        print(f"\n🏆 {mejor_modelo['Algoritmo']} | {mejor_modelo['NDetalle']}")
        print(f"   Cluster:    {mejor_modelo.get('Cluster', None)}")
        print(f"   Producto:   {mejor_modelo.get('Producto', None)}")
        print(f"   MAE:        {mejor_modelo['MAE']:.2f}")
        print(f"   MSE:        {mejor_modelo['MSE']:.2f}")
        print(f"   RMSE:       {mejor_modelo['RMSE']:.2f}")
        print(f"   R²:         {mejor_modelo['R2']:.4f}")
        print(f"   MAPE:       {mejor_modelo['MAPE (%)']:.2f}%")
        print(f"   SMAPE:      {mejor_modelo['SMAPE (%)']:.2f}%")
        print(f"   RMSSE:      {mejor_modelo['RMSSE']:.2f}")
        print(f"   MAE (%):    {mejor_modelo['MAE (%)']:.2f}")

    else:
        print("\n⚠️  No se generaron métricas para ningún modelo")

    print("\n" + "="*100)
    print("✅ ANÁLISIS COMPLETO FINALIZADO")
    print("="*100)
    print(f"\n📊 Total de modelos evaluados: {len(todas_metricas)}")
    print(f"   - Modelos globales: {len([m for m in todas_metricas if m.get('NDetalle') == 'Global'])}")
    print(f"   - Modelos por cluster: {len([m for m in todas_metricas if m.get('NDetalle') == 'Cluster'])}")
    print(f"   - Modelos por producto: {len([m for m in todas_metricas if m.get('NDetalle') == 'Producto'])}")