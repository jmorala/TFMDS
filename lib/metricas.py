import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calcular_metricas(y: pd.Series, y_pred: pd.Series, name: str) -> dict:
    """
    Calcula métricas de evaluación para modelos de series temporales.
    
    Parameters:
    -----------
    y : pd.Series
        Valores reales
    y_pred : pd.Series
        Valores predichos
    name : str
        Nombre del algoritmo/modelo
    
    Returns:
    --------
    dict
        Diccionario con el nombre del algoritmo y las métricas calculadas
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
        'Algoritmo': name,
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
    
    print("\n" + "="*80)
    print("📊 RESUMEN DE MÉTRICAS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
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