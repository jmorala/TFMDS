"""
Librería de utilidades generales para análisis de datos.
Contiene funciones de propósito general que pueden ser reutilizadas en diferentes notebooks.
"""

import pandas as pd
from typing import List, Dict, Optional


def obtener_top_productos_por_cluster(
    df: pd.DataFrame,
    col_ventas: str,
    col_cluster: str,
    col_producto: str,
    n_productos: int = 10
) -> Dict[int, List]:
    """
    Obtiene los productos con más ventas por cada cluster.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos de ventas
    col_ventas : str
        Nombre de la columna que contiene las ventas
    col_cluster : str
        Nombre de la columna que contiene el cluster
    col_producto : str
        Nombre de la columna que contiene el identificador del producto
    n_productos : int, opcional (default=10)
        Número de productos top a retornar por cada cluster
    
    Retorna
    -------
    Dict[int, List]
        Diccionario donde cada clave es un cluster y el valor es una lista
        con los identificadores de los productos top de ese cluster
    
    Ejemplo
    -------
    >>> df = pd.DataFrame({
    ...     'udsVenta': [100, 200, 150, 300, 250],
    ...     'Cluster': [0, 0, 1, 1, 1],
    ...     'producto': [1, 2, 3, 4, 5]
    ... })
    >>> top_productos = obtener_top_productos_por_cluster(
    ...     df, 'udsVenta', 'Cluster', 'producto', n_productos=2
    ... )
    >>> print(top_productos)
    {0: [2, 1], 1: [4, 5]}
    """
    # Validar que las columnas existen
    columnas_requeridas = [col_ventas, col_cluster, col_producto]
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    if columnas_faltantes:
        raise ValueError(f"Columnas faltantes en el DataFrame: {columnas_faltantes}")
    
    # Agrupar por cluster y producto, sumando las ventas
    ventas_por_producto = (
        df.groupby([col_cluster, col_producto], observed=True)[col_ventas]
        .sum()
        .reset_index()
    )
    
    # Diccionario para almacenar los resultados
    top_productos_dict = {}
    
    # Obtener clusters únicos ordenados
    clusters = sorted(ventas_por_producto[col_cluster].unique())
    
    # Para cada cluster, obtener los n productos con más ventas
    for cluster in clusters:
        # Filtrar datos del cluster
        cluster_data = ventas_por_producto[
            ventas_por_producto[col_cluster] == cluster
        ]
        
        # Ordenar por ventas descendente y tomar los top n
        top_productos = (
            cluster_data
            .sort_values(col_ventas, ascending=False)
            .head(n_productos)[col_producto]
            .tolist()
        )
        
        top_productos_dict[cluster] = top_productos
    
    return top_productos_dict


def imprimir_top_productos_por_cluster(
    top_productos: Dict[int, List],
    df: pd.DataFrame,
    col_ventas: str,
    col_cluster: str,
    col_producto: str
) -> None:
    """
    Imprime un resumen formateado de los productos top por cluster con sus ventas totales.
    
    Parámetros
    ----------
    top_productos : Dict[int, List]
        Diccionario con los productos top por cluster (resultado de obtener_top_productos_por_cluster)
    df : pd.DataFrame
        DataFrame original con los datos de ventas
    col_ventas : str
        Nombre de la columna que contiene las ventas
    col_cluster : str
        Nombre de la columna que contiene el cluster
    col_producto : str
        Nombre de la columna que contiene el identificador del producto
    """
    print("\n" + "="*80)
    print("🏆 TOP PRODUCTOS POR CLUSTER")
    print("="*80)
    
    # Calcular ventas totales por producto
    ventas_totales = (
        df.groupby([col_cluster, col_producto], observed=True)[col_ventas]
        .sum()
        .to_dict()
    )
    
    for cluster in sorted(top_productos.keys()):
        productos = top_productos[cluster]
        print(f"\n📍 Cluster {cluster} - Top {len(productos)} productos:")
        
        for i, producto in enumerate(productos, 1):
            ventas = ventas_totales.get((cluster, producto), 0)
            print(f"   {i:2d}. Producto {producto:4d} → {ventas:>10,.0f} unidades")
    
    print("\n" + "="*80)
