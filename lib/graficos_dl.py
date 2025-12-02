"""
Gráficos especializados para modelos de Deep Learning
Autor: Joaquín Mora
Fecha: 2025-11-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def grafico_prediccion_diaria_agregada(df: pd.DataFrame,
                                       col_fecha: str = 'idSecuencia',
                                       col_real: str = 'udsVenta',
                                       col_pred: str = 'prediccion',
                                       titulo: str = 'Ventas Diarias - Real vs Predicción',
                                       figsize: tuple = (14, 5)):
    """
    Gráfico de ventas diarias agregadas (suma de todos los productos).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con predicciones
    col_fecha : str
        Nombre de columna fecha
    col_real : str
        Nombre de columna valores reales
    col_pred : str
        Nombre de columna predicciones
    titulo : str
        Título del gráfico
    figsize : tuple
        Tamaño de figura
    """
    # Agregar por día
    df_daily = df.groupby(col_fecha)[[col_real, col_pred]].sum().reset_index()
    
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=figsize)
    plt.plot(df_daily[col_fecha], df_daily[col_real], 
             label='Real', marker='o', markersize=4, linewidth=2, alpha=0.8)
    plt.plot(df_daily[col_fecha], df_daily[col_pred], 
             label='Predicción', marker='s', markersize=4, linewidth=2, alpha=0.8)
    
    plt.title(titulo, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Fecha', fontsize=11)
    plt.ylabel('Unidades Vendidas', fontsize=11)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Estadísticas
    error_daily = df_daily[col_pred] - df_daily[col_real]
    print(f"\n📊 Estadísticas diarias agregadas:")
    print(f"   MAE diario:  {np.abs(error_daily).mean():.2f}")
    print(f"   RMSE diario: {np.sqrt((error_daily**2).mean()):.2f}")
    print(f"   Error % medio: {(np.abs(error_daily) / df_daily[col_real] * 100).mean():.2f}%")


def grafico_prediccion_por_cluster(df: pd.DataFrame,
                                   col_cluster: str = 'Cluster',
                                   col_fecha: str = 'idSecuencia',
                                   col_real: str = 'udsVenta',
                                   col_pred: str = 'prediccion',
                                   figsize: tuple = (16, 10)):
    """
    Gráficos de predicción vs real por cada cluster (subplots).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con predicciones
    col_cluster : str
        Nombre de columna cluster
    col_fecha : str
        Nombre de columna fecha
    col_real : str
        Nombre de columna valores reales
    col_pred : str
        Nombre de columna predicciones
    figsize : tuple
        Tamaño de figura
    """
    clusters = sorted(df[col_cluster].unique())
    n_clusters = len(clusters)
    
    # Determinar layout
    n_cols = 2
    n_rows = (n_clusters + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Predicciones por Cluster (Diarias Agregadas)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten() if n_clusters > 1 else [axes]
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        
        # Filtrar cluster
        df_cluster = df[df[col_cluster] == cluster].copy()
        
        # Agregar por día
        df_cluster_daily = df_cluster.groupby(col_fecha)[[col_real, col_pred]].sum().reset_index()
        
        # Plotear
        ax.plot(df_cluster_daily[col_fecha], df_cluster_daily[col_real],
                label='Real', marker='o', markersize=3, linewidth=1.5, alpha=0.8)
        ax.plot(df_cluster_daily[col_fecha], df_cluster_daily[col_pred],
                label='Predicción', marker='s', markersize=3, linewidth=1.5, alpha=0.8)
        
        # Calcular métricas
        error = df_cluster_daily[col_pred] - df_cluster_daily[col_real]
        mae = np.abs(error).mean()
        rmse = np.sqrt((error**2).mean())
        
        ax.set_title(f'Cluster {cluster} | MAE: {mae:.1f} | RMSE: {rmse:.1f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=9)
        ax.set_ylabel('Unidades', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Ocultar ejes vacíos
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def grafico_productos_por_cluster(df: pd.DataFrame,
                                  col_cluster: str = 'Cluster',
                                  col_producto: str = 'producto',
                                  col_fecha: str = 'idSecuencia',
                                  col_real: str = 'udsVenta',
                                  col_pred: str = 'prediccion',
                                  n_productos_por_cluster: int = 2,
                                  figsize: tuple = (18, 12)):
    """
    Muestra predicciones para N productos representativos por cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con predicciones
    col_cluster : str
        Nombre de columna cluster
    col_producto : str
        Nombre de columna producto
    col_fecha : str
        Nombre de columna fecha
    col_real : str
        Nombre de columna valores reales
    col_pred : str
        Nombre de columna predicciones
    n_productos_por_cluster : int
        Número de productos a mostrar por cluster
    figsize : tuple
        Tamaño de figura
    """
    clusters = sorted(df[col_cluster].unique())
    n_clusters = len(clusters)
    
    # Layout: cada cluster tiene n_productos_por_cluster gráficos
    n_rows = n_clusters
    n_cols = n_productos_por_cluster
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'Top {n_productos_por_cluster} Productos por Cluster', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Asegurar que axes es 2D
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    if n_productos_por_cluster == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, cluster in enumerate(clusters):
        # Filtrar cluster
        df_cluster = df[df[col_cluster] == cluster].copy()
        
        # Top N productos por ventas totales
        top_productos = (df_cluster.groupby(col_producto)[col_real]
                        .sum()
                        .nlargest(n_productos_por_cluster)
                        .index.tolist())
        
        for col_idx, producto in enumerate(top_productos):
            ax = axes[row_idx, col_idx]
            
            # Filtrar producto
            df_prod = df_cluster[df_cluster[col_producto] == producto].copy()
            df_prod = df_prod.sort_values(col_fecha)
            
            # Plotear
            ax.plot(df_prod[col_fecha], df_prod[col_real],
                   label='Real', marker='o', markersize=3, linewidth=1.5)
            ax.plot(df_prod[col_fecha], df_prod[col_pred],
                   label='Predicción', marker='s', markersize=3, linewidth=1.5)
            
            # Métricas
            error = df_prod[col_pred] - df_prod[col_real]
            mae = np.abs(error).mean()
            
            ax.set_title(f'Cluster {cluster} | Producto {producto} | MAE: {mae:.1f}',
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Fecha', fontsize=8)
            ax.set_ylabel('Unidades', fontsize=8)
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=7)
        
        # Rellenar celdas vacías si no hay suficientes productos
        for col_idx in range(len(top_productos), n_productos_por_cluster):
            ax = axes[row_idx, col_idx]
            ax.text(0.5, 0.5, 'Sin datos suficientes', 
                   ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()


def dashboard_metricas_dl(metricas_dict: dict,
                         titulo: str = 'Dashboard de Métricas - Deep Learning',
                         figsize: tuple = (16, 10)):
    """
    Dashboard con visualización de métricas: tabla + gráficos de barras.
    
    Parameters:
    -----------
    metricas_dict : dict
        Diccionario con métricas por algoritmo
        Formato: {'LSTM': {'MAE': x, 'RMSE': y, ...}, ...}
    titulo : str
        Título del dashboard
    figsize : tuple
        Tamaño de figura
    """
    # Convertir a DataFrame
    df_metricas = pd.DataFrame(metricas_dict).T
    df_metricas = df_metricas.reset_index().rename(columns={'index': 'Algoritmo'})
    
    # Crear figura
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    fig.suptitle(titulo, fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Tabla de métricas (ocupa fila superior completa)
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Crear tabla
    tabla = ax_table.table(
        cellText=df_metricas.round(2).values,
        colLabels=df_metricas.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] + [0.1] * (len(df_metricas.columns) - 1)
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 2)
    
    # Colorear encabezados
    for i in range(len(df_metricas.columns)):
        tabla[(0, i)].set_facecolor('#4CAF50')
        tabla[(0, i)].set_text_props(weight='bold', color='white')
    
    ax_table.set_title('Tabla de Métricas Comparativas', 
                      fontsize=12, fontweight='bold', pad=10)
    
    # 2-7. Gráficos de barras para cada métrica
    metricas_a_plotear = ['MAE', 'RMSE', 'R2', 'MAPE (%)', 'SMAPE (%)', 'RMSSE']
    
    positions = [
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2)
    ]
    
    for idx, metrica in enumerate(metricas_a_plotear):
        if metrica not in df_metricas.columns:
            continue
        
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        
        valores = df_metricas[metrica]
        colores = plt.cm.viridis(np.linspace(0.3, 0.9, len(valores)))
        
        bars = ax.bar(df_metricas['Algoritmo'], valores, color=colores, alpha=0.8, edgecolor='black')
        
        # Etiquetar barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_title(metrica, fontsize=11, fontweight='bold')
        ax.set_ylabel(metrica, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    plt.show()


def grafico_comparacion_algoritmos(df_resultados: pd.DataFrame,
                                   metricas: List[str] = None,
                                   figsize: tuple = (16, 5)):
    """
    Gráfico comparativo de múltiples algoritmos (barras agrupadas).
    
    Parameters:
    -----------
    df_resultados : pd.DataFrame
        DataFrame con columna 'Algoritmo' y métricas
    metricas : List[str]
        Lista de métricas a comparar
    figsize : tuple
        Tamaño de figura
    """
    if metricas is None:
        metricas = ['MAE', 'RMSE', 'R2']
    
    n_metricas = len(metricas)
    fig, axes = plt.subplots(1, n_metricas, figsize=figsize)
    
    if n_metricas == 1:
        axes = [axes]
    
    fig.suptitle('Comparación de Algoritmos de Deep Learning', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx]
        
        if metrica not in df_resultados.columns:
            ax.text(0.5, 0.5, f'{metrica}\nno disponible', 
                   ha='center', va='center', fontsize=10)
            continue
        
        # Ordenar por métrica
        df_sorted = df_resultados.sort_values(metrica, ascending=(metrica != 'R2'))
        
        valores = df_sorted[metrica]
        colores = plt.cm.plasma(np.linspace(0.2, 0.9, len(valores)))
        
        bars = ax.barh(df_sorted['Algoritmo'], valores, color=colores, alpha=0.8, edgecolor='black')
        
        # Etiquetar barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {width:.2f}',
                   ha='left', va='center', fontsize=9)
        
        ax.set_title(metrica, fontsize=12, fontweight='bold')
        ax.set_xlabel(metrica, fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def grafico_loss_entrenamiento(history: dict,
                               titulo: str = 'Evolución del Loss durante Entrenamiento',
                               figsize: tuple = (12, 5)):
    """
    Gráfico de evolución del loss durante el entrenamiento.
    
    Parameters:
    -----------
    history : dict
        Diccionario con 'train_loss' y opcionalmente 'val_loss'
    titulo : str
        Título del gráfico
    figsize : tuple
        Tamaño de figura
    """
    plt.figure(figsize=figsize)
    
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
    
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
    
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def dashboard_prediccion_dl(df, col_fecha='idSecuencia', col_real='udsVenta', 
                           col_pred='prediccion', titulo_principal='Dashboard de Predicción',
                           figsize=(14, 10)):
    """
    Dashboard completo de predicción con 6 gráficos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con predicciones y valores reales
    col_fecha : str
        Nombre de la columna de fecha
    col_real : str
        Nombre de la columna con valores reales
    col_pred : str
        Nombre de la columna con predicciones
    titulo_principal : str
        Título principal del dashboard
    figsize : tuple
        Tamaño de la figura
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Calcular error si no existe
    if 'error' not in df.columns:
        df = df.copy()
        df['error'] = df[col_pred] - df[col_real]
    
    plt.style.use('seaborn-v0_8')

    # Crear figura con subplots (2 filas, 2 columnas)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(titulo_principal, fontsize=16, fontweight='bold', y=0.995)
    
    # IMPORTANTE: Aplanar axes para acceso consistente
    axes = axes.flatten()
    
    # Gráfico 1: Real vs Predicción temporal
    ax1 = axes[0]
    ax1.plot(df[col_fecha], df[col_real], 
             label='Real', marker='o', markersize=3, linewidth=1.5)
    ax1.plot(df[col_fecha], df[col_pred], 
             label='Predicción', marker='x', markersize=3, linewidth=1.5)
    ax1.set_title('Evolución Temporal: Real vs Predicción')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Unidades Vendidas')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Gráfico 2: Scatter Real vs Predicción
    ax2 = axes[1]
    ax2.scatter(df[col_real], df[col_pred], alpha=0.6)
    
    # Línea de identidad (predicción perfecta)
    min_val = min(df[col_real].min(), df[col_pred].min())
    max_val = max(df[col_real].max(), df[col_pred].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Predicción perfecta')
    
    ax2.set_title('Real vs Predicción (Scatter)')
    ax2.set_xlabel('Valores Reales')
    ax2.set_ylabel('Valores Predichos')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Gráfico 3: Distribución del Error
    ax3 = axes[2]
    ax3.hist(df['error'], bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    ax3.axvline(x=df['error'].mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Media = {df["error"].mean():.2f}')
    ax3.set_title('Distribución del Error')
    ax3.set_xlabel('Error (Predicción - Real)')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Gráfico 4: Evolución del Error
    ax4 = axes[3]
    ax4.plot(df[col_fecha], df['error'], marker='o', markersize=3, linewidth=1)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.fill_between(df[col_fecha], df['error'], 0, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_title('Evolución Temporal del Error')
    ax4.set_xlabel('Fecha')
    ax4.set_ylabel('Error')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()