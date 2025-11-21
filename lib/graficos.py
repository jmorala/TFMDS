import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def grafico_real_vs_prediccion(df: pd.DataFrame, col_fecha: str, col_real: str, 
                                col_pred: str, titulo: str = 'Real vs Predicción',
                                figsize: tuple = (12, 5)):
    """
    Gráfico de línea temporal comparando valores reales vs predichos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    col_fecha : str
        Nombre de la columna con fechas
    col_real : str
        Nombre de la columna con valores reales
    col_pred : str
        Nombre de la columna con predicciones
    titulo : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    plt.plot(df[col_fecha], df[col_real], 
             label='Real', marker='o', markersize=3, linewidth=1.5, alpha=0.7)
    plt.plot(df[col_fecha], df[col_pred], 
             label='Predicción', marker='x', markersize=3, linewidth=1.5, alpha=0.7)
    
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=11)
    plt.ylabel('Unidades', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def grafico_scatter_prediccion(df: pd.DataFrame, col_real: str, col_pred: str,
                                titulo: str = 'Real vs Predicción',
                                figsize: tuple = (8, 6)):
    """
    Gráfico de dispersión comparando valores reales vs predichos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    col_real : str
        Nombre de la columna con valores reales
    col_pred : str
        Nombre de la columna con predicciones
    titulo : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    plt.scatter(df[col_real], df[col_pred], alpha=0.5)
    
    # Línea diagonal ideal
    min_val = min(df[col_real].min(), df[col_pred].min())
    max_val = max(df[col_real].max(), df[col_pred].max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Predicción perfecta')
    
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('Ventas Reales', fontsize=11)
    plt.ylabel('Ventas Predichas', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def grafico_distribucion_error(df: pd.DataFrame, col_error: str = 'error',
                                titulo: str = 'Distribución del Error',
                                bins: int = 30, figsize: tuple = (10, 5)):
    """
    Histograma de la distribución del error de predicción.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    col_error : str
        Nombre de la columna con errores
    titulo : str
        Título del gráfico
    bins : int
        Número de bins para el histograma
    figsize : tuple
        Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    plt.hist(df[col_error], bins=bins, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Error = 0')
    plt.axvline(x=df[col_error].mean(), color='g', linestyle='--', 
                linewidth=2, label=f'Media = {df[col_error].mean():.2f}')
    
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('Error (Predicción - Real)', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def grafico_feature_importance(modelo, feature_names: list, top_n: int = 15,
                                titulo: str = 'Importancia de Features',
                                figsize: tuple = (10, 8)):
    """
    Gráfico de barras horizontales con la importancia de features.
    
    Parameters:
    -----------
    modelo : sklearn model
        Modelo entrenado con atributo feature_importances_
    feature_names : list
        Lista con los nombres de las features
    top_n : int
        Número de features más importantes a mostrar
    titulo : str
        Título del gráfico
    figsize : tuple
        Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('Importancia', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def grafico_comparacion_metricas(df_metricas: pd.DataFrame, 
                                  metricas: list = None,
                                  figsize: tuple = (15, 5)):
    """
    Gráfico de barras comparando métricas entre modelos.
    
    Parameters:
    -----------
    df_metricas : pd.DataFrame
        DataFrame con las métricas (de comparar_metricas)
    metricas : list
        Lista de métricas a comparar
    figsize : tuple
        Tamaño de la figura
    """
    n_metricas = len(metricas)
    fig, axes = plt.subplots(1, n_metricas, figsize=figsize)
    
    if n_metricas == 1:
        axes = [axes]
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx]
        
        if metrica not in df_metricas.columns:
            ax.text(0.5, 0.5, f'Métrica {metrica}\nno encontrada', 
                   ha='center', va='center')
            continue
        
        # Extraer identificador numérico si existe (Cluster o Producto)
        labels = df_metricas['Algoritmo'].str.extract(r'(\d+)$')[0]
        if labels.isna().all():
            labels = df_metricas['Algoritmo']
        
        valores = df_metricas[metrica]
        
        ax.bar(range(len(valores)), valores, alpha=0.7, edgecolor='black')
        ax.set_title(f'{metrica}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Modelo', fontsize=10)
        ax.set_ylabel(metrica, fontsize=10)
        ax.set_xticks(range(len(valores)))
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def dashboard_prediccion(df: pd.DataFrame, col_fecha: str, col_real: str, 
                        col_pred: str, modelo=None, feature_names: list = None,
                        titulo_principal: str = 'Dashboard de Predicción',
                        figsize: tuple = (15, 10)):
    """
    Dashboard completo con 4 gráficos: temporal, scatter, error y feature importance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos (debe tener columna 'error')
    col_fecha : str
        Nombre de la columna con fechas
    col_real : str
        Nombre de la columna con valores reales
    col_pred : str
        Nombre de la columna con predicciones
    modelo : sklearn model, optional
        Modelo para extraer feature importance
    feature_names : list, optional
        Lista de nombres de features
    titulo_principal : str
        Título principal del dashboard
    figsize : tuple
        Tamaño de la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(titulo_principal, fontsize=16, fontweight='bold', y=0.995)
    
    # Gráfico 1: Real vs Predicción temporal
    ax1 = axes[0, 0]
    ax1.plot(df[col_fecha], df[col_real], 
             label='Real', marker='o', markersize=3, linewidth=1.5)
    ax1.plot(df[col_fecha], df[col_pred], 
             label='Predicción', marker='x', markersize=3, linewidth=1.5)
    ax1.set_title('Real vs Predicción (Temporal)', fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Unidades')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Scatter
    ax2 = axes[0, 1]
    ax2.scatter(df[col_real], df[col_pred], alpha=0.5)
    min_val = min(df[col_real].min(), df[col_pred].min())
    max_val = max(df[col_real].max(), df[col_pred].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_title('Real vs Predicción (Scatter)', fontweight='bold')
    ax2.set_xlabel('Ventas Reales')
    ax2.set_ylabel('Ventas Predichas')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Distribución del error
    ax3 = axes[1, 0]
    ax3.hist(df['error'], bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=df['error'].mean(), color='g', linestyle='--', linewidth=2)
    ax3.set_title('Distribución del Error', fontweight='bold')
    ax3.set_xlabel('Error (Predicción - Real)')
    ax3.set_ylabel('Frecuencia')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 4: Feature Importance
    ax4 = axes[1, 1]
    if modelo is not None and feature_names is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': modelo.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        ax4.barh(feature_importance['feature'], feature_importance['importance'])
        ax4.set_title('Top 10 Features Importantes', fontweight='bold')
        ax4.set_xlabel('Importancia')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
    else:
        ax4.text(0.5, 0.5, 'Feature Importance\nno disponible', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Feature Importance', fontweight='bold')
    
    plt.tight_layout()
    plt.show()