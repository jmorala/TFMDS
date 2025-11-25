# Estructura de Deep Learning para Forecasting

## 📁 Estructura del Proyecto

```
TFMDS/
├── cuadernos/
│   ├── 03_Preprocesamiento.ipynb          # Preprocesamiento general + DL
│   ├── 05_RF_Forecasting.ipynb            # Random Forest
│   ├── 05_XGBoost_Forescasting.ipynb      # XGBoost
│   ├── 05_LightGBM_Forescating.ipynb      # LightGBM
│   ├── 05_CatBoost_Forecasting.ipynb      # CatBoost
│   ├── 06_DL_LSTM.ipynb                   # LSTM (plantilla Deep Learning)
│   ├── 06_DL_GRU.ipynb                    # GRU (a crear)
│   ├── 06_DL_NBEATS.ipynb                 # NBEATS (a crear)
│   ├── 06_DL_DeepAR.ipynb                 # DeepAR (a crear)
│   ├── 06_DL_Transformer.ipynb            # Transformer (a crear)
│   └── 07_Comparacion_Global.ipynb        # Comparación de todos los modelos
│
├── lib/
│   ├── __init__.py                        # Inicialización del paquete
│   ├── metricas.py                        # Funciones de cálculo de métricas
│   ├── graficos.py                        # Gráficos para modelos tree-based
│   ├── graficos_dl.py                     # Gráficos especializados para DL
│   ├── dl_utils.py                        # Utilidades para modelos DL
│   └── ts_features.py                     # Features de series temporales
│
└── datos/
    ├── df_train_dl.csv                    # Dataset train preparado para DL
    ├── df_test_dl.csv                     # Dataset test preparado para DL
    ├── scaler_features.pkl                # Scaler para features numéricas
    ├── scaler_target.pkl                  # Scaler para target
    ├── resultados_metricas_rf.csv         # Métricas Random Forest
    ├── resultados_metricas_xgboost.csv    # Métricas XGBoost
    ├── resultados_metricas_lightgbm.csv   # Métricas LightGBM
    ├── resultados_metricas_catboost.csv   # Métricas CatBoost
    ├── resultados_metricas_lstm.csv       # Métricas LSTM
    ├── resultados_metricas_gru.csv        # Métricas GRU (a generar)
    ├── resultados_metricas_nbeats.csv     # Métricas NBEATS (a generar)
    └── comparacion_todos_modelos.csv      # Comparación consolidada
```

## 🎯 Flujo de Trabajo

### 1. Preprocesamiento de Datos
**Notebook:** `03_Preprocesamiento.ipynb`

- Cargar datos originales
- Crear features de series temporales (lags, EWMA, medias móviles)
- **Preparación para Deep Learning:**
  - Normalización con StandardScaler (media=0, std=1)
  - Codificación cíclica de variables temporales (sen/cos)
  - One-Hot encoding de variables categóricas de baja cardinalidad
  - Mantener producto como entero (para Embedding Layers)
- Guardar datasets:
  - `df_train_dl.csv` y `df_test_dl.csv` (para DL)
  - `df_train.csv` y `df_test.csv` (para tree-based)
  - Scalers en formato pickle

### 2. Modelos Tree-Based
**Notebooks:** `05_*_Forecasting.ipynb`

- Cargar `df_train.csv` y `df_test.csv`
- Optimización de hiperparámetros con Optuna
- Entrenamiento de modelos (Global, por Cluster, por Producto)
- Cálculo de métricas con `lib.metricas.calcular_metricas()`
- Visualizaciones con `lib.graficos`
- Guardar resultados en `datos/resultados_metricas_*.csv`

### 3. Modelos Deep Learning
**Notebooks:** `06_DL_*.ipynb`

#### Estructura Unificada para Todos los Modelos DL:

1. **Carga de Datos**
   ```python
   df_train_raw = pd.read_csv('datos/df_train_dl.csv', sep=';', parse_dates=['idSecuencia'])
   df_test_raw = pd.read_csv('datos/df_test_dl.csv', sep=';', parse_dates=['idSecuencia'])
   ```

2. **Conversión al Formato NeuralForecast**
   ```python
   from lib.dl_utils import preparar_datos_neuralforecast
   
   df_train_nf, df_test_nf = preparar_datos_neuralforecast(
       df_train_raw, df_test_raw,
       col_fecha='idSecuencia',
       col_producto='producto',
       col_target='udsVenta'
   )
   ```

3. **Configuración del Modelo**
   ```python
   from neuralforecast.models import LSTM  # GRU, NBEATS, etc.
   
   modelo = LSTM(
       h=30,                    # Horizonte de predicción
       input_size=60,           # Ventana de entrada
       max_steps=500,
       learning_rate=1e-3,
       # ... hiperparámetros específicos
   )
   ```

4. **Entrenamiento**
   ```python
   from neuralforecast import NeuralForecast
   
   nf = NeuralForecast(models=[modelo], freq='D')
   nf.fit(df=df_train_nf)
   ```

5. **Predicción**
   ```python
   y_hat = nf.predict(futr_df=df_test_nf)
   
   from lib.dl_utils import reconstruir_predicciones
   df_test_pred = reconstruir_predicciones(
       y_hat, df_test_raw, 'LSTM',
       col_producto='producto',
       col_fecha='idSecuencia'
   )
   ```

6. **Métricas**
   ```python
   from lib.metricas import calcular_metricas, resumen_metricas
   
   metricas = calcular_metricas(
       y=df_test_pred['udsVenta'],
       y_pred=df_test_pred['prediccion'],
       name='LSTM'
   )
   resumen_metricas([metricas])
   ```

7. **Visualizaciones**
   ```python
   from lib.graficos_dl import (
       grafico_prediccion_diaria_agregada,
       grafico_prediccion_por_cluster,
       grafico_productos_por_cluster,
       dashboard_metricas_dl
   )
   
   # Suma diaria de todos los productos
   grafico_prediccion_diaria_agregada(df_test_pred)
   
   # Predicciones por cluster
   grafico_prediccion_por_cluster(df_test_pred)
   
   # Top 2 productos por cluster
   grafico_productos_por_cluster(df_test_pred, n_productos_por_cluster=2)
   
   # Dashboard de métricas
   dashboard_metricas_dl({'LSTM': metricas})
   ```

8. **Guardar Resultados**
   ```python
   pd.DataFrame([metricas]).to_csv('datos/resultados_metricas_lstm.csv', index=False)
   ```

### 4. Comparación Global
**Notebook:** `07_Comparacion_Global.ipynb`

- Cargar todos los archivos `resultados_metricas_*.csv`
- Consolidar métricas en un único DataFrame
- Análisis comparativo:
  - Top 10 modelos
  - Mejores por tipo (Tree-Based vs Deep Learning)
  - Distribución de métricas (boxplots)
  - Heatmap de métricas
- Identificar mejor modelo global
- Exportar resultados consolidados

## 📊 Métricas Calculadas

Todas calculadas con `lib.metricas.calcular_metricas()`:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coeficiente de determinación)
- **MAPE** (Mean Absolute Percentage Error)
- **SMAPE** (Symmetric Mean Absolute Percentage Error)
- **RMSSE** (Root Mean Squared Scaled Error)
- **MAE (%)** (MAE como porcentaje de la media)

## 🖼️ Gráficos Disponibles

### Para Modelos Tree-Based (`lib.graficos`)
- `grafico_real_vs_prediccion()`: Serie temporal real vs predicción
- `grafico_scatter_prediccion()`: Scatter plot real vs predicción
- `grafico_feature_importance()`: Importancia de features
- `dashboard_prediccion()`: Dashboard completo de 4 gráficos

### Para Modelos Deep Learning (`lib.graficos_dl`)
- `grafico_prediccion_diaria_agregada()`: Suma diaria de todos los productos
- `grafico_prediccion_por_cluster()`: Subplots por cluster
- `grafico_productos_por_cluster()`: Top N productos por cluster
- `dashboard_metricas_dl()`: Tabla + gráficos de barras de métricas
- `grafico_comparacion_algoritmos()`: Comparación múltiples algoritmos
- `grafico_loss_entrenamiento()`: Evolución del loss (si disponible)

## 🔧 Utilidades Deep Learning (`lib.dl_utils`)

- `preparar_datos_neuralforecast()`: Convierte datasets al formato requerido (unique_id, ds, y)
- `crear_dataset_validacion()`: Divide train en train/validación temporal
- `seleccionar_features_exogenas()`: Identifica features exógenas automáticamente
- `reconstruir_predicciones()`: Convierte predicciones NeuralForecast a formato original
- `filtrar_productos_con_datos_suficientes()`: Filtra productos con observaciones mínimas

## 🤖 Algoritmos de Deep Learning con NeuralForecast

### LSTM (Long Short-Term Memory)
```python
from neuralforecast.models import LSTM

modelo = LSTM(
    h=30,
    input_size=60,
    encoder_hidden_size=128,
    encoder_n_layers=2,
    learning_rate=1e-3,
    max_steps=500
)
```

### GRU (Gated Recurrent Unit)
```python
from neuralforecast.models import GRU

modelo = GRU(
    h=30,
    input_size=60,
    encoder_hidden_size=128,
    encoder_n_layers=2,
    learning_rate=1e-3,
    max_steps=500
)
```

### NBEATS (Neural Basis Expansion Analysis)
```python
from neuralforecast.models import NBEATS

modelo = NBEATS(
    h=30,
    input_size=60,
    stack_types=['trend', 'seasonality'],
    n_blocks=[3, 3],
    mlp_units=[[512, 512], [512, 512]],
    learning_rate=1e-3,
    max_steps=500
)
```

### DeepAR (Probabilistic Forecasting)
```python
from neuralforecast.models import DeepAR

modelo = DeepAR(
    h=30,
    input_size=60,
    encoder_hidden_size=128,
    encoder_n_layers=2,
    learning_rate=1e-3,
    max_steps=500
)
```

### Vanilla Transformer
```python
from neuralforecast.models import VanillaTransformer

modelo = VanillaTransformer(
    h=30,
    input_size=60,
    hidden_size=128,
    n_head=4,
    learning_rate=1e-4,
    max_steps=500
)
```

### TFT (Temporal Fusion Transformer)
```python
from neuralforecast.models import TFT

modelo = TFT(
    h=30,
    input_size=60,
    hidden_size=128,
    lstm_layers=2,
    attention_heads=4,
    dropout=0.1,
    learning_rate=1e-3,
    max_steps=500
)
```

## 🎓 Optimización de Hiperparámetros con Optuna

Para modelos DL que soporten Optuna:

```python
import optuna
from neuralforecast.losses.pytorch import MAE

def objective(trial):
    modelo = LSTM(
        h=30,
        input_size=trial.suggest_int('input_size', 30, 90),
        encoder_hidden_size=trial.suggest_int('hidden_size', 64, 256),
        encoder_n_layers=trial.suggest_int('n_layers', 1, 3),
        learning_rate=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        max_steps=500
    )
    
    nf = NeuralForecast(models=[modelo], freq='D')
    nf.fit(df=df_train_nf)
    y_hat = nf.predict(futr_df=df_val_nf)
    
    # Calcular métrica
    mae = mean_absolute_error(y_true, y_pred)
    return mae

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

## 📝 Convenciones de Nomenclatura

- **Archivos de métricas:** `resultados_metricas_{algoritmo}.csv`
- **Notebooks DL:** `06_DL_{Algoritmo}.ipynb`
- **Notebooks tree-based:** `05_{Algoritmo}_Forecasting.ipynb`
- **Nombres de algoritmos en métricas:** Mayúsculas (LSTM, GRU, NBEATS)

## 🚀 Pasos para Añadir un Nuevo Algoritmo DL

1. Duplicar `06_DL_LSTM.ipynb`
2. Renombrar a `06_DL_{NuevoAlgoritmo}.ipynb`
3. Cambiar el import del modelo:
   ```python
   from neuralforecast.models import NuevoAlgoritmo
   ```
4. Ajustar hiperparámetros específicos
5. Actualizar nombre en métricas:
   ```python
   metricas = calcular_metricas(..., name='NuevoAlgoritmo')
   ```
6. Guardar resultados:
   ```python
   df_metricas.to_csv('datos/resultados_metricas_nuevoalgoritmo.csv')
   ```
7. Ejecutar `07_Comparacion_Global.ipynb` para actualizar comparación

## 📦 Dependencias Requeridas

```bash
pip install neuralforecast optuna pandas numpy matplotlib seaborn scikit-learn
```

## 💡 Recomendaciones

1. **Normalización obligatoria** para DL: Los datos ya están normalizados en `df_train_dl.csv`
2. **Embedding layers** para producto: Mantener como entero, no usar One-Hot (894 categorías)
3. **Frecuencia temporal**: Asegurar que sea regular (diaria='D')
4. **Horizonte de predicción**: 30 días (último mes de test)
5. **Validación temporal**: Usar últimos 20% de train para validación
6. **Early stopping**: Usar `early_stop_patience_steps` para evitar overfitting
7. **Scalers guardados**: Usar `scaler_target.pkl` para desnormalizar predicciones finales

## 🔍 Troubleshooting

### Error: "unique_id not found"
- Verificar que se llamó a `preparar_datos_neuralforecast()`

### Error: "Irregular frequency"
- Asegurar que todos los productos tengan todas las fechas
- Usar `df.groupby('unique_id')['ds'].diff()` para verificar

### Predicciones NaN
- Verificar que input_size <= longitud mínima de serie temporal
- Filtrar productos con datos insuficientes usando `filtrar_productos_con_datos_suficientes()`

### Modelo no converge
- Reducir learning_rate (1e-4 en lugar de 1e-3)
- Aumentar max_steps
- Verificar que los datos estén normalizados

## 📞 Contacto

Para dudas o mejoras, contactar con el autor del proyecto.
