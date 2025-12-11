# TFMDS
Repositorio para Trabajo Fin de Master de Data Science

## Estructura del Proyecto

### 📂 `cuadernos/`
Notebooks de Jupyter con el desarrollo completo del proyecto:

- **`01_Creacion_datos.ipynb`**: Creación y preparación inicial del dataset
- **`02-EDA_datos.ipynb`**: Análisis Exploratorio de Datos (EDA)
- **`03_Preprocesamiento.ipynb`**: Limpieza y transformación de datos
- **`04_Baseline.ipynb`**: Modelos baseline de referencia (Naive, Media Móvil)
- **`05_ML_CatBoost.ipynb`**: Modelo de Machine Learning con CatBoost
- **`05_ML_LightGBM.ipynb`**: Modelo de Machine Learning con LightGBM
- **`05_ML_RF.ipynb`**: Modelo de Machine Learning con Random Forest
- **`05_ML_XGBoost.ipynb`**: Modelo de Machine Learning con XGBoost
- **`06_DL_DEEPAR.ipynb`**: Modelo de Deep Learning DeepAR
- **`06_DL_GRU.ipynb`**: Modelo de Deep Learning con GRU (Gated Recurrent Unit)
- **`06_DL_LSTM.ipynb`**: Modelo de Deep Learning con LSTM
- **`06_DL_NBEATS.ipynb`**: Modelo de Deep Learning N-BEATS
- **`06_DL_TFT_GPU.ipynb`**: Modelo Temporal Fusion Transformer (GPU)
- **`06_DL_VanillaTransformer_GPU.ipynb`**: Transformer Vanilla (GPU)
- **`07_Comparativa_modelos.ipynb`**: Comparación exhaustiva de todos los modelos
- **`08_Analisis_Stock.ipynb`**: Análisis de costes de inventario y stock de seguridad

### 📊 `datos/`
Datasets y archivos de resultados:

**Datos originales:**
- **`Ventas.csv`**: Datos históricos de ventas
- **`Stock.csv`**: Información de inventario
- **`STDatosVentasTienda.csv`**: Datos de ventas por tienda
- **`Calendario.csv`**: Información de calendario y festivos
- **`Promociones.csv`**: Registro de promociones
- **`DatosCicloAprovisionamiento.csv`**: Ciclos de reaprovisionamiento por producto
- **`DatosPrecioMedio.csv`**: Precios medios de productos

**Datos procesados:**
- **`df_train.csv`**: Dataset de entrenamiento para modelos ML
- **`df_test.csv`**: Dataset de test para modelos ML
- **`df_train_dl.csv`**: Dataset de entrenamiento para modelos DL
- **`df_train_dl_normalized.csv`**: Dataset de entrenamiento normalizado para DL
- **`df_test_dl.csv`**: Dataset de test para modelos DL
- **`df_test_dl_normalized.csv`**: Dataset de test normalizado para DL
- **`df_test_catboost.csv`**: Predicciones del modelo CatBoost

**Resultados de modelos:**
- **`resultados_metricas_baseline.csv`**: Métricas de modelos baseline
- **`resultados_metricas_catboost.csv`**: Métricas del modelo CatBoost
- **`resultados_metricas_lightgbm.csv`**: Métricas del modelo LightGBM
- **`resultados_metricas_rf.csv`**: Métricas del modelo Random Forest
- **`resultados_metricas_xgboost.csv`**: Métricas del modelo XGBoost
- **`resultados_metricas_deepar.csv`**: Métricas del modelo DeepAR
- **`resultados_metricas_GRU.csv`**: Métricas del modelo GRU
- **`resultados_metricas_lstm.csv`**: Métricas del modelo LSTM
- **`resultados_metricas_NBEATS.csv`**: Métricas del modelo N-BEATS
- **`resultados_metricas_TFT_gpu.csv`**: Métricas del modelo TFT
- **`resultados_metricas_transformer_gpu.csv`**: Métricas del Transformer

### 📚 `documentacion/`
Documentación técnica y referencias:

- **`2016_yamazaki_StockSeguridad.pdf`**: Paper de referencia para cálculo de stock de seguridad (Yamazaki, 2015)
- **`EstimacionCostesStock.docx`**: Documentación metodológica para estimación de costes
- **`EstimacionCostesStock.pdf`**: Versión PDF de la documentación de costes

### 🔧 `lib/`
Librería de utilidades reutilizables:

- **`__init__.py`**: Inicialización del paquete Python
- **`utils.py`**: Funciones de utilidad general
- **`metricas.py`**: Funciones para cálculo de métricas de evaluación
- **`graficos.py`**: Funciones para visualización de modelos ML
- **`graficos_dl.py`**: Funciones para visualización de modelos DL
- **`dl_utils.py`**: Utilidades específicas para modelos de Deep Learning
- **`EJEMPLO_IMPORT.txt`**: Ejemplo de cómo importar las utilidades

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

Los notebooks están numerados secuencialmente. Se recomienda ejecutarlos en orden para reproducir el análisis completo.
