# Optimización de Hiperparámetros LSTM con Optuna

## 📋 Descripción

El cuaderno `06_DL_LSTM_Optuna.ipynb` implementa la búsqueda automática de los mejores hiperparámetros para el modelo LSTM usando **Optuna**, un framework de optimización de hiperparámetros de última generación.

## 🎯 Objetivo

Encontrar la configuración óptima de hiperparámetros del modelo LSTM que minimice el error MAE en el conjunto de validación, mejorando así el rendimiento predictivo.

## 🔧 Hiperparámetros Optimizados

El proceso de optimización explora los siguientes hiperparámetros:

| Hiperparámetro | Rango/Valores | Descripción |
|----------------|---------------|-------------|
| `input_size` | 30-90 (paso 15) | Ventana temporal de entrada |
| `encoder_hidden_size` | [64, 128, 256] | Tamaño de la capa oculta del encoder |
| `encoder_n_layers` | 1-3 | Número de capas LSTM del encoder |
| `decoder_hidden_size` | [64, 128, 256] | Tamaño de la capa oculta del decoder |
| `decoder_layers` | 1-3 | Número de capas del decoder |
| `context_size` | 5-15 (paso 5) | Contexto adicional |
| `learning_rate` | 1e-4 a 1e-2 (log) | Tasa de aprendizaje |
| `batch_size` | [16, 32, 64] | Tamaño del batch |
| `scaler_type` | [standard, robust, minmax] | Tipo de escalado |
| `max_steps` | 300-800 (paso 100) | Pasos máximos de entrenamiento |

## 🚀 Uso

### 1. Instalación de Optuna

```powershell
pip install optuna plotly
```

### 2. Configuración

Ajustar el número de trials según el tiempo disponible:

```python
N_TRIALS = 20  # Para pruebas rápidas (30-60 min)
# N_TRIALS = 50  # Para búsqueda más exhaustiva (2-4 horas)
# N_TRIALS = 100  # Para búsqueda completa (4-8 horas)
```

### 3. Ejecución

Ejecutar todas las celdas del cuaderno secuencialmente. El proceso:

1. **Prepara los datos** en formato NeuralForecast
2. **Divide train/validation** (últimos 30 días del train para validación)
3. **Define función objetivo** que entrena y evalúa cada configuración
4. **Ejecuta optimización** con el número de trials especificado
5. **Visualiza resultados** de la búsqueda
6. **Entrena modelo final** con mejores hiperparámetros en train completo
7. **Evalúa en test** y guarda resultados

## 📊 Salidas Generadas

### 1. Archivos CSV

- `datos/resultados_metricas_lstm_optuna.csv`: Métricas del modelo optimizado en test
- `datos/mejores_hiperparametros_lstm_optuna.csv`: Mejores hiperparámetros encontrados
- `datos/historial_trials_optuna.csv`: Historial completo de todos los trials

### 2. Visualizaciones Interactivas

- **Historia de Optimización**: Evolución del MAE a lo largo de los trials
- **Importancia de Parámetros**: Qué hiperparámetros tienen mayor impacto
- **Coordenadas Paralelas**: Relación entre hiperparámetros y rendimiento
- **Gráficos de predicción**: Igual que el cuaderno base

## 🔍 Metodología

### Algoritmo TPE (Tree-structured Parzen Estimator)

Optuna usa TPE como sampler por defecto, que:
- Construye un modelo probabilístico de la función objetivo
- Explora inteligentemente el espacio de hiperparámetros
- Balancea exploración y explotación
- Es más eficiente que búsqueda aleatoria o grid search

### Validación Temporal

- **Train**: Primeros 330 días (hasta día 330)
- **Validation**: Días 331-360 (últimos 30 del train original)
- **Test**: Días 361-390 (conjunto de test original)

Esto evita data leakage y simula un escenario realista de predicción futura.

## ⚡ Rendimiento y Tiempo

### Estimación de Tiempos (CPU i7, 16GB RAM)

- **1 trial**: ~2-3 minutos
- **10 trials**: ~20-30 minutos
- **20 trials**: ~40-60 minutos
- **50 trials**: ~2-3 horas
- **100 trials**: ~4-6 horas

**Nota**: Los tiempos varían según:
- Hardware disponible
- Complejidad de la configuración probada
- Tamaño de los datos

## 📈 Interpretación de Resultados

### 1. Historia de Optimización

Muestra cómo mejora el mejor valor encontrado a lo largo de los trials. Idealmente:
- Debe converger (estabilizarse)
- Si sigue bajando, considerar más trials

### 2. Importancia de Parámetros

Indica qué hiperparámetros tienen mayor impacto en el rendimiento:
- **Alta importancia**: Ajustar con mayor precisión
- **Baja importancia**: Puede usar valores por defecto

### 3. Coordenadas Paralelas

Visualiza la relación entre todos los hiperparámetros y el MAE:
- Líneas azules: Mejores configuraciones
- Líneas rojas: Peores configuraciones
- Identifica patrones y combinaciones óptimas

## 🔄 Comparación con Modelo Base

Para evaluar la mejora obtenida:

1. Ejecutar `06_DL_LSTM.ipynb` (modelo base con hiperparámetros manuales)
2. Ejecutar `06_DL_LSTM_Optuna.ipynb` (modelo optimizado)
3. Comparar métricas en test:

```python
# Ejemplo de comparación
print("Modelo Base vs Optuna:")
print(f"MAE: {mae_base:.4f} vs {mae_optuna:.4f}")
print(f"Mejora: {((mae_base - mae_optuna) / mae_base * 100):.2f}%")
```

## 💡 Consejos y Mejores Prácticas

### 1. Para Pruebas Iniciales
- Usar `N_TRIALS = 10-20`
- Reducir `max_steps` a 200-400
- Verificar que el proceso funciona correctamente

### 2. Para Búsqueda Seria
- Usar `N_TRIALS = 50-100`
- Ejecutar durante la noche o fin de semana
- Guardar estudio con `study.trials_dataframe()` periódicamente

### 3. Optimizaciones Adicionales
- **Poda temprana**: Optuna puede detener trials que no prometen
- **Paralelización**: Si tienes múltiples GPUs, usar `n_jobs > 1`
- **Warm start**: Continuar optimización desde un estudio anterior

### 4. Debugging
- Si un trial falla, retorna `1e6` (MAE muy alto)
- Revisar mensajes de error en consola
- Verificar que todas las columnas exógenas existen

## 🎓 Referencias

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [NeuralForecast Documentation](https://nixtla.github.io/neuralforecast/)
- [TPE Algorithm Paper](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)

## ⚠️ Limitaciones

1. **Tiempo de ejecución**: Puede tardar varias horas
2. **Recursos**: Requiere suficiente RAM y CPU
3. **Sobreajuste**: Validar siempre en test independiente
4. **Espacio de búsqueda**: Los rangos definidos pueden no incluir el óptimo global

## 🔮 Próximos Pasos

1. **Aplicar a GRU**: Usar mismos rangos en modelo GRU
2. **Ensemble**: Combinar múltiples configuraciones top
3. **Features**: Optimizar también selección de features
4. **Multi-objetivo**: Optimizar MAE y tiempo de entrenamiento simultáneamente
5. **Cross-validation**: Múltiples folds temporales para mayor robustez
