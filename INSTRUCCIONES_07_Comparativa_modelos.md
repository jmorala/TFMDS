# 📊 Cuaderno 07_Comparativa_modelos.ipynb

## Descripción General

Cuaderno comprehensive para comparar el desempeño de **todos los algoritmos** de Machine Learning y Deep Learning aplicados al pronóstico de series temporales de ventas.

**Ubicación:** `cuadernos/07_Comparativa_modelos.ipynb`

---

## 🎯 Características Principales

### 1. **Carga y Consolidación de Resultados** (Sección 1)
- Carga automática de todos los CSV con resultados (`resultados_metricas_*.csv`)
- Consolidación en un único dataset
- Estadísticas descriptivas de los datos

**Algoritmos soportados:**
- Baseline: `NAIVE`, `MEDIA7`
- Tree-based: `RF`, `XGBoost`, `LightGBM`, `CatBoost`
- Deep Learning: `DEEPAR`, `TRANSFORMER`, `LSTM`, `GRU`, etc.

### 2. **Métricas de Comparación** (Sección 2)
Métricas clave:
- **MAE**: Error Medio Absoluto
- **RMSE**: Raíz del Error Cuadrático Medio
- **R²**: Coeficiente de Determinación
- **MAPE (%)**: Error Porcentual Medio Absoluto
- **SMAPE (%)**: Error Porcentual Medio Simétrico Absoluto

### 3. **Tablas Comparativas Elegantes** (Secciones 3-5)
- ✅ **Tabla Global**: Comparación a nivel agregado
- ✅ **Tablas por Cluster**: Análisis segmentado
- ✅ **Tablas por Producto**: Granularidad máxima

**Formato:** Colores dinámicos
- 🟢 Verde = Mejor desempeño
- 🔴 Rojo = Peor desempeño

### 4. **Heatmap de Rankings** (Sección 6)
- Matriz de posiciones (1 = mejor, N = peor)
- Rankings por métrica individual
- Puntuación agregada total

### 5. **Visualizaciones Gráficas**
- 📈 **Gráficos de Barras**: Comparativa de métricas por algoritmo
- 🎯 **Heatmaps**: Rankings con codificación de colores
- 🔄 **Gráficos por Cluster**: Rendimiento en cada segmento
- 📊 **Boxplots**: Distribución de métricas
- 🌍 **Scatter Plot**: RMSE vs R² (Trade-off)
- 🎪 **Radar Charts**: Perfil de fortalezas/debilidades

### 6. **Análisis de Escalabilidad** (Sección 9)
- Cómo varía el rendimiento según granularidad
- Global → Cluster → Producto
- Estabilidad y consistencia de modelos

### 7. **Matriz de Comparación Relativa** (Sección 11)
- Normalización respecto al mejor algoritmo
- Valores en porcentaje (100% = mejor)
- Fácil identificación de brechas de desempeño

### 8. **Resumen Ejecutivo** (Sección 12)
- 🏆 Mejor algoritmo global
- 📈 Mejor correlación (R²)
- 📊 Mejor error porcentual (MAPE)
- ⚖️ Algoritmo más estable
- 🎯 Ranking top 5
- 📌 Análisis por familia de algoritmos

---

## 🚀 Cómo Usar el Cuaderno

### Paso 1: Preparar Datos
Asegúrese de que tenga los siguientes archivos en `datos/`:
```
resultados_metricas_baseline.csv
resultados_metricas_rf.csv
resultados_metricas_xgboost.csv
resultados_metricas_lightgbm.csv
resultados_metricas_catboost.csv
resultados_metricas_deepar.csv
resultados_metricas_transformer.csv
(y otros modelos)
```

### Paso 2: Ejecutar el Cuaderno
1. Abrir `07_Comparativa_modelos.ipynb`
2. Ejecutar las celdas en orden
3. Las visualizaciones aparecerán automáticamente

### Paso 3: Interpretar Resultados
- **Tablas coloreadas**: Identificar rápidamente mejor/peor
- **Gráficos**: Visualizar tendencias y patrones
- **Rankings**: Comparación directa de algoritmos
- **Recomendaciones**: Guía para selección de modelos

---

## 📊 Estructura del Cuaderno

```
07_Comparativa_modelos.ipynb
├── 🔧 Setup y configuración
├── 📂 SECCIÓN 1: Carga de resultados
├── 🎯 SECCIÓN 2: Definición de métricas
├── 🌍 SECCIÓN 3: Tabla global
├── 🎯 SECCIÓN 4: Comparativa por cluster
├── 🏆 SECCIÓN 5: Comparativa por productos
├── 🔥 SECCIÓN 6: Heatmap de rankings
├── 📈 SECCIÓN 7: Gráficos comparativos
├── 🎯 SECCIÓN 8: Gráficos por cluster
├── ⚖️ SECCIÓN 9: Análisis de escalabilidad
├── 📦 SECCIÓN 10: Boxplots de distribución
├── 📊 SECCIÓN 11: Matriz relativa
├── 📋 SECCIÓN 12: Resumen ejecutivo
├── 🎪 SECCIÓN 13: Radar charts
├── 🔄 SECCIÓN 14: Scatter RMSE vs R²
├── 💾 SECCIÓN 15: Exportar CSV
└── 📋 SECCIÓN 16: Conclusiones
```

---

## 🎯 Salidas Generadas

El cuaderno genera los siguientes archivos en `datos/`:
```
✓ comparativa_rankings_global.csv
✓ comparativa_global_metricas.csv
✓ comparativa_relativa.csv
✓ comparativa_escalabilidad.csv
```

---

## 💡 Casos de Uso

### Caso 1: Seleccionar el Mejor Modelo
→ Revisar **Sección 12 (Resumen Ejecutivo)**
→ Usar el algoritmo recomendado

### Caso 2: Comparar Desempeño por Cluster
→ Revisar **Sección 4 y 8 (Gráficos por Cluster)**
→ Identificar clusters problemáticos

### Caso 3: Evaluar Estabilidad
→ Revisar **Sección 9 (Escalabilidad)**
→ Elegiralgormo con menor variación

### Caso 4: Entender Trade-offs
→ Revisar **Sección 14 (Scatter RMSE vs R²)**
→ Balancear entre precisión y correlación

### Caso 5: Análisis de Fortalezas
→ Revisar **Sección 13 (Radar Charts)**
→ Comparar perfiles de algoritmos

---

## 🔍 Métricas Clave a Interpretar

| Métrica | Mejor | Interpretación |
|---------|-------|----------------|
| **RMSE** | Mínimo ↓ | Error promedio en escala original |
| **MAE** | Mínimo ↓ | Error absoluto promedio |
| **R²** | Máximo ↑ | Varianza explicada (0-1) |
| **MAPE %** | Mínimo ↓ | Error porcentual (target ~30%) |
| **SMAPE %** | Mínimo ↓ | Error simétrico (más robusto) |

---

## ⚙️ Personalización

### Cambiar Métricas de Comparación
```python
METRICAS_COMPARACION = ['MAE', 'RMSE', 'R2', 'MAPE (%)', 'SMAPE (%)']
```

### Cambiar Colores
```python
COLOR_BEST = '#2ecc71'    # Verde
COLOR_WORST = '#e74c3c'   # Rojo
```

### Filtrar Algoritmos Específicos
```python
df_filtrado = df_global[df_global['Algoritmo'].isin(['XGBoost', 'LightGBM'])]
```

---

## 📈 Recomendaciones Finales

### Para Máxima Precisión
→ Usar el algoritmo con menor RMSE global

### Para Estabilidad
→ Usar algoritmo con menor desviación estándar entre niveles

### Para Producción
→ Ensemble de top-3 algoritmos

### Para Nuevos Datos
→ Algoritmos con mejor generalización (mayor R²)

---

## 🐛 Troubleshooting

**Problema:** "FileNotFoundError" al cargar CSV
- **Solución:** Ejecutar los modelos primero (05_*, 06_*) para generar archivos

**Problema:** Gráficos no se muestran
- **Solución:** Usar Jupyter con plotly habilitado, o exportar como HTML

**Problema:** Datos inconsistentes en tabla
- **Solución:** Verificar que todos los CSV tienen las mismas columnas

---

## 📞 Contacto y Apoyo

Para más información:
- Revisar notebooks de modelos individuales (05_*, 06_*)
- Consultar documentación de librerías (pandas, plotly, seaborn)
- Verificar estructura de datos en `datos/`

---

**Creado:** 2025-12-05  
**Versión:** 1.0  
**Estado:** ✅ Completado y Funcional
