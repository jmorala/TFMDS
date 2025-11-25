# Instalación de Dependencias - Deep Learning

## Dependencias Base (ya instaladas)
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
optuna
```

## Dependencias para Modelos Tree-Based (ya instaladas)
```bash
xgboost
lightgbm
catboost
```

## Dependencias para Deep Learning (INSTALAR)

### NeuralForecast (Nixtla)
Framework unificado para modelos de Deep Learning en series temporales.

```bash
pip install neuralforecast
```

**Incluye:**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- NBEATS (Neural Basis Expansion Analysis)
- DeepAR (Amazon's probabilistic forecasting)
- TFT (Temporal Fusion Transformer)
- VanillaTransformer
- TCN (Temporal Convolutional Network)
- DilatedRNN
- Y más...

### PyTorch (dependencia de NeuralForecast)
Se instala automáticamente con neuralforecast, pero puedes instalarlo manualmente:

```bash
# Para CPU
pip install torch

# Para GPU (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Verificación de Instalación

Ejecuta este script en Python para verificar:

```python
import sys
print("Python version:", sys.version)

# Dependencias base
try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__}")
except ImportError:
    print("✗ pandas no instalado")

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError:
    print("✗ numpy no instalado")

try:
    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ matplotlib no instalado")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError:
    print("✗ scikit-learn no instalado")

try:
    import optuna
    print(f"✓ optuna {optuna.__version__}")
except ImportError:
    print("✗ optuna no instalado")

# Tree-based
try:
    import xgboost as xgb
    print(f"✓ xgboost {xgb.__version__}")
except ImportError:
    print("✗ xgboost no instalado")

try:
    import lightgbm as lgb
    print(f"✓ lightgbm {lgb.__version__}")
except ImportError:
    print("✗ lightgbm no instalado")

try:
    import catboost
    print(f"✓ catboost {catboost.__version__}")
except ImportError:
    print("✗ catboost no instalado")

# Deep Learning
try:
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"  CUDA disponible: {torch.cuda.is_available()}")
except ImportError:
    print("✗ torch no instalado")

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM, GRU, NBEATS
    print(f"✓ neuralforecast instalado")
    print(f"  Modelos disponibles: LSTM, GRU, NBEATS, DeepAR, TFT, etc.")
except ImportError:
    print("✗ neuralforecast no instalado")

print("\n" + "="*60)
print("Verificación completa")
print("="*60)
```

## Instalación Completa (Google Colab)

Si trabajas en Google Colab, ejecuta esto al inicio de tu notebook:

```python
# Instalar dependencias faltantes
!pip install neuralforecast optuna lightgbm catboost

# Verificar instalación
import neuralforecast
print(f"✓ NeuralForecast {neuralforecast.__version__} instalado")
```

## Instalación Completa (Entorno Local)

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv_dl
source venv_dl/bin/activate  # En Windows: venv_dl\Scripts\activate

# Instalar todas las dependencias
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn
pip install optuna
pip install xgboost lightgbm catboost
pip install torch
pip install neuralforecast

# O instalar desde requirements.txt
pip install -r requirements.txt
```

## Archivo requirements.txt

Crea un archivo `requirements.txt` con:

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
optuna>=3.4.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
torch>=2.0.0
neuralforecast>=1.6.0
```

Luego instala con:
```bash
pip install -r requirements.txt
```

## Notas Importantes

### GPU vs CPU
- **NeuralForecast** puede usar GPU si PyTorch está instalado con soporte CUDA
- Para CPU: la instalación por defecto funciona
- Para GPU: verificar compatibilidad CUDA con tu tarjeta gráfica

### Compatibilidad de Versiones
- Python: >= 3.8
- PyTorch: >= 2.0 (recomendado)
- NeuralForecast: >= 1.6.0

### Memoria RAM
Los modelos DL requieren más memoria que tree-based:
- Mínimo: 8 GB RAM
- Recomendado: 16 GB RAM
- Ideal: 32 GB RAM + GPU

### Troubleshooting

#### Error: "No module named 'neuralforecast'"
```bash
pip install neuralforecast
```

#### Error: "CUDA not available"
Es normal si no tienes GPU. Los modelos funcionan en CPU (más lento).

#### Error al importar torch
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### Error: "Memory Error"
- Reducir `batch_size` en los modelos
- Reducir `input_size` (ventana de entrada)
- Procesar menos productos simultáneamente

## Enlaces Útiles

- **NeuralForecast Documentation**: https://nixtla.github.io/neuralforecast/
- **NeuralForecast GitHub**: https://github.com/Nixtla/neuralforecast
- **PyTorch**: https://pytorch.org/
- **Optuna**: https://optuna.org/

## Actualizar Dependencias

Para actualizar a las últimas versiones:

```bash
pip install --upgrade neuralforecast
pip install --upgrade torch
pip install --upgrade optuna
```
