#!/usr/bin/env python3
"""
fix_notebook_widgets.py

Limpia o repara metadata.widgets en un notebook Jupyter para evitar
el error "the 'state' key is missing from 'metadata.widgets'".

Uso:
    # Para ELIMINAR metadata.widgets (por defecto)
    python fix_notebook_widgets.py cuadernos/06_DL_TFT_GPU.ipynb

    # Para AÑADIR 'state' vacío a cada entrada en metadata.widgets (no lo recomendado si no usas widgets)
    python fix_notebook_widgets.py --fix cuadernos/06_DL_TFT_GPU.ipynb
"""
import nbformat
import argparse
from pathlib import Path
import sys

def remove_widgets(nb):
    if 'widgets' in nb.metadata:
        nb.metadata.pop('widgets', None)
        return True
    return False

def fix_widgets(nb):
    changed = False
    widgets = nb.metadata.get('widgets')
    if isinstance(widgets, dict):
        for k, v in list(widgets.items()):
            if not isinstance(v, dict):
                nb.metadata['widgets'][k] = {'state': {}}
                changed = True
            else:
                if 'state' not in v:
                    nb.metadata['widgets'][k]['state'] = {}
                    changed = True
    return changed

def main():
    parser = argparse.ArgumentParser(description="Quitar o reparar metadata.widgets en un .ipynb")
    parser.add_argument('notebook', type=Path, help='Ruta al notebook (.ipynb)')
    parser.add_argument('--fix', action='store_true', help="Añadir 'state': {} en vez de eliminar metadata.widgets")
    args = parser.parse_args()

    nb_path = args.notebook
    if not nb_path.exists():
        print(f"ERROR: No existe {nb_path}", file=sys.stderr)
        sys.exit(2)

    nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
    changed = False
    if args.fix:
        changed = fix_widgets(nb)
        if changed:
            print("Se añadió 'state': {} a entradas de metadata.widgets (si faltaban).")
        else:
            print("No se detectaron cambios al intentar 'fix' (o metadata.widgets no presente).")
    else:
        changed = remove_widgets(nb)
        if changed:
            print("Se eliminó metadata.widgets del notebook.")
        else:
            print("No se encontró metadata.widgets para eliminar.")

    # Guardar sólo si hubo cambio
    if changed:
        nbformat.write(nb, nb_path)
        print(f"Notebook guardado: {nb_path}")
    else:
        print("No se realizaron cambios. El archivo no fue modificado.")

if __name__ == "__main__":
    main()