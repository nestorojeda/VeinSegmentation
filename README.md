# Segmentación de venas en imagen infrarroja

Repositorio con el código del trabajo de fin de grado titulado 'Segmentación de venas en imagen infrarroja'.

## Requerimientos

- Python 3.8 

## Montar proyecto

```
python -m venv ./venv # Crear entorno virtual
pip install -r requirements.txt # Instalar dependencias
``` 

## Ejecutable

Comando para montar el ejecutable:
```
pyinstaller .\src\app.py --additional-hooks-dir=hooks --hidden-import="skimage.filters.rank.core_cy_3d" --onefile
```


Alumno: Néstor Ojeda González

Tutor: Agustín Rafael Trujillo Pino
