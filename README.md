# Proyecto integrador: ciclo local de MLOps

**Estudiantes:** Rubiel Velásquez y Valentina Figueroa

Este proyecto desarrolla un flujo de aprendizaje automático para clasificar registros del conjunto **Heart Failure Prediction**. Se recorren las etapas solicitadas en el ejercicio: revisión de datos, prevención de fuga de información, comparación y evaluación de modelos, API de predicción, contenedor, manifiestos de Kubernetes, pruebas automáticas y reporte de deriva.

El ejercicio es académico. El modelo no está validado para diagnóstico ni para apoyar decisiones clínicas.

## Datos y objetivo

El archivo data/raw/heart.csv contiene 918 registros y 12 columnas. La variable HeartDisease es la respuesta: 1 indica presencia de enfermedad en el registro y 0 indica ausencia. Las otras 11 columnas describen edad, sexo, presión, colesterol, glucosa, frecuencia cardíaca y características relacionadas con dolor torácico, electrocardiograma y ejercicio.

El conjunto no tiene celdas vacías ni filas duplicadas. Sin embargo, contiene 172 ceros en Cholesterol y uno en RestingBP. Como esos valores no son mediciones clínicas válidas, se marcan como faltantes y se imputan con la mediana calculada dentro del pipeline de entrenamiento. El CSV se conserva localmente y queda excluido del control de versiones por .gitignore.

La variable objetivo aparece como HeartDisease en el archivo entregado, por lo que se usa ese nombre en el código. El conjunto se obtuvo de [Heart Failure Prediction Dataset en Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

## Resultados del modelado

Se separó el 80 % de los datos para entrenamiento y el 20 % para la evaluación final, manteniendo proporciones similares de las dos clases. La búsqueda de hiperparámetros usó validación cruzada estratificada de cinco particiones. Todas las transformaciones se ajustan dentro del Pipeline, sin aprender información del conjunto de prueba.

| Posición por AUC CV | Modelo | AUC promedio CV | AUC en prueba | Exactitud en prueba |
| ---: | --- | ---: | ---: | ---: |
| 1 | Bosque aleatorio | 0.933 | 0.937 | 0.886 |
| 2 | Gradient Boosting | 0.930 | 0.931 | 0.908 |
| 3 | Regresión logística | 0.928 | 0.932 | 0.891 |
| 4 | SVC | 0.925 | 0.944 | 0.880 |
| 5 | K vecinos (KNN) | 0.920 | 0.947 | 0.897 |

El bosque aleatorio quedó primero por AUC promedio de validación cruzada y se guardó para la API. En la prueba reservada obtuvo AUC 0.937 y exactitud 0.886. La sensibilidad fue 0.941 para los registros con respuesta 1 y 0.817 para los registros con respuesta 0. Esto muestra que el resultado depende de la métrica observada: KNN tuvo el AUC más alto en esta partición y Gradient Boosting la mayor exactitud. La elección se basó en la validación cruzada, no en cuál modelo obtuvo el mejor número aislado en la prueba.

La notebook de fuga compara un AUC cercano a 1.00 cuando se introduce deliberadamente una variable calculada a partir de la respuesta con un AUC cercano a 0.95 en el flujo correcto. El primer valor es inválido como estimación de desempeño: el modelo recibe indirectamente la respuesta que se supone debe predecir.

## Organización del proyecto

~~~text
heart-disease-mlops/
├── app/
│   ├── api.py
│   ├── model.joblib
│   ├── modeling.py
│   ├── monitoring.py
│   └── train_model.py
├── data/raw/heart.csv
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb
│   └── 2_model_pipeline_cv.ipynb
├── tests/
├── .github/workflows/ci.yml
├── drift_report.html
└── README.md
~~~

## Preparación del entorno

Se requiere Python 3.12. Desde esta carpeta, se puede crear un entorno aislado e instalar las dependencias:

~~~powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
~~~

Si se usa uv, las mismas dependencias se pueden instalar con:

~~~powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
~~~

El entorno de desarrollo incluye las dependencias de la API, notebooks, pruebas, linting y monitoreo. La imagen Docker utiliza las versiones fijadas en docker/requirements.txt, incluida la misma versión de scikit-learn con la que se guarda el modelo.

## Ejecución de las notebooks y entrenamiento

Desde la carpeta principal del proyecto, iniciar JupyterLab y ejecutar las notebooks en orden:

~~~powershell
jupyter lab
~~~

La primera notebook revisa los datos, ilustra la fuga y compara cinco clasificadores. La segunda realiza la selección por validación cruzada, presenta matriz de confusión y curva ROC, e informa las métricas del modelo seleccionado. Esta segunda notebook guarda y verifica app/model.joblib.

El entrenamiento también se puede repetir desde la terminal:

~~~powershell
python -m app.train_model
~~~

## API de predicción

Con el modelo disponible, iniciar FastAPI:

~~~powershell
uvicorn app.api:app --reload
~~~

La documentación interactiva se abre en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) y el estado del servicio en [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health). El endpoint POST /predict espera las variables con nombre para evitar errores por su orden. Se pueden enviar null para presión o colesterol no disponibles.

## Pruebas y estilo

Las pruebas verifican el preprocesamiento y las respuestas de la API:

~~~powershell
pytest -q
flake8 app tests --max-line-length=100 --ignore=E203,W503
~~~

El flujo de GitHub Actions definido en .github/workflows/ci.yml instala dependencias, revisa el estilo y ejecuta las pruebas en cada envío de cambios o solicitud de integración. Para que se ejecute en GitHub, esta carpeta debe estar en un repositorio y el workflow debe subirse junto con el resto del proyecto.

## Docker y Kubernetes local

Con Docker Desktop instalado y en ejecución, construir y probar la API en un contenedor:

~~~powershell
docker build -t heart-api:latest -f docker/Dockerfile .
docker run --rm -p 8000:8000 heart-api:latest
~~~

El comando `docker run` mantiene el contenedor en primer plano; se puede detener con `Ctrl + C` antes de probar Kubernetes.

Con Docker Desktop en ejecución, Minikube y kubectl disponibles, iniciar el clúster local y construir la imagen dentro de Minikube:

~~~powershell
minikube start --driver=docker
minikube image build -t heart-api:latest -f docker/Dockerfile .
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/heart-model
kubectl get pods,services
~~~

El manifiesto define un Deployment con comprobaciones de salud y un Service de tipo LoadBalancer. Para probar la API sin abrir un túnel, mantener el siguiente comando en ejecución en una terminal:

~~~powershell
kubectl port-forward service/heart-service 8000:80
~~~

Con el reenvío activo, la documentación de la API está en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) y el estado del servicio en [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health). Detener el reenvío con `Ctrl + C`. En Windows, también se puede obtener una dirección externa para el LoadBalancer ejecutando `minikube tunnel` en otra terminal con permisos de administrador.

La API se verificó desde el clúster: el pod quedó en estado `Running` y `/health` respondió con `status: ok`.

## Monitoreo de deriva

El reporte incluido se genera con:

~~~powershell
python -m app.monitoring
~~~

El archivo drift_report.html compara las variables del conjunto de entrenamiento con las del conjunto de prueba, que aquí funciona como una aproximación de datos nuevos. La comparación sirve para practicar el monitoreo de cambios en los datos; no representa observaciones reales recibidas por un servicio en producción.

## Alcance y limitaciones

El flujo sirve para aprender cómo se conectan preparación, entrenamiento, servicio, empaquetado y monitoreo. El conjunto es pequeño y no se ha validado como muestra representativa de pacientes reales. Los resultados dependen de la partición usada y de la calidad de los datos; tampoco establecen relaciones causales. Una aplicación clínica requeriría evaluación externa, revisión de especialistas y controles adicionales.
