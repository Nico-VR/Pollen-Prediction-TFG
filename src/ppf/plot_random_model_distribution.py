import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import random
import os

def plot_random_model_distribution(csv_path: str, columnas: list[str], save_path: str = None):
    """
    Selecciona un modelo aleatorio de 'model_name' y grafica la distribución
    de las columnas especificadas (histograma + QQ-plot).
    Puede guardar los gráficos si se proporciona una ruta.
    
    Parameters
    ----------
    csv_path : str
        Ruta al CSV con resultados.
    columnas : list[str]
        Columnas a analizar (ej. ['start_dev', 'end_dev']).
    save_path : str, optional
        Carpeta donde guardar los gráficos. Si no se indica, solo se muestran.
    """
    # Leer CSV
    df = pd.read_csv(csv_path)
    
    # Seleccionar modelo aleatorio
    modelos = df['model_name'].unique()
    modelo_aleatorio = random.choice(modelos)
    df_modelo = df[df['model_name'] == modelo_aleatorio]
    
    print(f"Modelo seleccionado aleatoriamente: {modelo_aleatorio}")
    
    # Crear carpeta si se indica guardar y no existe
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # Graficar cada columna
    for col in columnas:
        valores = df_modelo[col].dropna()
        
        plt.figure(figsize=(10,4))
        
        # Histograma
        plt.subplot(1,2,1)
        sns.histplot(valores, kde=True, bins=20)
        plt.title(f'Histograma de {col} ({modelo_aleatorio})')
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        
        # QQ-plot
        plt.subplot(1,2,2)
        stats.probplot(valores, dist="norm", plot=plt)
        plt.title(f'QQ-plot de {col} ({modelo_aleatorio})')
        
        plt.tight_layout()
        
        # Guardar o mostrar
        if save_path:
            filename = f"{modelo_aleatorio}_{col}.png"
            plt.savefig(os.path.join(save_path, filename))
            plt.close()
            print(f"Gráfico guardado en: {os.path.join(save_path, filename)}")
        else:
            plt.show()