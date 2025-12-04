"""
Sistema de Detección de Anomalías en Datos de Energía
Utiliza Isolation Forest para identificar comportamientos anómalos
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Clase para detectar anomalías en datos de energía usando Isolation Forest
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Inicializa el detector de anomalías
        
        Parameters:
        -----------
        contamination : float, default=0.1
            Proporción esperada de anomalías en los datos (0.0 a 0.5)
        random_state : int, default=42
            Semilla para reproducibilidad
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, df):
        """
        Prepara las características para el modelo
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame con los datos de energía
            
        Returns:
        --------
        pandas.DataFrame : DataFrame con características preparadas
        """
        # Crear una copia para no modificar el original
        data = df.copy()
        
        # Convertir columnas numéricas explícitamente
        numeric_columns = [
            'Generación total',
            'Consumo total',
            'Autoconsumo',
            'Energía suministrada a la red',
            'Energía obtenida de la red'
        ]
        
        for col in numeric_columns:
            if col in data.columns:
                # Convertir a numérico, manejando errores
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Manejar diferentes nombres de columna de fecha
        fecha_col = None
        if 'Fecha' in data.columns:
            fecha_col = 'Fecha'
        elif 'Fecha y hora' in data.columns:
            fecha_col = 'Fecha y hora'
            data['Fecha'] = data['Fecha y hora']  # Crear columna estándar
        
        # Convertir fecha a datetime si es necesario
        if fecha_col or 'Fecha' in data.columns:
            if fecha_col:
                date_series = data[fecha_col]
            else:
                date_series = data['Fecha']
            
            # Intentar diferentes formatos de fecha
            data['Fecha'] = pd.to_datetime(date_series, format='%d.%m.%Y', errors='coerce')
            if data['Fecha'].isna().all():
                # Intentar sin formato específico
                data['Fecha'] = pd.to_datetime(date_series, errors='coerce')
            
            # Extraer características temporales
            data['Dia'] = data['Fecha'].dt.day
            data['Mes'] = data['Fecha'].dt.month
            data['DiaSemana'] = data['Fecha'].dt.dayofweek
            data['EsFinSemana'] = (data['DiaSemana'] >= 5).astype(int)
        
        # Calcular métricas derivadas que pueden indicar anomalías
        if 'Generación total' in data.columns and 'Consumo total' in data.columns:
            data['Ratio_Generacion_Consumo'] = data['Generación total'] / (data['Consumo total'] + 1e-6)
            data['Diferencia_Generacion_Consumo'] = data['Generación total'] - data['Consumo total']
        
        if 'Autoconsumo' in data.columns and 'Generación total' in data.columns:
            data['Eficiencia_Autoconsumo'] = data['Autoconsumo'] / (data['Generación total'] + 1e-6)
        
        if 'Energía suministrada a la red' in data.columns and 'Energía obtenida de la red' in data.columns:
            data['Balance_Red'] = data['Energía suministrada a la red'] - data['Energía obtenida de la red']
        
        # Seleccionar características numéricas para el modelo
        feature_columns = [
            'Generación total',
            'Consumo total',
            'Autoconsumo',
            'Energía suministrada a la red',
            'Energía obtenida de la red',
            'Ratio_Generacion_Consumo',
            'Diferencia_Generacion_Consumo',
            'Eficiencia_Autoconsumo',
            'Balance_Red',
            'Dia',
            'Mes',
            'DiaSemana',
            'EsFinSemana'
        ]
        
        # Filtrar columnas que existen en el DataFrame
        available_features = [col for col in feature_columns if col in data.columns]
        
        return data[available_features], data
    
    def train(self, df):
        """
        Entrena el modelo de Isolation Forest
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame con los datos de entrenamiento
        """
        print("🔧 Preparando características...")
        X, self.data_processed = self.prepare_features(df)
        
        print(f"📊 Características seleccionadas: {list(X.columns)}")
        print(f"📈 Forma de los datos: {X.shape}")
        
        # Normalizar los datos
        print("⚙️ Normalizando datos...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar el modelo
        print(f"🎯 Entrenando Isolation Forest (contamination={self.contamination})...")
        self.model.fit(X_scaled)
        
        self.is_fitted = True
        self.feature_names = X.columns.tolist()
        print("✅ Modelo entrenado exitosamente!")
        
    def predict(self, df):
        """
        Predice anomalías en nuevos datos
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame con los datos a evaluar
            
        Returns:
        --------
        numpy.ndarray : Predicciones (-1 para anomalía, 1 para normal)
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero usando train()")
        
        X, _ = self.prepare_features(df)
        
        # Asegurar que las columnas coincidan
        X = X[self.feature_names]
        
        # Normalizar
        X_scaled = self.scaler.transform(X)
        
        # Predecir
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, df):
        """
        Obtiene el score de anomalía (más negativo = más anómalo)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame con los datos a evaluar
            
        Returns:
        --------
        numpy.ndarray : Scores de anomalía
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero usando train()")
        
        X, _ = self.prepare_features(df)
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Obtener scores (más negativo = más anómalo)
        scores = self.model.score_samples(X_scaled)
        
        return scores
    
    def get_anomalies(self, df, threshold_percentile=10):
        """
        Identifica anomalías y retorna un DataFrame con los resultados
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame con los datos
        threshold_percentile : int, default=10
            Percentil para determinar el umbral de anomalía (0-100)
            
        Returns:
        --------
        pandas.DataFrame : DataFrame con columnas adicionales de anomalía
        """
        predictions = self.predict(df)
        scores = self.predict_proba(df)
        
        # Crear DataFrame de resultados
        results = df.copy()
        results['Es_Anomalia'] = (predictions == -1)
        results['Score_Anomalia'] = scores
        results['Severidad'] = pd.cut(
            scores,
            bins=[-np.inf, np.percentile(scores, threshold_percentile), np.inf],
            labels=['Alta', 'Normal']
        )
        
        return results
    
    def visualize_anomalies(self, results_df, save_path=None):
        """
        Visualiza las anomalías detectadas
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame con los resultados de anomalías
        save_path : str, optional
            Ruta para guardar las visualizaciones
        """
        anomalies = results_df[results_df['Es_Anomalia'] == True]
        normal = results_df[results_df['Es_Anomalia'] == False]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Anomalías en Datos de Energía', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Generación vs Consumo
        ax1 = axes[0, 0]
        ax1.scatter(normal['Consumo total'], normal['Generación total'], 
                   alpha=0.5, label='Normal', color='blue', s=50)
        ax1.scatter(anomalies['Consumo total'], anomalies['Generación total'], 
                   alpha=0.8, label='Anomalía', color='red', s=100, marker='x')
        ax1.set_xlabel('Consumo Total (Wh)')
        ax1.set_ylabel('Generación Total (Wh)')
        ax1.set_title('Generación vs Consumo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Serie temporal de scores
        ax2 = axes[0, 1]
        fecha_col = 'Fecha' if 'Fecha' in results_df.columns else ('Fecha y hora' if 'Fecha y hora' in results_df.columns else None)
        if fecha_col:
            dates = pd.to_datetime(results_df[fecha_col], format='%d.%m.%Y', errors='coerce')
            if dates.isna().all():
                dates = pd.to_datetime(results_df[fecha_col], errors='coerce')
            if not dates.isna().all():
                ax2.plot(dates, results_df['Score_Anomalia'], alpha=0.6, color='blue', linewidth=1)
                anomaly_dates = dates[results_df['Es_Anomalia'] == True]
                anomaly_scores = results_df.loc[results_df['Es_Anomalia'] == True, 'Score_Anomalia']
                if len(anomaly_dates) > 0:
                    ax2.scatter(anomaly_dates, anomaly_scores, color='red', s=100, marker='x', 
                               label='Anomalías', zorder=5)
                ax2.set_xlabel('Fecha')
                ax2.set_ylabel('Score de Anomalía')
                ax2.set_title('Score de Anomalía a lo Largo del Tiempo')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Gráfico 3: Distribución de scores
        ax3 = axes[1, 0]
        ax3.hist(results_df['Score_Anomalia'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(np.percentile(results_df['Score_Anomalia'], 
                                 results_df['Es_Anomalia'].sum() / len(results_df) * 100),
                   color='red', linestyle='--', linewidth=2, label='Umbral')
        ax3.set_xlabel('Score de Anomalía')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Scores de Anomalía')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Resumen de anomalías
        ax4 = axes[1, 1]
        anomaly_counts = results_df['Es_Anomalia'].value_counts()
        colors = ['green', 'red']
        ax4.pie(anomaly_counts.values, labels=['Normal', 'Anomalía'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Distribución de Anomalías')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualización guardada en: {save_path}")
        
        plt.show()


def main():
    """
    Función principal para ejecutar la detección de anomalías
    """
    print("=" * 60)
    print("🚀 SISTEMA DE DETECCIÓN DE ANOMALÍAS - ISOLATION FOREST")
    print("=" * 60)
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    try:
        # Intentar cargar desde CSV, saltando la segunda fila si tiene unidades
        df = pd.read_csv('datos_energia.csv', encoding='utf-8', skiprows=[1])
        print(f"✅ Datos cargados: {len(df)} registros")
        
        # Limpiar nombres de columnas (remover espacios extra)
        df.columns = df.columns.str.strip()
        
        # Convertir columnas numéricas explícitamente
        numeric_columns = [
            'Generación total',
            'Consumo total',
            'Autoconsumo',
            'Energía suministrada a la red',
            'Energía obtenida de la red'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
    except FileNotFoundError:
        print("⚠️ Archivo 'datos_energia.csv' no encontrado.")
        print("💡 Creando datos de ejemplo...")
        # Crear datos de ejemplo basados en la estructura proporcionada
        df = create_sample_data()
        df.to_csv('datos_energia.csv', index=False, encoding='utf-8')
        print("✅ Archivo de ejemplo creado: 'datos_energia.csv'")
    except Exception as e:
        # Si falla al saltar la segunda fila, intentar sin saltar
        print(f"⚠️ Error al cargar con skiprows: {e}")
        print("🔄 Intentando cargar sin saltar filas...")
        df = pd.read_csv('datos_energia.csv', encoding='utf-8')
        # Limpiar filas que no sean numéricas en las columnas numéricas
        numeric_columns = [
            'Generación total',
            'Consumo total',
            'Autoconsumo',
            'Energía suministrada a la red',
            'Energía obtenida de la red'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Eliminar filas con valores NaN en columnas numéricas (probablemente la fila de unidades)
        df = df.dropna(subset=numeric_columns[:2])  # Al menos las dos primeras deben tener valores
        print(f"✅ Datos cargados después de limpieza: {len(df)} registros")
    
    print("\n📋 Primeras filas del dataset:")
    print(df.head())
    print(f"\n📊 Información del dataset:")
    print(df.info())
    
    # Inicializar detector
    print("\n" + "=" * 60)
    contamination = 0.1  # Ajustar según necesidad (10% de anomalías esperadas)
    detector = AnomalyDetector(contamination=contamination)
    
    # Entrenar modelo
    detector.train(df)
    
    # Detectar anomalías
    print("\n" + "=" * 60)
    print("🔍 Detectando anomalías...")
    results = detector.get_anomalies(df)
    
    # Mostrar resultados
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DE DETECCIÓN DE ANOMALÍAS")
    print("=" * 60)
    
    num_anomalies = results['Es_Anomalia'].sum()
    num_normal = len(results) - num_anomalies
    percentage = (num_anomalies / len(results)) * 100
    
    print(f"\n✅ Registros normales: {num_normal} ({100-percentage:.2f}%)")
    print(f"⚠️ Anomalías detectadas: {num_anomalies} ({percentage:.2f}%)")
    
    if num_anomalies > 0:
        print("\n🚨 DETALLES DE ANOMALÍAS DETECTADAS:")
        print("-" * 60)
        anomalies_df = results[results['Es_Anomalia'] == True].sort_values('Score_Anomalia')
        # Seleccionar columnas disponibles
        display_cols = ['Score_Anomalia']
        fecha_col = 'Fecha' if 'Fecha' in anomalies_df.columns else ('Fecha y hora' if 'Fecha y hora' in anomalies_df.columns else None)
        if fecha_col:
            display_cols.insert(0, fecha_col)
        if 'Generación total' in anomalies_df.columns:
            display_cols.append('Generación total')
        if 'Consumo total' in anomalies_df.columns:
            display_cols.append('Consumo total')
        print(anomalies_df[display_cols].to_string(index=False))
        
        # Guardar resultados
        results.to_csv('resultados_anomalias.csv', index=False, encoding='utf-8')
        print(f"\n💾 Resultados guardados en: 'resultados_anomalias.csv'")
    
    # Visualizar
    print("\n📈 Generando visualizaciones...")
    detector.visualize_anomalies(results, save_path='visualizacion_anomalias.png')
    
    print("\n" + "=" * 60)
    print("✅ PROCESO COMPLETADO")
    print("=" * 60)


def create_sample_data():
    """
    Crea datos de ejemplo basados en la estructura proporcionada
    """
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    data = {
        'Fecha': [d.strftime('%d.%m.%Y') for d in dates],
        'Generación total': np.random.normal(20000, 8000, 365),
        'Consumo total': np.random.normal(50000, 15000, 365),
    }
    
    # Asegurar valores positivos
    data['Generación total'] = np.abs(data['Generación total'])
    data['Consumo total'] = np.abs(data['Consumo total'])
    
    # Calcular otras métricas de manera realista
    data['Autoconsumo'] = data['Generación total'] * np.random.uniform(0.95, 0.99, 365)
    data['Energía suministrada a la red'] = data['Generación total'] - data['Autoconsumo']
    data['Energía obtenida de la red'] = np.maximum(
        0, 
        data['Consumo total'] - data['Autoconsumo']
    )
    
    # Agregar algunas anomalías intencionales
    anomaly_indices = np.random.choice(365, size=30, replace=False)
    for idx in anomaly_indices:
        # Anomalía: consumo muy alto
        if np.random.random() > 0.5:
            data['Consumo total'][idx] *= np.random.uniform(2, 4)
        # Anomalía: generación muy baja
        else:
            data['Generación total'][idx] *= np.random.uniform(0.1, 0.3)
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    main()

