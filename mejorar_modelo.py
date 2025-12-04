"""
Sistema para Mejorar y Optimizar el Modelo de Detección de Anomalías
Incluye: guardar/cargar modelos, re-entrenamiento, validación y ajuste de parámetros
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import joblib
import os
from datetime import datetime
import json
from anomaly_detection import AnomalyDetector
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Clase para entrenar, mejorar y optimizar el modelo de detección de anomalías
    """
    
    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.training_history = []
        
    def save_model(self, detector, filepath='modelo_entrenado.pkl'):
        """
        Guarda el modelo entrenado para uso futuro
        
        Parameters:
        -----------
        detector : AnomalyDetector
            Detector entrenado
        filepath : str
            Ruta donde guardar el modelo
        """
        if not detector.is_fitted:
            raise ValueError("El modelo debe estar entrenado antes de guardarlo")
        
        model_data = {
            'model': detector.model,
            'scaler': detector.scaler,
            'feature_names': detector.feature_names,
            'contamination': detector.contamination,
            'random_state': detector.random_state,
            'trained_date': datetime.now().isoformat()
        }
        
        # Guardar feature_scaler si existe (para MultiCompanyAnomalyDetector)
        if hasattr(detector, 'feature_scaler') and detector.feature_scaler is not None:
            model_data['feature_scaler'] = detector.feature_scaler
        
        # Guardar atributos adicionales de MultiCompanyAnomalyDetector
        if hasattr(detector, 'normalization_method'):
            model_data['normalization_method'] = detector.normalization_method
        if hasattr(detector, 'company_scalers'):
            model_data['company_scalers'] = detector.company_scalers
        if hasattr(detector, 'companies_info'):
            model_data['companies_info'] = detector.companies_info
        if hasattr(detector, 'companies'):
            model_data['companies'] = detector.companies
        
        joblib.dump(model_data, filepath)
        print(f"✅ Modelo guardado en: {filepath}")
        
        # Guardar metadatos
        metadata = {
            'trained_date': model_data['trained_date'],
            'contamination': detector.contamination,
            'num_features': len(detector.feature_names),
            'features': detector.feature_names
        }
        
        with open(filepath.replace('.pkl', '_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def load_model(self, filepath='modelo_entrenado.pkl'):
        """
        Carga un modelo previamente entrenado
        
        Parameters:
        -----------
        filepath : str
            Ruta del modelo guardado
            
        Returns:
        --------
        AnomalyDetector : Detector cargado y listo para usar
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
        
        model_data = joblib.load(filepath)
        
        detector = AnomalyDetector(
            contamination=model_data['contamination'],
            random_state=model_data['random_state']
        )
        
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.feature_names = model_data['feature_names']
        detector.is_fitted = True
        
        # Cargar feature_scaler si existe
        if 'feature_scaler' in model_data:
            detector.feature_scaler = model_data['feature_scaler']
        
        # Cargar atributos adicionales de MultiCompanyAnomalyDetector
        if 'normalization_method' in model_data:
            detector.normalization_method = model_data['normalization_method']
        if 'company_scalers' in model_data:
            detector.company_scalers = model_data['company_scalers']
        if 'companies_info' in model_data:
            detector.companies_info = model_data['companies_info']
        if 'companies' in model_data:
            detector.companies = model_data['companies']
        
        print(f"✅ Modelo cargado desde: {filepath}")
        print(f"📅 Fecha de entrenamiento: {model_data.get('trained_date', 'Desconocida')}")
        
        return detector
    
    def retrain_with_new_data(self, old_model_path, new_data_df, contamination=None):
        """
        Re-entrena el modelo combinando datos antiguos y nuevos
        
        Parameters:
        -----------
        old_model_path : str
            Ruta del modelo anterior
        new_data_df : pandas.DataFrame
            Nuevos datos para agregar al entrenamiento
        contamination : float, optional
            Nuevo valor de contamination (usa el del modelo anterior si es None)
            
        Returns:
        --------
        AnomalyDetector : Nuevo modelo re-entrenado
        """
        print("🔄 Re-entrenando modelo con nuevos datos...")
        
        # Cargar modelo anterior
        old_model_data = joblib.load(old_model_path)
        old_contamination = contamination if contamination else old_model_data['contamination']
        
        # Cargar datos históricos si existen
        historical_data_path = 'datos_energia_historico.csv'
        if os.path.exists(historical_data_path):
            print("📂 Cargando datos históricos...")
            historical_df = pd.read_csv(historical_data_path, encoding='utf-8', skiprows=[1])
            historical_df.columns = historical_df.columns.str.strip()
            
            # Convertir columnas numéricas
            numeric_columns = [
                'Generación total', 'Consumo total', 'Autoconsumo',
                'Energía suministrada a la red', 'Energía obtenida de la red'
            ]
            for col in numeric_columns:
                if col in historical_df.columns:
                    historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce')
            
            # Combinar datos históricos con nuevos
            combined_df = pd.concat([historical_df, new_data_df], ignore_index=True)
            print(f"✅ Datos combinados: {len(historical_df)} históricos + {len(new_data_df)} nuevos = {len(combined_df)} total")
        else:
            combined_df = new_data_df
            print(f"✅ Usando solo nuevos datos: {len(combined_df)} registros")
        
        # Crear nuevo detector
        new_detector = AnomalyDetector(contamination=old_contamination)
        new_detector.train(combined_df)
        
        # Guardar nuevo modelo
        new_model_path = f"modelo_entrenado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.save_model(new_detector, new_model_path)
        
        # Actualizar datos históricos
        combined_df.to_csv(historical_data_path, index=False, encoding='utf-8')
        print(f"💾 Datos históricos actualizados en: {historical_data_path}")
        
        return new_detector
    
    def optimize_parameters(self, df, contamination_range=[0.05, 0.1, 0.15, 0.2], 
                           n_estimators_range=[50, 100, 200]):
        """
        Optimiza los parámetros del modelo usando búsqueda en grid
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Datos para entrenamiento y validación
        contamination_range : list
            Valores de contamination a probar
        n_estimators_range : list
            Número de árboles a probar
            
        Returns:
        --------
        dict : Mejores parámetros encontrados
        """
        print("🔍 Optimizando parámetros del modelo...")
        print("=" * 60)
        
        # Preparar datos
        detector_temp = AnomalyDetector()
        X, _ = detector_temp.prepare_features(df)
        X_scaled = StandardScaler().fit_transform(X)
        
        # Dividir en entrenamiento y validación (80/20)
        split_idx = int(len(X_scaled) * 0.8)
        X_train = X_scaled[:split_idx]
        X_val = X_scaled[split_idx:]
        
        best_score = -np.inf
        best_params = None
        results = []
        
        # Búsqueda en grid
        param_grid = {
            'contamination': contamination_range,
            'n_estimators': n_estimators_range
        }
        
        total_combinations = len(contamination_range) * len(n_estimators_range)
        current = 0
        
        for params in ParameterGrid(param_grid):
            current += 1
            print(f"\n[{current}/{total_combinations}] Probando: {params}")
            
            # Entrenar modelo
            model = IsolationForest(
                contamination=params['contamination'],
                n_estimators=params['n_estimators'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train)
            
            # Evaluar en conjunto de validación
            scores = model.score_samples(X_val)
            # Usar la media de scores como métrica (más alto = mejor separación)
            mean_score = np.mean(scores)
            
            # Contar anomalías detectadas
            predictions = model.predict(X_val)
            num_anomalies = (predictions == -1).sum()
            anomaly_rate = num_anomalies / len(X_val)
            
            result = {
                'params': params,
                'mean_score': mean_score,
                'anomaly_rate': anomaly_rate,
                'num_anomalies': num_anomalies
            }
            results.append(result)
            
            print(f"  Score promedio: {mean_score:.4f}, Tasa de anomalías: {anomaly_rate:.2%}")
            
            # Actualizar mejor modelo
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
                print(f"  ⭐ Nuevo mejor modelo!")
        
        print("\n" + "=" * 60)
        print("📊 RESULTADOS DE OPTIMIZACIÓN")
        print("=" * 60)
        print(f"\n🏆 Mejores parámetros encontrados:")
        print(f"  Contamination: {best_params['contamination']}")
        print(f"  N_estimators: {best_params['n_estimators']}")
        print(f"  Score: {best_score:.4f}")
        
        # Guardar resultados
        results_df = pd.DataFrame(results)
        results_df.to_csv('optimizacion_parametros.csv', index=False, encoding='utf-8')
        print(f"\n💾 Resultados guardados en: optimizacion_parametros.csv")
        
        self.best_params = best_params
        return best_params
    
    def validate_model(self, detector, df, known_anomalies=None):
        """
        Valida el modelo comparando con anomalías conocidas (si están disponibles)
        
        Parameters:
        -----------
        detector : AnomalyDetector
            Modelo a validar
        df : pandas.DataFrame
            Datos para validación
        known_anomalies : list, optional
            Lista de índices de anomalías conocidas (si tienes datos etiquetados)
            
        Returns:
        --------
        dict : Métricas de validación
        """
        print("📊 Validando modelo...")
        
        results = detector.get_anomalies(df)
        predictions = results['Es_Anomalia'].astype(int)
        
        metrics = {
            'total_samples': len(df),
            'anomalies_detected': predictions.sum(),
            'anomaly_rate': predictions.mean(),
            'mean_anomaly_score': results[results['Es_Anomalia']]['Score_Anomalia'].mean() if predictions.sum() > 0 else 0
        }
        
        # Si hay anomalías conocidas, calcular precisión
        if known_anomalies is not None:
            known_anomalies_set = set(known_anomalies)
            predicted_anomalies_set = set(results[results['Es_Anomalia']].index.tolist())
            
            true_positives = len(known_anomalies_set & predicted_anomalies_set)
            false_positives = len(predicted_anomalies_set - known_anomalies_set)
            false_negatives = len(known_anomalies_set - predicted_anomalies_set)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            })
            
            print(f"\n📈 Métricas de Validación:")
            print(f"  Precisión: {precision:.2%}")
            print(f"  Recall: {recall:.2%}")
            print(f"  F1-Score: {f1_score:.2%}")
        
        print(f"\n📊 Estadísticas Generales:")
        print(f"  Total de muestras: {metrics['total_samples']}")
        print(f"  Anomalías detectadas: {metrics['anomalies_detected']} ({metrics['anomaly_rate']:.2%})")
        print(f"  Score promedio de anomalías: {metrics['mean_anomaly_score']:.4f}")
        
        return metrics
    
    def train_with_validation(self, df, contamination=0.1, validation_split=0.2):
        """
        Entrena el modelo con validación automática
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Datos completos
        contamination : float
            Valor de contamination
        validation_split : float
            Proporción de datos para validación
            
        Returns:
        --------
        tuple : (detector entrenado, métricas de validación)
        """
        print("🎯 Entrenando modelo con validación...")
        
        # Dividir datos
        split_idx = int(len(df) * (1 - validation_split))
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        print(f"📊 División: {len(train_df)} entrenamiento, {len(val_df)} validación")
        
        # Entrenar
        detector = AnomalyDetector(contamination=contamination)
        detector.train(train_df)
        
        # Validar
        metrics = self.validate_model(detector, val_df)
        
        return detector, metrics


def main():
    """
    Función principal para mejorar el modelo
    """
    print("=" * 60)
    print("🚀 SISTEMA DE MEJORA Y OPTIMIZACIÓN DEL MODELO")
    print("=" * 60)
    
    trainer = ModelTrainer()
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    try:
        df = pd.read_csv('datos_energia.csv', encoding='utf-8', skiprows=[1])
        df.columns = df.columns.str.strip()
        
        numeric_columns = [
            'Generación total', 'Consumo total', 'Autoconsumo',
            'Energía suministrada a la red', 'Energía obtenida de la red'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✅ Datos cargados: {len(df)} registros")
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return
    
    # Menú de opciones
    print("\n" + "=" * 60)
    print("¿Qué deseas hacer?")
    print("1. Optimizar parámetros del modelo")
    print("2. Entrenar y guardar modelo")
    print("3. Cargar modelo existente")
    print("4. Re-entrenar con nuevos datos")
    print("5. Validar modelo existente")
    
    choice = input("\nSelecciona una opción (1-5): ").strip()
    
    if choice == '1':
        # Optimizar parámetros
        best_params = trainer.optimize_parameters(df)
        
        # Entrenar con mejores parámetros
        print("\n🎯 Entrenando modelo con mejores parámetros...")
        detector = AnomalyDetector(
            contamination=best_params['contamination'],
            random_state=42
        )
        detector.model.n_estimators = best_params['n_estimators']
        detector.train(df)
        trainer.save_model(detector, 'modelo_optimizado.pkl')
        
    elif choice == '2':
        # Entrenar y guardar
        contamination = float(input("Contamination (0.05-0.5, default 0.1): ") or "0.1")
        detector = AnomalyDetector(contamination=contamination)
        detector.train(df)
        trainer.save_model(detector)
        
    elif choice == '3':
        # Cargar modelo
        model_path = input("Ruta del modelo (default: modelo_entrenado.pkl): ").strip() or "modelo_entrenado.pkl"
        detector = trainer.load_model(model_path)
        
        # Probar modelo
        results = detector.get_anomalies(df.tail(10))
        print("\n📊 Predicciones en últimos 10 registros:")
        print(results[['Fecha' if 'Fecha' in results.columns else results.columns[0], 
                      'Es_Anomalia', 'Score_Anomalia']].to_string(index=False))
        
    elif choice == '4':
        # Re-entrenar
        model_path = input("Ruta del modelo anterior (default: modelo_entrenado.pkl): ").strip() or "modelo_entrenado.pkl"
        new_detector = trainer.retrain_with_new_data(model_path, df)
        
    elif choice == '5':
        # Validar
        model_path = input("Ruta del modelo (default: modelo_entrenado.pkl): ").strip() or "modelo_entrenado.pkl"
        detector = trainer.load_model(model_path)
        metrics = trainer.validate_model(detector, df)
        
    else:
        print("❌ Opción no válida")


if __name__ == "__main__":
    main()


