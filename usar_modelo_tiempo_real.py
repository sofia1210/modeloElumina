"""
Sistema para Usar Modelo Multi-Empresa en Tiempo Real
Carga el modelo guardado y predice anomalías en datos nuevos
"""

import pandas as pd
import numpy as np
from mejorar_modelo import ModelTrainer
from entrenar_multiples_empresas import MultiCompanyAnomalyDetector
import os
from datetime import datetime
import json

class RealTimeMultiCompanyPredictor:
    """
    Sistema para usar modelo multi-empresa en tiempo real
    """
    
    def __init__(self, model_path='modelo_multi_empresa.pkl'):
        """
        Inicializa el predictor cargando el modelo
        
        Parameters:
        -----------
        model_path : str
            Ruta al modelo PKL guardado
        """
        self.model_path = model_path
        self.detector = None
        self.model_loaded = False
        self.load_time = None
        
    def load_model(self):
        """
        Carga el modelo una vez (esto toma ~1-2 segundos)
        Después, las predicciones son instantáneas
        """
        if self.model_loaded:
            return
        
        print("=" * 60)
        print("🔄 CARGANDO MODELO MULTI-EMPRESA")
        print("=" * 60)
        
        import time
        start_time = time.time()
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Modelo no encontrado: {self.model_path}")
        
        trainer = ModelTrainer()
        model_data = trainer.load_model(self.model_path)
        
        # Verificar si es MultiCompanyAnomalyDetector
        if hasattr(model_data, 'normalization_method'):
            self.detector = model_data
            print(f"✅ Modelo Multi-Empresa cargado")
        else:
            # Convertir a MultiCompanyAnomalyDetector si es necesario
            self.detector = MultiCompanyAnomalyDetector(
                contamination=model_data.contamination,
                normalization_method=getattr(model_data, 'normalization_method', 'global'),
                scaler=getattr(model_data, 'scaler', None),
                company_scalers=getattr(model_data, 'company_scalers', {}),
                companies_info=getattr(model_data, 'companies_info', {})
            )
            self.detector.model = model_data.model
            self.detector.scaler = model_data.scaler
            self.detector.feature_scaler = getattr(model_data, 'feature_scaler', None)
            self.detector.feature_names = model_data.feature_names
            self.detector.is_fitted = True
            self.detector.companies = getattr(model_data, 'companies', [])
            print(f"✅ Modelo convertido a Multi-Empresa")
        
        self.load_time = time.time() - start_time
        self.model_loaded = True
        
        print(f"✅ Modelo cargado en {self.load_time:.3f} segundos")
        print(f"📊 Características: {len(self.detector.feature_names)}")
        print(f"🏢 Empresas en modelo: {len(self.detector.companies) if hasattr(self.detector, 'companies') else 'N/A'}")
        print("⚡ Listo para predicciones en tiempo real")
    
    def predict_single(self, registro, company_name=None):
        """
        Predice anomalía en un SOLO registro nuevo (muy rápido: < 10ms)
        
        Parameters:
        -----------
        registro : dict o pandas.Series
            Un solo registro con las columnas necesarias
        company_name : str, optional
            Nombre de la empresa (si no está en el registro)
            
        Returns:
        --------
        dict : Resultado de la predicción
        """
        if not self.model_loaded:
            self.load_model()
        
        # Convertir a DataFrame
        if isinstance(registro, dict):
            df = pd.DataFrame([registro])
        else:
            df = pd.DataFrame([registro])
        
        # Agregar nombre de empresa si no existe
        if 'Empresa' not in df.columns and company_name:
            df['Empresa'] = company_name
        
        # Predecir
        import time
        start_time = time.time()
        results = self.detector.get_anomalies(df)
        prediction_time = time.time() - start_time
        
        resultado = results.iloc[0].to_dict()
        resultado['tiempo_prediccion_ms'] = prediction_time * 1000
        
        return resultado
    
    def predict_batch(self, datos, mostrar_tiempo=True):
        """
        Predice anomalías en múltiples registros (batch)
        
        Parameters:
        -----------
        datos : pandas.DataFrame o list of dicts
            Múltiples registros
        mostrar_tiempo : bool
            Si mostrar tiempo de procesamiento
            
        Returns:
        --------
        pandas.DataFrame : Resultados de predicción
        """
        if not self.model_loaded:
            self.load_model()
        
        # Convertir a DataFrame si es necesario
        if isinstance(datos, list):
            df = pd.DataFrame(datos)
        else:
            df = datos.copy()
        
        # Predecir
        import time
        start_time = time.time()
        results = self.detector.get_anomalies(df)
        prediction_time = time.time() - start_time
        
        if mostrar_tiempo:
            print(f"⚡ {len(df)} registros procesados en {prediction_time:.4f} segundos")
            print(f"📊 Velocidad: {len(df)/prediction_time:.1f} registros/segundo")
        
        return results
    
    def predict_from_csv(self, csv_path, output_path=None, company_name=None):
        """
        Predice anomalías desde un archivo CSV
        
        Parameters:
        -----------
        csv_path : str
            Ruta al archivo CSV con datos nuevos
        output_path : str, optional
            Ruta donde guardar resultados (si None, no guarda)
        company_name : str, optional
            Nombre de la empresa (si no está en el CSV)
            
        Returns:
        --------
        pandas.DataFrame : Resultados de predicción
        """
        if not self.model_loaded:
            self.load_model()
        
        print(f"\n📂 Cargando datos desde: {csv_path}")
        
        # Cargar CSV (intentar saltar fila de unidades)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', skiprows=[1])
            df.columns = df.columns.str.strip()
        except:
            df = pd.read_csv(csv_path, encoding='utf-8')
            df.columns = df.columns.str.strip()
        
        # Convertir numéricas
        numeric_columns = [
            'Generación total', 'Consumo total', 'Autoconsumo',
            'Energía suministrada a la red', 'Energía obtenida de la red'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Agregar empresa si no existe
        if 'Empresa' not in df.columns and company_name:
            df['Empresa'] = company_name
        
        print(f"✅ {len(df)} registros cargados")
        
        # Predecir
        print(f"🔍 Detectando anomalías...")
        results = self.predict_batch(df)
        
        # Guardar si se especifica
        if output_path:
            results.to_csv(output_path, index=False, encoding='utf-8')
            print(f"💾 Resultados guardados en: {output_path}")
        
        # Resumen
        num_anomalies = results['Es_Anomalia'].sum()
        print(f"\n📊 RESUMEN:")
        print(f"   Total registros: {len(results)}")
        print(f"   Anomalías detectadas: {num_anomalies} ({num_anomalies/len(results)*100:.2f}%)")
        
        if num_anomalies > 0:
            print(f"\n🚨 ANOMALÍAS DETECTADAS:")
            anomalies = results[results['Es_Anomalia'] == True].sort_values('Score_Anomalia')
            fecha_col = 'Fecha' if 'Fecha' in anomalies.columns else ('Fecha y hora' if 'Fecha y hora' in anomalies.columns else anomalies.columns[0])
            display_cols = [fecha_col, 'Score_Anomalia']
            if 'Empresa' in anomalies.columns:
                display_cols.insert(1, 'Empresa')
            print(anomalies[display_cols].head(10).to_string(index=False))
        
        return results


def ejemplo_uso_individual():
    """Ejemplo: Predicción de un registro individual"""
    print("=" * 60)
    print("⚡ EJEMPLO: PREDICCIÓN INDIVIDUAL")
    print("=" * 60)
    
    predictor = RealTimeMultiCompanyPredictor('modelo_multi_empresa.pkl')
    predictor.load_model()
    
    # Nuevo registro que llega
    nuevo_registro = {
        'Fecha y hora': '15.01.2025',
        'Generación total': 25000.0,
        'Consumo total': 60000.0,
        'Autoconsumo': 24000.0,
        'Energía suministrada a la red': 1000.0,
        'Energía obtenida de la red': 36000.0,
        'Empresa': 'NEUROCIENCIAS'  # O cualquier empresa del entrenamiento
    }
    
    print(f"\n📥 Nuevo registro recibido...")
    resultado = predictor.predict_single(nuevo_registro)
    
    print(f"\n📊 RESULTADO:")
    print(f"   Es Anomalía: {'🚨 SÍ' if resultado['Es_Anomalia'] else '✅ NO'}")
    print(f"   Score: {resultado['Score_Anomalia']:.4f}")
    print(f"   Tiempo: {resultado['tiempo_prediccion_ms']:.2f} milisegundos")
    
    return resultado


def ejemplo_uso_csv():
    """Ejemplo: Predicción desde archivo CSV"""
    print("\n" + "=" * 60)
    print("⚡ EJEMPLO: PREDICCIÓN DESDE CSV")
    print("=" * 60)
    
    predictor = RealTimeMultiCompanyPredictor('modelo_multi_empresa.pkl')
    predictor.load_model()
    
    # Archivo CSV con datos nuevos
    csv_path = input("Ruta al CSV con datos nuevos (o Enter para usar ejemplo): ").strip()
    
    if not csv_path:
        print("💡 Proporciona la ruta a un archivo CSV con datos nuevos")
        return
    
    if not os.path.exists(csv_path):
        print(f"❌ Archivo no encontrado: {csv_path}")
        return
    
    # Nombre de empresa (opcional)
    company_name = input("Nombre de la empresa (opcional, Enter para omitir): ").strip() or None
    
    # Predecir
    results = predictor.predict_from_csv(
        csv_path,
        output_path='resultados_tiempo_real.csv',
        company_name=company_name
    )
    
    return results


def ejemplo_api_style():
    """Ejemplo: Uso estilo API para integración"""
    print("\n" + "=" * 60)
    print("⚡ EJEMPLO: USO ESTILO API")
    print("=" * 60)
    
    # Inicializar predictor (cargar modelo una vez)
    predictor = RealTimeMultiCompanyPredictor('modelo_multi_empresa.pkl')
    predictor.load_model()
    
    print("\n✅ Servicio de predicción listo")
    print("   (En producción, esto se mantendría corriendo)")
    
    # Simular múltiples requests
    print("\n📥 Simulando requests de predicción...")
    
    requests = [
        {
            'Generación total': 20000, 'Consumo total': 50000, 'Autoconsumo': 19000,
            'Energía suministrada a la red': 1000, 'Energía obtenida de la red': 31000,
            'Empresa': 'NEUROCIENCIAS'
        },
        {
            'Generación total': 5000, 'Consumo total': 60000, 'Autoconsumo': 4800,
            'Energía suministrada a la red': 200, 'Energía obtenida de la red': 55200,
            'Empresa': 'BIOPETROL_BEREA'
        },
        {
            'Generación total': 35000, 'Consumo total': 40000, 'Autoconsumo': 34000,
            'Energía suministrada a la red': 1000, 'Energía obtenida de la red': 6000,
            'Empresa': 'SHOPPING_BOLIVAR_214'
        },
    ]
    
    for i, req in enumerate(requests, 1):
        resultado = predictor.predict_single(req)
        status = "🚨 ANOMALÍA" if resultado['Es_Anomalia'] else "✅ Normal"
        print(f"Request {i}: {status} (Score: {resultado['Score_Anomalia']:.4f}, "
              f"Tiempo: {resultado['tiempo_prediccion_ms']:.2f}ms)")


def main():
    """Menú principal"""
    print("=" * 60)
    print("⚡ SISTEMA DE PREDICCIÓN EN TIEMPO REAL - MULTI-EMPRESA")
    print("=" * 60)
    
    # Verificar que existe el modelo
    model_path = 'modelo_multi_empresa.pkl'
    if not os.path.exists(model_path):
        print(f"\n❌ Modelo no encontrado: {model_path}")
        print("💡 Primero entrena el modelo ejecutando: python multientrenamiento.py")
        return
    
    print("\n¿Qué ejemplo deseas ver?")
    print("1. Predicción rápida de un registro")
    print("2. Predicción desde archivo CSV")
    print("3. Uso estilo API")
    print("4. Todos los ejemplos")
    
    opcion = input("\nSelecciona (1-4): ").strip()
    
    if opcion == '1':
        ejemplo_uso_individual()
    elif opcion == '2':
        ejemplo_uso_csv()
    elif opcion == '3':
        ejemplo_api_style()
    elif opcion == '4':
        ejemplo_uso_individual()
        ejemplo_api_style()
        print("\n💡 Para usar con CSV, ejecuta opción 2")
    else:
        print("❌ Opción no válida")


if __name__ == "__main__":
    main()

