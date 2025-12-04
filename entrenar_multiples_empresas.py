"""
Sistema para Entrenar Modelo con Datos de Múltiples Empresas
Maneja diferentes escalas y normaliza adecuadamente
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from anomaly_detection import AnomalyDetector
from mejorar_modelo import ModelTrainer
import os
from glob import glob
import json

class MultiCompanyTrainer:
    """
    Sistema para entrenar modelos con datos de múltiples empresas
    """
    
    def __init__(self, normalization_method='global'):
        """
        Inicializa el entrenador multi-empresa
        
        Parameters:
        -----------
        normalization_method : str
            'global': Normaliza todos los datos juntos
            'per_company': Normaliza por empresa (mantiene escalas relativas)
            'robust': Usa RobustScaler (más resistente a outliers)
        """
        self.normalization_method = normalization_method
        self.companies_info = {}
        self.scaler = None
        self.company_scalers = {}
        self.trainer = ModelTrainer()
    
    def load_company_data(self, filepath, company_name=None, skip_header=True):
        """
        Carga datos de una empresa
        
        Parameters:
        -----------
        filepath : str
            Ruta al archivo CSV
        company_name : str, optional
            Nombre de la empresa (si no se infiere del nombre del archivo)
        skip_header : bool
            Si saltar fila de unidades
            
        Returns:
        --------
        pandas.DataFrame : Datos de la empresa con columna 'Empresa'
        """
        try:
            # Inferir nombre de empresa del archivo si no se proporciona
            if company_name is None:
                company_name = os.path.splitext(os.path.basename(filepath))[0]
                # Limpiar nombre (remover prefijos comunes)
                company_name = company_name.replace('datos_energia_', '').replace('datos_', '')
            
            # Cargar datos
            skiprows = [1] if skip_header and self._has_header_row(filepath) else None
            df = pd.read_csv(filepath, encoding='utf-8', skiprows=skiprows)
            df.columns = df.columns.str.strip()
            
            # Convertir numéricas
            numeric_columns = [
                'Generación total', 'Consumo total', 'Autoconsumo',
                'Energía suministrada a la red', 'Energía obtenida de la red'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Agregar columna de empresa
            df['Empresa'] = company_name
            
            # Guardar info de la empresa
            self.companies_info[company_name] = {
                'file': os.path.basename(filepath),
                'records': len(df),
                'mean_generation': df['Generación total'].mean() if 'Generación total' in df.columns else 0,
                'mean_consumption': df['Consumo total'].mean() if 'Consumo total' in df.columns else 0,
                'std_generation': df['Generación total'].std() if 'Generación total' in df.columns else 0,
                'std_consumption': df['Consumo total'].std() if 'Consumo total' in df.columns else 0
            }
            
            print(f"✅ {company_name}: {len(df)} registros cargados")
            print(f"   Generación promedio: {self.companies_info[company_name]['mean_generation']:.2f} Wh")
            print(f"   Consumo promedio: {self.companies_info[company_name]['mean_consumption']:.2f} Wh")
            
            return df
            
        except Exception as e:
            print(f"❌ Error cargando {filepath}: {e}")
            return None
    
    def _has_header_row(self, filepath):
        """Verifica si tiene fila de unidades"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
                return '[' in second_line and ']' in second_line
        except:
            return False
    
    def combine_companies(self, company_dataframes):
        """
        Combina datos de múltiples empresas
        
        Parameters:
        -----------
        company_dataframes : list
            Lista de DataFrames, cada uno con columna 'Empresa'
            
        Returns:
        --------
        pandas.DataFrame : Datos combinados
        """
        if not company_dataframes:
            return None
        
        print(f"\n🔄 Combinando datos de {len(company_dataframes)} empresas...")
        
        # Combinar
        combined_df = pd.concat(company_dataframes, ignore_index=True)
        
        print(f"✅ Total registros combinados: {len(combined_df)}")
        print(f"\n📊 Distribución por empresa:")
        company_counts = combined_df['Empresa'].value_counts()
        for company, count in company_counts.items():
            print(f"   {company}: {count} registros ({count/len(combined_df)*100:.1f}%)")
        
        return combined_df
    
    def normalize_data(self, df, method=None):
        """
        Normaliza datos según el método seleccionado
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Datos a normalizar
        method : str, optional
            Método de normalización (usa self.normalization_method si es None)
            
        Returns:
        --------
        pandas.DataFrame : Datos normalizados
        """
        method = method or self.normalization_method
        
        print(f"\n⚙️ Normalizando datos (método: {method})...")
        
        numeric_columns = [
            'Generación total', 'Consumo total', 'Autoconsumo',
            'Energía suministrada a la red', 'Energía obtenida de la red'
        ]
        
        df_normalized = df.copy()
        
        if method == 'global':
            # Normalización global: todos los datos juntos
            print("   Normalizando todos los datos juntos (escala global)...")
            self.scaler = StandardScaler()
            
            for col in numeric_columns:
                if col in df.columns:
                    values = df[col].values.reshape(-1, 1)
                    df_normalized[col] = self.scaler.fit_transform(values).flatten()
            
        elif method == 'per_company':
            # Normalización por empresa: mantiene escalas relativas
            print("   Normalizando por empresa (mantiene escalas relativas)...")
            
            for company in df['Empresa'].unique():
                company_mask = df['Empresa'] == company
                company_data = df[company_mask]
                
                if company not in self.company_scalers:
                    self.company_scalers[company] = StandardScaler()
                
                for col in numeric_columns:
                    if col in df.columns:
                        values = company_data[col].values.reshape(-1, 1)
                        normalized = self.company_scalers[company].fit_transform(values).flatten()
                        df_normalized.loc[company_mask, col] = normalized
                
                print(f"   ✅ {company}: normalizado")
        
        elif method == 'robust':
            # RobustScaler: más resistente a outliers
            print("   Usando RobustScaler (resistente a outliers)...")
            self.scaler = RobustScaler()
            
            for col in numeric_columns:
                if col in df.columns:
                    values = df[col].values.reshape(-1, 1)
                    df_normalized[col] = self.scaler.fit_transform(values).flatten()
        
        else:
            print(f"⚠️ Método desconocido: {method}, usando global")
            return self.normalize_data(df, method='global')
        
        print("✅ Normalización completada")
        
        return df_normalized
    
    def prepare_features_multi_company(self, df):
        """
        Prepara características incluyendo información de empresa
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Datos con columna 'Empresa'
            
        Returns:
        --------
        pandas.DataFrame : Características preparadas
        """
        # Usar el método de prepare_features pero adaptado
        from anomaly_detection import AnomalyDetector
        temp_detector = AnomalyDetector()
        
        # Preparar características base
        X_base, _ = temp_detector.prepare_features(df)
        
        # Agregar características de empresa (encoding)
        if 'Empresa' in df.columns:
            # One-hot encoding de empresa
            company_dummies = pd.get_dummies(df['Empresa'], prefix='Empresa')
            X_base = pd.concat([X_base, company_dummies], axis=1)
        
        return X_base, df
    
    def train_multi_company(self, company_files, contamination=0.1, 
                           normalization_method=None, save_model='modelo_multi_empresa.pkl'):
        """
        Entrena modelo con datos de múltiples empresas
        
        Parameters:
        -----------
        company_files : list
            Lista de rutas a archivos CSV de diferentes empresas
        contamination : float
            Valor de contamination
        normalization_method : str, optional
            Método de normalización
        save_model : str
            Nombre del modelo a guardar
            
        Returns:
        --------
        AnomalyDetector : Modelo entrenado
        """
        print("=" * 60)
        print("🏢 ENTRENAMIENTO CON MÚLTIPLES EMPRESAS")
        print("=" * 60)
        
        if normalization_method:
            self.normalization_method = normalization_method
        
        # 1. Cargar datos de todas las empresas
        print(f"\n📂 Cargando datos de {len(company_files)} empresas...")
        company_dfs = []
        
        for filepath in company_files:
            df = self.load_company_data(filepath)
            if df is not None:
                company_dfs.append(df)
        
        if not company_dfs:
            print("❌ No se pudieron cargar datos de empresas")
            return None
        
        # 2. Combinar
        combined_df = self.combine_companies(company_dfs)
        
        # 3. Normalizar
        normalized_df = self.normalize_data(combined_df)
        
        # 4. Preparar características
        print(f"\n🔧 Preparando características...")
        X, df_processed = self.prepare_features_multi_company(normalized_df)
        
        print(f"✅ Características preparadas: {X.shape[1]} características")
        print(f"   Incluye: {len([c for c in X.columns if c.startswith('Empresa_')])} empresas")
        
        # 5. Entrenar modelo
        print(f"\n🎯 Entrenando Isolation Forest...")
        print(f"   Contamination: {contamination}")
        print(f"   Registros: {len(X)}")
        print(f"   Características: {X.shape[1]}")
        
        # Crear detector personalizado
        detector = MultiCompanyAnomalyDetector(
            contamination=contamination,
            normalization_method=self.normalization_method,
            scaler=self.scaler,
            company_scalers=self.company_scalers,
            companies_info=self.companies_info
        )
        
        # Normalizar características finales (todas las características preparadas)
        feature_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X)
        
        # Guardar el scaler de características (no el de columnas individuales)
        detector.feature_scaler = feature_scaler
        
        # Entrenar
        detector.model.fit(X_scaled)
        detector.is_fitted = True
        detector.feature_names = X.columns.tolist()
        detector.companies = list(self.companies_info.keys())
        
        # Guardar
        self.trainer.save_model(detector, save_model)
        
        # Guardar info de empresas
        info_file = save_model.replace('.pkl', '_empresas.json')
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(self.companies_info, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ Modelo entrenado y guardado:")
        print(f"   📁 Modelo: {save_model}")
        print(f"   📁 Info empresas: {info_file}")
        print(f"   🏢 Empresas: {len(self.companies_info)}")
        
        return detector
    
    def compare_normalization_methods(self, company_files, contamination=0.1):
        """
        Compara diferentes métodos de normalización
        
        Parameters:
        -----------
        company_files : list
            Lista de archivos de empresas
        contamination : float
            Valor de contamination
        """
        print("=" * 60)
        print("⚖️ COMPARACIÓN DE MÉTODOS DE NORMALIZACIÓN")
        print("=" * 60)
        
        methods = ['global', 'per_company', 'robust']
        results = {}
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"🔍 Probando método: {method}")
            print(f"{'='*60}")
            
            trainer = MultiCompanyTrainer(normalization_method=method)
            detector = trainer.train_multi_company(
                company_files, 
                contamination=contamination,
                save_model=f'modelo_test_{method}.pkl'
            )
            
            if detector:
                # Evaluar (cargar datos y predecir)
                company_dfs = []
                for filepath in company_files:
                    df = trainer.load_company_data(filepath)
                    if df is not None:
                        company_dfs.append(df)
                
                if company_dfs:
                    combined = trainer.combine_companies(company_dfs)
                    normalized = trainer.normalize_data(combined)
                    results_df = detector.get_anomalies(normalized)
                    
                    num_anomalies = results_df['Es_Anomalia'].sum()
                    results[method] = {
                        'anomalies': num_anomalies,
                        'percentage': num_anomalies / len(results_df) * 100
                    }
        
        # Mostrar comparación
        print(f"\n{'='*60}")
        print("📊 RESULTADOS DE COMPARACIÓN")
        print(f"{'='*60}")
        
        for method, result in results.items():
            print(f"\n{method.upper()}:")
            print(f"   Anomalías: {result['anomalies']}")
            print(f"   Porcentaje: {result['percentage']:.2f}%")
        
        # Recomendación
        print(f"\n💡 RECOMENDACIÓN:")
        # El método con separación más clara sería mejor, pero necesitaríamos evaluar
        print("   - 'global': Mejor si las empresas tienen escalas similares")
        print("   - 'per_company': Mejor si las empresas tienen escalas muy diferentes")
        print("   - 'robust': Mejor si hay muchos outliers")
        
        return results


class MultiCompanyAnomalyDetector(AnomalyDetector):
    """
    Detector de anomalías adaptado para múltiples empresas
    """
    
    def __init__(self, contamination=0.1, normalization_method='global',
                 scaler=None, company_scalers=None, companies_info=None):
        super().__init__(contamination=contamination)
        self.normalization_method = normalization_method
        self.scaler = scaler  # Scaler para columnas numéricas originales
        self.company_scalers = company_scalers or {}
        self.companies_info = companies_info or {}
        self.companies = []
        self.feature_scaler = None  # Scaler para características finales preparadas
    
    def prepare_features(self, df):
        """Prepara características para múltiples empresas"""
        from anomaly_detection import AnomalyDetector
        temp_detector = AnomalyDetector()
        
        # Preparar características base
        X_base, df_processed = temp_detector.prepare_features(df)
        
        # Agregar encoding de empresa si existe
        if 'Empresa' in df.columns:
            company_dummies = pd.get_dummies(df['Empresa'], prefix='Empresa')
            # Asegurar que todas las empresas del entrenamiento estén presentes
            for company in self.companies:
                col_name = f'Empresa_{company}'
                if col_name not in company_dummies.columns:
                    company_dummies[col_name] = 0
            
            # Mantener solo empresas conocidas
            known_company_cols = [c for c in company_dummies.columns 
                                if any(c.endswith(f'_{comp}') for comp in self.companies)]
            company_dummies = company_dummies[known_company_cols]
            
            X_base = pd.concat([X_base, company_dummies], axis=1)
        
        return X_base, df_processed
    
    def predict(self, df):
        """Predice con normalización adecuada"""
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado")
        
        # Normalizar datos según método
        df_normalized = df.copy()
        
        numeric_columns = [
            'Generación total', 'Consumo total', 'Autoconsumo',
            'Energía suministrada a la red', 'Energía obtenida de la red'
        ]
        
        if self.normalization_method == 'global' and self.scaler:
            for col in numeric_columns:
                if col in df.columns:
                    values = df[col].values.reshape(-1, 1)
                    df_normalized[col] = self.scaler.transform(values).flatten()
        
        elif self.normalization_method == 'per_company':
            for company in df['Empresa'].unique():
                company_mask = df['Empresa'] == company
                if company in self.company_scalers:
                    for col in numeric_columns:
                        if col in df.columns:
                            values = df.loc[company_mask, col].values.reshape(-1, 1)
                            df_normalized.loc[company_mask, col] = \
                                self.company_scalers[company].transform(values).flatten()
        
        # Preparar características y predecir
        X, _ = self.prepare_features(df_normalized)
        X = X[self.feature_names]
        
        # Usar el scaler de características que se usó en entrenamiento
        if self.feature_scaler is not None:
            X_scaled = self.feature_scaler.transform(X)
        else:
            # Fallback: crear nuevo scaler si no existe
            X_scaled = StandardScaler().fit_transform(X)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, df):
        """Obtiene scores de anomalía con normalización adecuada"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Normalizar datos según método
        df_normalized = df.copy()
        
        numeric_columns = [
            'Generación total', 'Consumo total', 'Autoconsumo',
            'Energía suministrada a la red', 'Energía obtenida de la red'
        ]
        
        if self.normalization_method == 'global' and self.scaler:
            for col in numeric_columns:
                if col in df.columns:
                    values = df[col].values.reshape(-1, 1)
                    df_normalized[col] = self.scaler.transform(values).flatten()
        
        elif self.normalization_method == 'per_company':
            for company in df['Empresa'].unique():
                company_mask = df['Empresa'] == company
                if company in self.company_scalers:
                    for col in numeric_columns:
                        if col in df.columns:
                            values = df.loc[company_mask, col].values.reshape(-1, 1)
                            df_normalized.loc[company_mask, col] = \
                                self.company_scalers[company].transform(values).flatten()
        
        # Preparar características
        X, _ = self.prepare_features(df_normalized)
        X = X[self.feature_names]
        
        # Usar el scaler de características que se usó en entrenamiento
        if self.feature_scaler is not None:
            X_scaled = self.feature_scaler.transform(X)
        else:
            # Fallback: crear nuevo scaler si no existe
            X_scaled = StandardScaler().fit_transform(X)
        
        # Obtener scores
        scores = self.model.score_samples(X_scaled)
        
        return scores


def main():
    """Función principal"""
    print("=" * 60)
    print("🏢 ENTRENAMIENTO CON MÚLTIPLES EMPRESAS")
    print("=" * 60)
    
    # Buscar archivos de empresas
    print("\n🔍 Buscando archivos de empresas...")
    company_files = glob('datos_energia*.csv') + glob('empresa_*.csv') + glob('*_energia.csv')
    
    if not company_files:
        print("⚠️ No se encontraron archivos automáticamente")
        print("💡 Proporciona manualmente las rutas:")
        company_files = []
        while True:
            filepath = input("Ruta al archivo (Enter para terminar): ").strip()
            if not filepath:
                break
            if os.path.exists(filepath):
                company_files.append(filepath)
            else:
                print(f"⚠️ Archivo no encontrado: {filepath}")
    else:
        print(f"✅ Encontrados {len(company_files)} archivos:")
        for f in company_files:
            print(f"   - {os.path.basename(f)}")
    
    if not company_files:
        print("❌ No hay archivos para procesar")
        return
    
    # Menú
    print("\n" + "=" * 60)
    print("¿Qué deseas hacer?")
    print("1. Entrenar con normalización global (recomendado)")
    print("2. Entrenar con normalización por empresa")
    print("3. Entrenar con RobustScaler")
    print("4. Comparar todos los métodos")
    
    opcion = input("\nSelecciona (1-4): ").strip()
    contamination = float(input("Contamination (default 0.05): ").strip() or "0.05")
    
    trainer = MultiCompanyTrainer()
    
    if opcion == '1':
        detector = trainer.train_multi_company(
            company_files, 
            contamination=contamination,
            normalization_method='global',
            save_model='modelo_multi_empresa_global.pkl'
        )
    elif opcion == '2':
        detector = trainer.train_multi_company(
            company_files,
            contamination=contamination,
            normalization_method='per_company',
            save_model='modelo_multi_empresa_por_empresa.pkl'
        )
    elif opcion == '3':
        detector = trainer.train_multi_company(
            company_files,
            contamination=contamination,
            normalization_method='robust',
            save_model='modelo_multi_empresa_robust.pkl'
        )
    elif opcion == '4':
        trainer.compare_normalization_methods(company_files, contamination)
    else:
        print("❌ Opción no válida")


if __name__ == "__main__":
    main()

