"""
Sistema Completo de Alertas en Tiempo Real
Usa el modelo multi-empresa entrenado para detectar anomalías y generar alertas
"""

import pandas as pd
import numpy as np
from usar_modelo_tiempo_real import RealTimeMultiCompanyPredictor
import os
import json
from datetime import datetime
import time
from glob import glob

class SistemaAlertasTiempoReal:
    """
    Sistema completo para alertas en tiempo real usando modelo multi-empresa
    """
    
    def __init__(self, model_path='modelo_multi_empresa.pkl'):
        """
        Inicializa el sistema de alertas
        
        Parameters:
        -----------
        model_path : str
            Ruta al modelo entrenado
        """
        self.model_path = model_path
        self.predictor = RealTimeMultiCompanyPredictor(model_path)
        self.alertas_historial = []
        self.alertas_file = 'alertas_historial.json'
        self.running = False
        
    def iniciar(self):
        """Inicia el sistema cargando el modelo"""
        print("=" * 60)
        print("🚀 INICIANDO SISTEMA DE ALERTAS EN TIEMPO REAL")
        print("=" * 60)
        
        if not os.path.exists(self.model_path):
            print(f"❌ Modelo no encontrado: {self.model_path}")
            print("💡 Primero entrena el modelo: python multientrenamiento.py")
            return False
        
        print(f"\n📂 Cargando modelo: {self.model_path}")
        self.predictor.load_model()
        
        # Cargar historial de alertas si existe
        if os.path.exists(self.alertas_file):
            with open(self.alertas_file, 'r', encoding='utf-8') as f:
                self.alertas_historial = json.load(f)
            print(f"✅ Historial cargado: {len(self.alertas_historial)} alertas previas")
        
        print("\n✅ Sistema listo para procesar datos en tiempo real")
        return True
    
    def procesar_dato(self, dato, empresa=None, enviar_alerta=True):
        """
        Procesa un dato nuevo y genera alerta si es anomalía
        
        Parameters:
        -----------
        dato : dict o pandas.Series
            Dato nuevo a procesar
        empresa : str, optional
            Nombre de la empresa (si no está en el dato)
        enviar_alerta : bool
            Si enviar alerta automáticamente
            
        Returns:
        --------
        dict : Resultado con información de la predicción
        """
        # Agregar empresa si no existe
        if isinstance(dato, dict) and 'Empresa' not in dato and empresa:
            dato['Empresa'] = empresa
        
        # Predecir
        resultado = self.predictor.predict_single(dato, company_name=empresa)
        
        # Si es anomalía, generar alerta
        if resultado['Es_Anomalia'] and enviar_alerta:
            alerta = self._crear_alerta(dato, resultado)
            self._guardar_alerta(alerta)
            self._enviar_alerta(alerta)
        
        return resultado
    
    def procesar_csv(self, csv_path, empresa=None, output_path=None):
        """
        Procesa un archivo CSV con datos nuevos
        
        Parameters:
        -----------
        csv_path : str
            Ruta al archivo CSV
        empresa : str, optional
            Nombre de la empresa
        output_path : str, optional
            Ruta para guardar resultados
            
        Returns:
        --------
        pandas.DataFrame : Resultados con alertas
        """
        print(f"\n📂 Procesando archivo: {csv_path}")
        
        # Cargar CSV
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
        if 'Empresa' not in df.columns and empresa:
            df['Empresa'] = empresa
        
        # Predecir
        print(f"🔍 Detectando anomalías en {len(df)} registros...")
        resultados = self.predictor.predict_batch(df, mostrar_tiempo=True)
        
        # Generar alertas para anomalías
        anomalias = resultados[resultados['Es_Anomalia'] == True]
        if len(anomalias) > 0:
            print(f"\n🚨 {len(anomalias)} anomalías detectadas - Generando alertas...")
            for idx, row in anomalias.iterrows():
                alerta = self._crear_alerta(row.to_dict(), row.to_dict())
                self._guardar_alerta(alerta)
                self._enviar_alerta(alerta)
        
        # Guardar resultados
        if output_path:
            resultados.to_csv(output_path, index=False, encoding='utf-8')
            print(f"💾 Resultados guardados en: {output_path}")
        
        return resultados
    
    def _crear_alerta(self, dato, resultado):
        """Crea un objeto de alerta"""
        alerta = {
            'timestamp': datetime.now().isoformat(),
            'fecha_dato': dato.get('Fecha', dato.get('Fecha y hora', 'N/A')),
            'empresa': dato.get('Empresa', 'Desconocida'),
            'es_anomalia': bool(resultado.get('Es_Anomalia', False)),
            'score': float(resultado.get('Score_Anomalia', 0)),
            'severidad': self._calcular_severidad(resultado.get('Score_Anomalia', 0)),
            'datos': {
                'Generación total': dato.get('Generación total', 'N/A'),
                'Consumo total': dato.get('Consumo total', 'N/A'),
                'Autoconsumo': dato.get('Autoconsumo', 'N/A'),
            },
            'tiempo_prediccion_ms': resultado.get('tiempo_prediccion_ms', 0)
        }
        return alerta
    
    def _calcular_severidad(self, score):
        """Calcula severidad basada en el score"""
        if score < -0.5:
            return 'CRÍTICA'
        elif score < -0.3:
            return 'ALTA'
        elif score < -0.1:
            return 'MEDIA'
        else:
            return 'BAJA'
    
    def _guardar_alerta(self, alerta):
        """Guarda alerta en historial"""
        self.alertas_historial.append(alerta)
        
        # Mantener solo últimas 1000 alertas
        if len(self.alertas_historial) > 1000:
            self.alertas_historial = self.alertas_historial[-1000:]
        
        # Guardar en archivo
        with open(self.alertas_file, 'w', encoding='utf-8') as f:
            json.dump(self.alertas_historial, f, indent=2, ensure_ascii=False, default=str)
    
    def _enviar_alerta(self, alerta):
        """Envía alerta (puedes personalizar aquí)"""
        print(f"\n🚨 ALERTA GENERADA:")
        print(f"   Timestamp: {alerta['timestamp']}")
        print(f"   Empresa: {alerta['empresa']}")
        print(f"   Fecha dato: {alerta['fecha_dato']}")
        print(f"   Severidad: {alerta['severidad']}")
        print(f"   Score: {alerta['score']:.4f}")
        print(f"   Generación: {alerta['datos']['Generación total']}")
        print(f"   Consumo: {alerta['datos']['Consumo total']}")
        
        # Aquí puedes agregar:
        # - Envío de email
        # - Notificación push
        # - Llamada a API
        # - Guardar en base de datos
        # - etc.
        
        # Ejemplo: Guardar alerta crítica en archivo separado
        if alerta['severidad'] == 'CRÍTICA':
            self._guardar_alerta_critica(alerta)
    
    def _guardar_alerta_critica(self, alerta):
        """Guarda alertas críticas en archivo separado"""
        criticas_file = 'alertas_criticas.json'
        
        if os.path.exists(criticas_file):
            with open(criticas_file, 'r', encoding='utf-8') as f:
                criticas = json.load(f)
        else:
            criticas = []
        
        criticas.append(alerta)
        
        with open(criticas_file, 'w', encoding='utf-8') as f:
            json.dump(criticas, f, indent=2, ensure_ascii=False, default=str)
    
    def monitorear_carpeta(self, carpeta='.', patron='*.csv', intervalo=60):
        """
        Monitorea una carpeta buscando nuevos archivos CSV
        
        Parameters:
        -----------
        carpeta : str
            Carpeta a monitorear
        patron : str
            Patrón de archivos (ej: 'datos_nuevos_*.csv')
        intervalo : int
            Intervalo en segundos para verificar
        """
        print("=" * 60)
        print("👁️ MONITOREO DE CARPETA EN TIEMPO REAL")
        print("=" * 60)
        print(f"\n📂 Monitoreando: {carpeta}")
        print(f"🔍 Patrón: {patron}")
        print(f"⏱️ Intervalo: {intervalo} segundos")
        print("   (Presiona Ctrl+C para detener)\n")
        
        self.running = True
        archivos_procesados = set()
        
        try:
            while self.running:
                # Buscar archivos nuevos
                archivos = glob(os.path.join(carpeta, patron))
                
                for archivo in archivos:
                    if archivo not in archivos_procesados:
                        print(f"\n📥 Nuevo archivo detectado: {os.path.basename(archivo)}")
                        
                        # Procesar
                        resultados = self.procesar_csv(archivo)
                        
                        # Marcar como procesado
                        archivos_procesados.add(archivo)
                        
                        # Resumen
                        anomalias = resultados[resultados['Es_Anomalia'] == True]
                        print(f"✅ Procesado: {len(anomalias)} anomalías detectadas")
                
                # Esperar antes de siguiente verificación
                time.sleep(intervalo)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoreo detenido")
            self.running = False
    
    def obtener_resumen_alertas(self, dias=7):
        """
        Obtiene resumen de alertas de los últimos días
        
        Parameters:
        -----------
        dias : int
            Número de días a revisar
            
        Returns:
        --------
        dict : Resumen de alertas
        """
        if not self.alertas_historial:
            return {"total": 0, "mensaje": "No hay alertas en el historial"}
        
        fecha_limite = datetime.now().timestamp() - (dias * 24 * 60 * 60)
        
        alertas_recientes = [
            a for a in self.alertas_historial
            if datetime.fromisoformat(a['timestamp']).timestamp() > fecha_limite
        ]
        
        resumen = {
            'total': len(alertas_recientes),
            'por_severidad': {},
            'por_empresa': {},
            'ultimas_24h': 0
        }
        
        fecha_24h = datetime.now().timestamp() - (24 * 60 * 60)
        
        for alerta in alertas_recientes:
            # Por severidad
            sev = alerta['severidad']
            resumen['por_severidad'][sev] = resumen['por_severidad'].get(sev, 0) + 1
            
            # Por empresa
            emp = alerta['empresa']
            resumen['por_empresa'][emp] = resumen['por_empresa'].get(emp, 0) + 1
            
            # Últimas 24h
            if datetime.fromisoformat(alerta['timestamp']).timestamp() > fecha_24h:
                resumen['ultimas_24h'] += 1
        
        return resumen


def ejemplo_uso_individual():
    """Ejemplo: Procesar un dato individual"""
    print("=" * 60)
    print("⚡ EJEMPLO: PROCESAR DATO INDIVIDUAL")
    print("=" * 60)
    
    sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
    
    if not sistema.iniciar():
        return
    
    # Simular dato nuevo que llega
    nuevo_dato = {
        'Fecha y hora': '15.01.2025',
        'Generación total': 5000.0,  # Generación muy baja
        'Consumo total': 80000.0,     # Consumo muy alto
        'Autoconsumo': 4800.0,
        'Energía suministrada a la red': 200.0,
        'Energía obtenida de la red': 75200.0,
        'Empresa': 'NEUROCIENCIAS'
    }
    
    print(f"\n📥 Procesando nuevo dato...")
    resultado = sistema.procesar_dato(nuevo_dato)
    
    print(f"\n📊 RESULTADO:")
    print(f"   Es Anomalía: {'🚨 SÍ' if resultado['Es_Anomalia'] else '✅ NO'}")
    print(f"   Score: {resultado['Score_Anomalia']:.4f}")
    
    # Resumen de alertas
    resumen = sistema.obtener_resumen_alertas(dias=7)
    print(f"\n📈 RESUMEN DE ALERTAS (últimos 7 días):")
    print(f"   Total: {resumen.get('total', 0)}")
    print(f"   Últimas 24h: {resumen.get('ultimas_24h', 0)}")


def ejemplo_monitoreo_continuo():
    """Ejemplo: Monitoreo continuo de carpeta"""
    print("=" * 60)
    print("👁️ EJEMPLO: MONITOREO CONTINUO")
    print("=" * 60)
    
    sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
    
    if not sistema.iniciar():
        return
    
    # Monitorear carpeta actual buscando archivos nuevos
    carpeta = input("Carpeta a monitorear (Enter para actual): ").strip() or '.'
    patron = input("Patrón de archivos (default: datos_nuevos_*.csv): ").strip() or 'datos_nuevos_*.csv'
    intervalo = int(input("Intervalo en segundos (default: 60): ").strip() or "60")
    
    sistema.monitorear_carpeta(carpeta, patron, intervalo)


def main():
    """Menú principal"""
    print("=" * 60)
    print("🚨 SISTEMA DE ALERTAS EN TIEMPO REAL")
    print("=" * 60)
    
    print("\n¿Qué deseas hacer?")
    print("1. Procesar un dato individual")
    print("2. Procesar archivo CSV")
    print("3. Monitorear carpeta continuamente")
    print("4. Ver resumen de alertas")
    print("5. Ejemplo completo")
    
    opcion = input("\nSelecciona (1-5): ").strip()
    
    sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
    
    if opcion == '1':
        if sistema.iniciar():
            # Pedir datos
            print("\n📥 Ingresa los datos del registro:")
            dato = {
                'Fecha y hora': input("Fecha (dd.MM.yyyy): ").strip(),
                'Generación total': float(input("Generación total: ").strip() or "0"),
                'Consumo total': float(input("Consumo total: ").strip() or "0"),
                'Autoconsumo': float(input("Autoconsumo: ").strip() or "0"),
                'Energía suministrada a la red': float(input("Energía suministrada: ").strip() or "0"),
                'Energía obtenida de la red': float(input("Energía obtenida: ").strip() or "0"),
            }
            empresa = input("Empresa: ").strip() or None
            
            sistema.procesar_dato(dato, empresa=empresa)
    
    elif opcion == '2':
        if sistema.iniciar():
            csv_path = input("Ruta al archivo CSV: ").strip()
            empresa = input("Empresa (opcional): ").strip() or None
            
            if os.path.exists(csv_path):
                sistema.procesar_csv(csv_path, empresa=empresa, 
                                    output_path='resultados_alertas.csv')
            else:
                print(f"❌ Archivo no encontrado: {csv_path}")
    
    elif opcion == '3':
        ejemplo_monitoreo_continuo()
    
    elif opcion == '4':
        if sistema.iniciar():
            dias = int(input("Días a revisar (default: 7): ").strip() or "7")
            resumen = sistema.obtener_resumen_alertas(dias=dias)
            
            print(f"\n📊 RESUMEN DE ALERTAS (últimos {dias} días):")
            print(f"   Total: {resumen.get('total', 0)}")
            print(f"   Últimas 24h: {resumen.get('ultimas_24h', 0)}")
            
            if resumen.get('por_severidad'):
                print(f"\n   Por Severidad:")
                for sev, count in resumen['por_severidad'].items():
                    print(f"     {sev}: {count}")
            
            if resumen.get('por_empresa'):
                print(f"\n   Por Empresa:")
                for emp, count in list(resumen['por_empresa'].items())[:10]:
                    print(f"     {emp}: {count}")
    
    elif opcion == '5':
        ejemplo_uso_individual()
        print("\n" + "="*60)
        print("💡 Para monitoreo continuo, usa opción 3")
    else:
        print("❌ Opción no válida")


if __name__ == "__main__":
    main()

