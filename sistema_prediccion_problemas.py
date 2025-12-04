"""
Sistema Avanzado de Detección y Predicción de Problemas
Detecta anomalías actuales y analiza tendencias para predecir problemas futuros
"""

import pandas as pd
import numpy as np
from sistema_alertas_tiempo_real import SistemaAlertasTiempoReal
from datetime import datetime, timedelta
import json
import os

class SistemaPrediccionProblemas:
    """
    Sistema que detecta anomalías actuales y predice problemas futuros
    """
    
    def __init__(self, model_path='modelo_multi_empresa.pkl'):
        """
        Inicializa el sistema de predicción
        
        Parameters:
        -----------
        model_path : str
            Ruta al modelo entrenado
        """
        self.model_path = model_path
        self.sistema_alertas = SistemaAlertasTiempoReal(model_path)
        self.historial_datos = []
        self.historial_file = 'historial_datos.json'
        
    def iniciar(self):
        """Inicia el sistema"""
        print("=" * 60)
        print("🔮 SISTEMA DE PREDICCIÓN DE PROBLEMAS")
        print("=" * 60)
        
        if not self.sistema_alertas.iniciar():
            return False
        
        # Cargar historial si existe
        if os.path.exists(self.historial_file):
            with open(self.historial_file, 'r', encoding='utf-8') as f:
                self.historial_datos = json.load(f)
            print(f"✅ Historial cargado: {len(self.historial_datos)} registros")
        
        return True
    
    def analizar_dato(self, dato, empresa=None):
        """
        Analiza un dato: detecta anomalía actual y predice problemas futuros
        
        Parameters:
        -----------
        dato : dict
            Dato a analizar
        empresa : str, optional
            Nombre de la empresa
            
        Returns:
        --------
        dict : Análisis completo con detección y predicción
        """
        # Agregar empresa si no existe
        if 'Empresa' not in dato and empresa:
            dato['Empresa'] = empresa
        
        # 1. Detectar anomalía actual
        resultado_actual = self.sistema_alertas.procesar_dato(dato, empresa=empresa, enviar_alerta=False)
        
        # 2. Analizar tendencias para predecir problemas
        prediccion = self._analizar_tendencias(dato, empresa)
        
        # 3. Clasificar tipo de problema
        tipo_problema = self._clasificar_problema(dato, resultado_actual, prediccion)
        
        # 4. Generar alerta mejorada
        alerta_mejorada = self._crear_alerta_mejorada(
            dato, resultado_actual, prediccion, tipo_problema
        )
        
        # 5. Guardar en historial
        self._guardar_en_historial(dato, resultado_actual, prediccion)
        
        # 6. Enviar alerta si es necesario
        if resultado_actual['Es_Anomalia'] or prediccion['riesgo_futuro']:
            self.sistema_alertas._guardar_alerta(alerta_mejorada)
            self.sistema_alertas._enviar_alerta(alerta_mejorada)
        
        return {
            'anomalia_actual': resultado_actual,
            'prediccion': prediccion,
            'tipo_problema': tipo_problema,
            'alerta': alerta_mejorada
        }
    
    def _analizar_tendencias(self, dato, empresa):
        """
        Analiza tendencias históricas para predecir problemas futuros
        
        Parameters:
        -----------
        dato : dict
            Dato actual
        empresa : str
            Nombre de la empresa
            
        Returns:
        --------
        dict : Predicción de problemas futuros
        """
        # Filtrar historial de la misma empresa
        historial_empresa = [
            h for h in self.historial_datos 
            if h.get('empresa') == (empresa or dato.get('Empresa', ''))
        ]
        
        if len(historial_empresa) < 3:
            return {
                'riesgo_futuro': False,
                'confianza': 0.0,
                'mensaje': 'Datos insuficientes para predicción',
                'tendencias': {}
            }
        
        # Obtener últimos N registros (últimos 7 días)
        ultimos_registros = sorted(historial_empresa, 
                                  key=lambda x: x.get('timestamp', ''))[-7:]
        
        # Analizar tendencias
        generacion_tendencia = self._calcular_tendencia(
            [r['datos'].get('Generación total', 0) for r in ultimos_registros],
            dato.get('Generación total', 0)
        )
        
        consumo_tendencia = self._calcular_tendencia(
            [r['datos'].get('Consumo total', 0) for r in ultimos_registros],
            dato.get('Consumo total', 0)
        )
        
        eficiencia_tendencia = self._calcular_tendencia(
            [r['datos'].get('Autoconsumo', 0) / max(r['datos'].get('Generación total', 1), 1) 
             for r in ultimos_registros],
            dato.get('Autoconsumo', 0) / max(dato.get('Generación total', 1), 1)
        )
        
        # Detectar patrones de riesgo
        riesgos = []
        riesgo_score = 0.0
        
        # Riesgo 1: Consumo aumentando rápidamente
        if consumo_tendencia['tendencia'] == 'aumentando' and consumo_tendencia['velocidad'] > 0.15:
            riesgos.append('Consumo aumentando rápidamente - Posible sobrecarga futura')
            riesgo_score += 0.3
        
        # Riesgo 2: Generación disminuyendo
        if generacion_tendencia['tendencia'] == 'disminuyendo' and generacion_tendencia['velocidad'] > 0.1:
            riesgos.append('Generación disminuyendo - Posible fallo en paneles')
            riesgo_score += 0.4
        
        # Riesgo 3: Eficiencia disminuyendo
        if eficiencia_tendencia['tendencia'] == 'disminuyendo' and eficiencia_tendencia['velocidad'] > 0.1:
            riesgos.append('Eficiencia de autoconsumo disminuyendo - Sistema perdiendo rendimiento')
            riesgo_score += 0.3
        
        # Riesgo 4: Patrón de degradación
        if self._detectar_degradacion(ultimos_registros):
            riesgos.append('Patrón de degradación detectado - Mantenimiento recomendado')
            riesgo_score += 0.5
        
        return {
            'riesgo_futuro': riesgo_score > 0.3,
            'riesgo_score': riesgo_score,
            'confianza': min(len(ultimos_registros) / 7, 1.0),
            'mensaje': '; '.join(riesgos) if riesgos else 'Sin riesgos detectados',
            'tendencias': {
                'generacion': generacion_tendencia,
                'consumo': consumo_tendencia,
                'eficiencia': eficiencia_tendencia
            },
            'riesgos_detectados': riesgos
        }
    
    def _calcular_tendencia(self, valores_historicos, valor_actual):
        """Calcula tendencia de una serie de valores"""
        if len(valores_historicos) < 2:
            return {'tendencia': 'insuficiente', 'velocidad': 0.0}
        
        # Calcular cambio porcentual promedio
        cambios = []
        for i in range(1, len(valores_historicos)):
            if valores_historicos[i-1] > 0:
                cambio = (valores_historicos[i] - valores_historicos[i-1]) / valores_historicos[i-1]
                cambios.append(cambio)
        
        if not cambios:
            return {'tendencia': 'estable', 'velocidad': 0.0}
        
        cambio_promedio = np.mean(cambios)
        cambio_actual = (valor_actual - valores_historicos[-1]) / max(valores_historicos[-1], 1) if valores_historicos[-1] > 0 else 0
        
        # Determinar tendencia
        if cambio_promedio > 0.05:
            tendencia = 'aumentando'
        elif cambio_promedio < -0.05:
            tendencia = 'disminuyendo'
        else:
            tendencia = 'estable'
        
        return {
            'tendencia': tendencia,
            'velocidad': abs(cambio_promedio),
            'cambio_actual': cambio_actual,
            'valores_historicos': valores_historicos[-5:]  # Últimos 5
        }
    
    def _detectar_degradacion(self, registros):
        """Detecta patrón de degradación continua"""
        if len(registros) < 5:
            return False
        
        # Analizar eficiencia a lo largo del tiempo
        eficiencias = []
        for r in registros:
            gen = r['datos'].get('Generación total', 0)
            auto = r['datos'].get('Autoconsumo', 0)
            if gen > 0:
                eficiencias.append(auto / gen)
        
        if len(eficiencias) < 5:
            return False
        
        # Si hay tendencia decreciente consistente
        tendencia = np.polyfit(range(len(eficiencias)), eficiencias, 1)[0]
        return tendencia < -0.01  # Degradación de más del 1% por día
    
    def _clasificar_problema(self, dato, resultado_actual, prediccion):
        """
        Clasifica el tipo de problema detectado
        
        Returns:
        --------
        dict : Clasificación del problema
        """
        generacion = dato.get('Generación total', 0)
        consumo = dato.get('Consumo total', 0)
        autoconsumo = dato.get('Autoconsumo', 0)
        
        problemas = []
        severidad = 'BAJA'
        
        # Problema 1: Anomalía actual detectada
        if resultado_actual['Es_Anomalia']:
            score = resultado_actual['Score_Anomalia']
            
            # Analizar qué tipo de anomalía
            if generacion < consumo * 0.1:
                problemas.append('FALLO EN GENERACIÓN')
                severidad = 'CRÍTICA'
            elif consumo > generacion * 3:
                problemas.append('CONSUMO EXCESIVO')
                severidad = 'ALTA'
            elif autoconsumo / max(generacion, 1) < 0.5:
                problemas.append('BAJA EFICIENCIA DE AUTOCONSUMO')
                severidad = 'MEDIA'
            else:
                problemas.append('ANOMALÍA GENERAL')
                severidad = 'MEDIA'
        
        # Problema 2: Riesgo futuro
        if prediccion['riesgo_futuro']:
            if prediccion['riesgo_score'] > 0.7:
                problemas.append('RIESGO CRÍTICO FUTURO')
                severidad = 'CRÍTICA' if severidad != 'CRÍTICA' else severidad
            elif prediccion['riesgo_score'] > 0.5:
                problemas.append('RIESGO ALTO FUTURO')
                severidad = 'ALTA' if severidad not in ['CRÍTICA', 'ALTA'] else severidad
            else:
                problemas.append('RIESGO MODERADO FUTURO')
                severidad = 'MEDIA' if severidad not in ['CRÍTICA', 'ALTA', 'MEDIA'] else severidad
        
        return {
            'tipos': problemas,
            'severidad': severidad,
            'es_anomalia_actual': resultado_actual['Es_Anomalia'],
            'hay_riesgo_futuro': prediccion['riesgo_futuro'],
            'descripcion': self._generar_descripcion(problemas, dato, prediccion)
        }
    
    def _generar_descripcion(self, problemas, dato, prediccion):
        """Genera descripción legible del problema"""
        descripciones = []
        
        if 'FALLO EN GENERACIÓN' in problemas:
            descripciones.append("⚠️ FALLO DETECTADO: La generación es extremadamente baja. "
                               "Posible fallo en paneles solares o sistema de generación.")
        
        if 'CONSUMO EXCESIVO' in problemas:
            descripciones.append("⚠️ CONSUMO EXCESIVO: El consumo es mucho mayor que la generación. "
                               "Revisar equipos o posibles cortocircuitos.")
        
        if 'BAJA EFICIENCIA DE AUTOCONSUMO' in problemas:
            descripciones.append("⚠️ EFICIENCIA BAJA: El sistema no está aprovechando bien la energía generada.")
        
        if 'RIESGO CRÍTICO FUTURO' in problemas:
            descripciones.append(f"🔮 PREDICCIÓN: {prediccion['mensaje']} - "
                               "Se recomienda acción inmediata para prevenir fallos.")
        
        if 'RIESGO ALTO FUTURO' in problemas:
            descripciones.append(f"🔮 PREDICCIÓN: {prediccion['mensaje']} - "
                               "Monitorear de cerca en los próximos días.")
        
        if not descripciones:
            return "✅ Sistema funcionando normalmente"
        
        return " | ".join(descripciones)
    
    def _crear_alerta_mejorada(self, dato, resultado_actual, prediccion, tipo_problema):
        """Crea alerta mejorada con información de predicción"""
        alerta = {
            'timestamp': datetime.now().isoformat(),
            'fecha_dato': dato.get('Fecha', dato.get('Fecha y hora', 'N/A')),
            'empresa': dato.get('Empresa', 'Desconocida'),
            'tipo_alerta': 'ANOMALÍA ACTUAL' if resultado_actual['Es_Anomalia'] else 'PREDICCIÓN',
            'es_anomalia_actual': bool(resultado_actual['Es_Anomalia']),
            'hay_riesgo_futuro': bool(prediccion['riesgo_futuro']),
            'score_anomalia': float(resultado_actual.get('Score_Anomalia', 0)),
            'riesgo_score': float(prediccion.get('riesgo_score', 0)),
            'severidad': tipo_problema['severidad'],
            'tipos_problema': tipo_problema['tipos'],
            'descripcion': tipo_problema['descripcion'],
            'prediccion_mensaje': prediccion.get('mensaje', ''),
            'datos': {
                'Generación total': dato.get('Generación total', 'N/A'),
                'Consumo total': dato.get('Consumo total', 'N/A'),
                'Autoconsumo': dato.get('Autoconsumo', 'N/A'),
            },
            'tendencias': prediccion.get('tendencias', {})
        }
        
        return alerta
    
    def _guardar_en_historial(self, dato, resultado, prediccion):
        """Guarda dato en historial para análisis futuro"""
        registro = {
            'timestamp': datetime.now().isoformat(),
            'empresa': dato.get('Empresa', 'Desconocida'),
            'datos': {
                'Generación total': dato.get('Generación total', 0),
                'Consumo total': dato.get('Consumo total', 0),
                'Autoconsumo': dato.get('Autoconsumo', 0),
            },
            'es_anomalia': bool(resultado['Es_Anomalia']),
            'score': float(resultado.get('Score_Anomalia', 0))
        }
        
        self.historial_datos.append(registro)
        
        # Mantener solo últimos 1000 registros
        if len(self.historial_datos) > 1000:
            self.historial_datos = self.historial_datos[-1000:]
        
        # Guardar en archivo
        with open(self.historial_file, 'w', encoding='utf-8') as f:
            json.dump(self.historial_datos, f, indent=2, ensure_ascii=False, default=str)


def ejemplo_uso():
    """Ejemplo de uso del sistema"""
    print("=" * 60)
    print("🔮 EJEMPLO: DETECCIÓN Y PREDICCIÓN DE PROBLEMAS")
    print("=" * 60)
    
    sistema = SistemaPrediccionProblemas('modelo_multi_empresa.pkl')
    
    if not sistema.iniciar():
        return
    
    # Simular datos con diferentes escenarios
    escenarios = [
        {
            'nombre': 'Anomalía Actual: Fallo en Generación',
            'dato': {
                'Fecha y hora': '15.01.2025',
                'Generación total': 1000.0,  # Muy baja
                'Consumo total': 50000.0,
                'Autoconsumo': 950.0,
                'Energía suministrada a la red': 50.0,
                'Energía obtenida de la red': 49050.0,
                'Empresa': 'NEUROCIENCIAS'
            }
        },
        {
            'nombre': 'Predicción: Tendencia de Degradación',
            'dato': {
                'Fecha y hora': '16.01.2025',
                'Generación total': 20000.0,  # Disminuyendo
                'Consumo total': 60000.0,    # Aumentando
                'Autoconsumo': 18000.0,
                'Energía suministrada a la red': 2000.0,
                'Energía obtenida de la red': 42000.0,
                'Empresa': 'BIOPETROL_BEREA'
            }
        }
    ]
    
    for escenario in escenarios:
        print(f"\n{'='*60}")
        print(f"📊 ESCENARIO: {escenario['nombre']}")
        print(f"{'='*60}")
        
        # Primero agregar algunos datos históricos para la predicción
        for i in range(5):
            dato_historico = escenario['dato'].copy()
            dato_historico['Generación total'] = escenario['dato']['Generación total'] * (1.1 - i * 0.02)
            dato_historico['Consumo total'] = escenario['dato']['Consumo total'] * (0.9 + i * 0.02)
            sistema._guardar_en_historial(dato_historico, {'Es_Anomalia': False, 'Score_Anomalia': 0.1}, {})
        
        # Analizar
        resultado = sistema.analizar_dato(escenario['dato'])
        
        # Mostrar resultados
        print(f"\n📊 RESULTADO DEL ANÁLISIS:")
        print(f"   Anomalía Actual: {'🚨 SÍ' if resultado['anomalia_actual']['Es_Anomalia'] else '✅ NO'}")
        print(f"   Riesgo Futuro: {'🔮 SÍ' if resultado['prediccion']['riesgo_futuro'] else '✅ NO'}")
        print(f"   Severidad: {resultado['tipo_problema']['severidad']}")
        print(f"   Tipos de Problema: {', '.join(resultado['tipo_problema']['tipos'])}")
        print(f"\n   Descripción:")
        print(f"   {resultado['tipo_problema']['descripcion']}")
        
        if resultado['prediccion']['riesgo_futuro']:
            print(f"\n   🔮 Predicción:")
            print(f"   {resultado['prediccion']['mensaje']}")
            print(f"   Riesgo Score: {resultado['prediccion']['riesgo_score']:.2f}")


def main():
    """Función principal"""
    print("=" * 60)
    print("🔮 SISTEMA DE DETECCIÓN Y PREDICCIÓN DE PROBLEMAS")
    print("=" * 60)
    
    print("\nEste sistema:")
    print("✅ Detecta anomalías ACTUALES (problemas que ya están ocurriendo)")
    print("🔮 Predice problemas FUTUROS (analizando tendencias)")
    print("📊 Clasifica el tipo de problema (fallo, consumo excesivo, etc.)")
    print("🚨 Genera alertas descriptivas con recomendaciones")
    
    print("\n¿Qué deseas hacer?")
    print("1. Analizar un dato (detección + predicción)")
    print("2. Ver ejemplo completo")
    print("3. Procesar archivo CSV con análisis completo")
    
    opcion = input("\nSelecciona (1-3): ").strip()
    
    sistema = SistemaPrediccionProblemas('modelo_multi_empresa.pkl')
    
    if not sistema.iniciar():
        return
    
    if opcion == '1':
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
        
        resultado = sistema.analizar_dato(dato, empresa=empresa)
        
        print(f"\n📊 RESULTADO:")
        print(f"   Anomalía Actual: {'🚨 SÍ' if resultado['anomalia_actual']['Es_Anomalia'] else '✅ NO'}")
        print(f"   Riesgo Futuro: {'🔮 SÍ' if resultado['prediccion']['riesgo_futuro'] else '✅ NO'}")
        print(f"   {resultado['tipo_problema']['descripcion']}")
    
    elif opcion == '2':
        ejemplo_uso()
    
    elif opcion == '3':
        csv_path = input("Ruta al archivo CSV: ").strip()
        empresa = input("Empresa (opcional): ").strip() or None
        
        if os.path.exists(csv_path):
            # Cargar y procesar
            df = pd.read_csv(csv_path, encoding='utf-8', skiprows=[1] if True else None)
            df.columns = df.columns.str.strip()
            
            resultados = []
            for idx, row in df.iterrows():
                resultado = sistema.analizar_dato(row.to_dict(), empresa=empresa)
                resultados.append(resultado)
            
            print(f"\n✅ Procesados {len(resultados)} registros")
            print(f"📊 Anomalías actuales: {sum(1 for r in resultados if r['anomalia_actual']['Es_Anomalia'])}")
            print(f"🔮 Riesgos futuros: {sum(1 for r in resultados if r['prediccion']['riesgo_futuro'])}")
        else:
            print(f"❌ Archivo no encontrado: {csv_path}")
    
    else:
        print("❌ Opción no válida")


if __name__ == "__main__":
    main()

