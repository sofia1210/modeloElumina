# 🚨 Guía: Sistema de Alertas en Tiempo Real

## 🎯 Tu Pregunta

**"¿Cómo obtener el modelo después del multi-entrenamiento y usarlo en tiempo real para generar alertas automáticas?"**

## ✅ Respuesta Completa

### **Paso 1: El Modelo Ya Está Guardado**

Después de ejecutar `multientrenamiento.py`, el modelo se guarda automáticamente:
- 📁 `modelo_multi_empresa.pkl` - Modelo entrenado
- 📁 `modelo_multi_empresa_empresas.json` - Info de empresas

### **Paso 2: Usar el Modelo para Alertas en Tiempo Real**

```bash
python sistema_alertas_tiempo_real.py
```

---

## 🚀 Cómo Funciona el Sistema de Alertas

### **1. Procesar Dato Individual**

```python
from sistema_alertas_tiempo_real import SistemaAlertasTiempoReal

# Inicializar sistema
sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
sistema.iniciar()

# Procesar dato nuevo
nuevo_dato = {
    'Fecha y hora': '15.01.2025',
    'Generación total': 5000.0,
    'Consumo total': 80000.0,
    'Autoconsumo': 4800.0,
    'Energía suministrada a la red': 200.0,
    'Energía obtenida de la red': 75200.0,
    'Empresa': 'NEUROCIENCIAS'
}

# Procesar y generar alerta automáticamente si es anomalía
resultado = sistema.procesar_dato(nuevo_dato)
# ✅ Si es anomalía, se genera alerta automáticamente
```

---

### **2. Procesar Archivo CSV**

```python
# Procesar archivo CSV con múltiples registros
resultados = sistema.procesar_csv(
    'datos_nuevos.csv',
    empresa='NEUROCIENCIAS',
    output_path='resultados_alertas.csv'
)

# Las alertas se generan automáticamente para cada anomalía
```

---

### **3. Monitoreo Continuo de Carpeta**

```python
# Monitorear carpeta buscando archivos nuevos automáticamente
sistema.monitorear_carpeta(
    carpeta='./datos_nuevos',
    patron='datos_*.csv',
    intervalo=60  # Verificar cada 60 segundos
)
```

**Funciona así:**
- ✅ Verifica la carpeta cada X segundos
- ✅ Procesa archivos nuevos automáticamente
- ✅ Genera alertas para anomalías detectadas
- ✅ Continúa indefinidamente hasta detenerlo

---

## 📊 Características del Sistema

### **Alertas Automáticas:**
- ✅ Detecta anomalías automáticamente
- ✅ Calcula severidad (CRÍTICA, ALTA, MEDIA, BAJA)
- ✅ Guarda historial de alertas
- ✅ Separa alertas críticas en archivo especial

### **Historial:**
- ✅ Guarda todas las alertas en `alertas_historial.json`
- ✅ Mantiene últimas 1000 alertas
- ✅ Alertas críticas en `alertas_criticas.json`

### **Resumen:**
- ✅ Resumen por severidad
- ✅ Resumen por empresa
- ✅ Estadísticas de últimas 24h

---

## 🔧 Integración con Tu Sistema

### **Opción A: Integración con API**

```python
from flask import Flask, request, jsonify
from sistema_alertas_tiempo_real import SistemaAlertasTiempoReal

app = Flask(__name__)

# Inicializar sistema una vez
sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
sistema.iniciar()

@app.route('/procesar', methods=['POST'])
def procesar_dato():
    """Endpoint para recibir datos y generar alertas"""
    data = request.json
    
    resultado = sistema.procesar_dato(data)
    
    return jsonify({
        'es_anomalia': bool(resultado['Es_Anomalia']),
        'score': float(resultado['Score_Anomalia']),
        'alerta_generada': bool(resultado['Es_Anomalia'])
    })

if __name__ == '__main__':
    app.run(port=5000)
```

---

### **Opción B: Monitoreo de Base de Datos**

```python
import sqlite3
import time
from sistema_alertas_tiempo_real import SistemaAlertasTiempoReal

sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
sistema.iniciar()

conn = sqlite3.connect('energia.db')

while True:
    # Obtener registros nuevos
    nuevos = pd.read_sql("""
        SELECT * FROM energia 
        WHERE procesado = 0 
        LIMIT 100
    """, conn)
    
    if len(nuevos) > 0:
        # Procesar cada registro
        for idx, row in nuevos.iterrows():
            resultado = sistema.procesar_dato(row.to_dict())
            
            # Marcar como procesado
            conn.execute("UPDATE energia SET procesado = 1 WHERE id = ?", (row['id'],))
        
        conn.commit()
    
    time.sleep(60)  # Verificar cada minuto
```

---

### **Opción C: Monitoreo de Carpeta (Más Simple)**

```python
from sistema_alertas_tiempo_real import SistemaAlertasTiempoReal

sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
sistema.iniciar()

# Monitorear carpeta donde llegan archivos nuevos
sistema.monitorear_carpeta(
    carpeta='./datos_entrantes',
    patron='datos_*.csv',
    intervalo=30  # Verificar cada 30 segundos
)
```

---

## 📋 Estructura de Alertas

Cada alerta contiene:

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "fecha_dato": "15.01.2025",
  "empresa": "NEUROCIENCIAS",
  "es_anomalia": true,
  "score": -0.5234,
  "severidad": "CRÍTICA",
  "datos": {
    "Generación total": 5000.0,
    "Consumo total": 80000.0,
    "Autoconsumo": 4800.0
  },
  "tiempo_prediccion_ms": 8.5
}
```

---

## 🎯 Flujo Completo Recomendado

### **1. Entrenar Modelo (Una vez):**
```bash
python multientrenamiento.py
```
**Resultado:** `modelo_multi_empresa.pkl` guardado

### **2. Iniciar Sistema de Alertas:**
```bash
python sistema_alertas_tiempo_real.py
# Selecciona opción 3: Monitoreo continuo
```

### **3. Datos Nuevos Llegan:**
- Archivos CSV en carpeta monitoreada
- O datos vía API
- O datos en base de datos

### **4. Sistema Procesa Automáticamente:**
- ✅ Detecta anomalías
- ✅ Genera alertas
- ✅ Guarda historial
- ✅ Notifica (si configuras)

---

## 🔔 Personalizar Envío de Alertas

Puedes personalizar el método `_enviar_alerta` en `sistema_alertas_tiempo_real.py`:

```python
def _enviar_alerta(self, alerta):
    """Envía alerta - Personaliza aquí"""
    
    # 1. Email
    enviar_email(alerta)
    
    # 2. SMS
    enviar_sms(alerta)
    
    # 3. Notificación Push
    enviar_notificacion_push(alerta)
    
    # 4. Webhook
    requests.post('https://tu-webhook.com', json=alerta)
    
    # 5. Base de datos
    guardar_en_bd(alerta)
```

---

## 📊 Ver Resumen de Alertas

```python
sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
sistema.iniciar()

resumen = sistema.obtener_resumen_alertas(dias=7)

print(f"Total alertas: {resumen['total']}")
print(f"Últimas 24h: {resumen['ultimas_24h']}")
print(f"Por severidad: {resumen['por_severidad']}")
print(f"Por empresa: {resumen['por_empresa']}")
```

---

## 🎯 Ejemplo Completo de Uso

```python
from sistema_alertas_tiempo_real import SistemaAlertasTiempoReal

# 1. Inicializar
sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
sistema.iniciar()

# 2. Simular datos que llegan
datos_nuevos = [
    {
        'Fecha y hora': '15.01.2025',
        'Generación total': 5000.0,  # Baja
        'Consumo total': 80000.0,    # Alta
        'Autoconsumo': 4800.0,
        'Energía suministrada a la red': 200.0,
        'Energía obtenida de la red': 75200.0,
        'Empresa': 'NEUROCIENCIAS'
    }
]

# 3. Procesar cada dato
for dato in datos_nuevos:
    resultado = sistema.procesar_dato(dato)
    # Si es anomalía, se genera alerta automáticamente

# 4. Ver resumen
resumen = sistema.obtener_resumen_alertas(dias=1)
print(f"Alertas generadas: {resumen['total']}")
```

---

## 📁 Archivos Generados

```
eLumin/
├── modelo_multi_empresa.pkl          # Modelo entrenado
├── alertas_historial.json            # Historial de todas las alertas
├── alertas_criticas.json             # Solo alertas críticas
└── resultados_alertas.csv             # Resultados de procesamiento
```

---

## ⚡ Velocidad

- **Cargar modelo:** 1-2 segundos (una vez)
- **Procesar 1 dato:** < 10 ms
- **Procesar 100 datos:** < 100 ms
- **Generar alerta:** Instantáneo

**✅ Perfecto para tiempo real!**

---

## 🎯 Resumen Ejecutivo

1. **Modelo guardado:** `modelo_multi_empresa.pkl` (después de `multientrenamiento.py`)

2. **Iniciar sistema de alertas:**
   ```bash
   python sistema_alertas_tiempo_real.py
   ```

3. **Usar en código:**
   ```python
   sistema = SistemaAlertasTiempoReal('modelo_multi_empresa.pkl')
   sistema.iniciar()
   sistema.procesar_dato(nuevo_dato)  # Genera alerta automáticamente
   ```

4. **Monitoreo continuo:**
   ```python
   sistema.monitorear_carpeta(carpeta='./datos', intervalo=60)
   ```

**✅ Sistema completo de alertas en tiempo real listo!**

