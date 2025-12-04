# 🔮 Guía: Detección y Predicción de Problemas

## 🎯 Tu Pregunta

**"¿Puede ayudarme a predecir que habrá un problema/fallo, o simplemente detectar anomalías cuando ocurren?"**

## ✅ Respuesta Completa

El sistema hace **AMBAS cosas**:

1. **✅ Detecta Anomalías ACTUALES** - Problemas que ya están ocurriendo
2. **🔮 Predice Problemas FUTUROS** - Analiza tendencias para predecir fallos

---

## 📊 Dos Tipos de Detección

### **1. Detección de Anomalías Actuales** ✅

**¿Qué hace?**
- Detecta cuando hay un problema **AHORA**
- Ejemplo: "Generación muy baja hoy" → FALLO ACTUAL

**Tipos de problemas detectados:**
- 🚨 **FALLO EN GENERACIÓN**: Generación extremadamente baja
- ⚠️ **CONSUMO EXCESIVO**: Consumo mucho mayor que generación
- 📉 **BAJA EFICIENCIA**: Sistema no aprovecha bien la energía

**Mensaje:** "🚨 HAY UNA ANOMALÍA" (problema actual)

---

### **2. Predicción de Problemas Futuros** 🔮

**¿Qué hace?**
- Analiza tendencias históricas
- Predice problemas que **podrían ocurrir** en el futuro
- Ejemplo: "Consumo aumentando rápidamente" → Posible sobrecarga futura

**Tipos de predicciones:**
- 🔮 **RIESGO CRÍTICO FUTURO**: Problema grave probable en próximos días
- ⚠️ **RIESGO ALTO FUTURO**: Problema probable que requiere monitoreo
- 📊 **RIESGO MODERADO FUTURO**: Tendencia que podría convertirse en problema

**Mensaje:** "🔮 PREDICCIÓN: [descripción del riesgo futuro]"

---

## 🚀 Cómo Usar

### **Opción Rápida:**

```bash
python sistema_prediccion_problemas.py
```

### **Código Directo:**

```python
from sistema_prediccion_problemas import SistemaPrediccionProblemas

# Inicializar
sistema = SistemaPrediccionProblemas('modelo_multi_empresa.pkl')
sistema.iniciar()

# Analizar dato (detección + predicción)
dato = {
    'Fecha y hora': '15.01.2025',
    'Generación total': 5000.0,
    'Consumo total': 80000.0,
    'Autoconsumo': 4800.0,
    'Energía suministrada a la red': 200.0,
    'Energía obtenida de la red': 75200.0,
    'Empresa': 'NEUROCIENCIAS'
}

resultado = sistema.analizar_dato(dato)

# Resultado contiene:
# - anomalia_actual: Si hay problema ahora
# - prediccion: Si habrá problema futuro
# - tipo_problema: Qué tipo de problema es
# - alerta: Alerta completa con descripción
```

---

## 📋 Ejemplos de Alertas

### **Ejemplo 1: Anomalía Actual**

```
🚨 ALERTA: ANOMALÍA ACTUAL DETECTADA
   Tipo: FALLO EN GENERACIÓN
   Severidad: CRÍTICA
   Descripción: ⚠️ FALLO DETECTADO: La generación es extremadamente baja. 
                Posible fallo en paneles solares o sistema de generación.
   Score: -0.5234
```

### **Ejemplo 2: Predicción de Problema Futuro**

```
🔮 ALERTA: PREDICCIÓN DE PROBLEMA FUTURO
   Tipo: RIESGO CRÍTICO FUTURO
   Severidad: CRÍTICA
   Descripción: 🔮 PREDICCIÓN: Consumo aumentando rápidamente - Posible sobrecarga futura. 
                Se recomienda acción inmediata para prevenir fallos.
   Riesgo Score: 0.75
   Tendencias:
     - Consumo: Aumentando (15% por día)
     - Generación: Disminuyendo (8% por día)
```

### **Ejemplo 3: Ambos (Actual + Futuro)**

```
🚨🔮 ALERTA: ANOMALÍA ACTUAL + PREDICCIÓN
   Anomalía Actual: ✅ SÍ
   Riesgo Futuro: ✅ SÍ
   Tipos: CONSUMO EXCESIVO, RIESGO ALTO FUTURO
   Severidad: ALTA
   Descripción: ⚠️ CONSUMO EXCESIVO: El consumo es mucho mayor que la generación. 
                | 🔮 PREDICCIÓN: Generación disminuyendo - Posible fallo en paneles - 
                Monitorear de cerca en los próximos días.
```

---

## 🔍 Cómo Funciona la Predicción

### **Análisis de Tendencias:**

El sistema analiza:
1. **Tendencia de Generación**: ¿Está aumentando o disminuyendo?
2. **Tendencia de Consumo**: ¿Está aumentando o disminuyendo?
3. **Tendencia de Eficiencia**: ¿Mejora o empeora?
4. **Patrones de Degradación**: ¿Hay degradación continua?

### **Detección de Riesgos:**

- **Consumo aumentando rápidamente** (>15% por día)
  → Predice: Posible sobrecarga futura
  
- **Generación disminuyendo** (>10% por día)
  → Predice: Posible fallo en paneles
  
- **Eficiencia disminuyendo** (>10% por día)
  → Predice: Sistema perdiendo rendimiento
  
- **Patrón de degradación** (tendencia continua negativa)
  → Predice: Mantenimiento recomendado

---

## 📊 Clasificación de Problemas

El sistema clasifica automáticamente:

| Tipo de Problema | Descripción | Severidad |
|------------------|-------------|-----------|
| **FALLO EN GENERACIÓN** | Generación extremadamente baja | CRÍTICA |
| **CONSUMO EXCESIVO** | Consumo mucho mayor que generación | ALTA |
| **BAJA EFICIENCIA** | Sistema no aprovecha bien la energía | MEDIA |
| **RIESGO CRÍTICO FUTURO** | Problema grave probable | CRÍTICA |
| **RIESGO ALTO FUTURO** | Problema probable | ALTA |
| **RIESGO MODERADO FUTURO** | Tendencia preocupante | MEDIA |

---

## 🎯 Casos de Uso

### **Caso 1: Detectar Fallo Actual**

```python
# Dato con generación muy baja
dato = {
    'Generación total': 1000.0,  # Muy baja
    'Consumo total': 50000.0,
    ...
}

resultado = sistema.analizar_dato(dato)

# Resultado:
# anomalia_actual: True
# tipo_problema: ['FALLO EN GENERACIÓN']
# severidad: 'CRÍTICA'
# mensaje: "⚠️ FALLO DETECTADO: La generación es extremadamente baja..."
```

### **Caso 2: Predecir Problema Futuro**

```python
# Dato con tendencia de consumo aumentando
# (después de varios días de datos históricos)

resultado = sistema.analizar_dato(dato)

# Resultado:
# anomalia_actual: False (no hay problema ahora)
# prediccion.riesgo_futuro: True
# tipo_problema: ['RIESGO ALTO FUTURO']
# mensaje: "🔮 PREDICCIÓN: Consumo aumentando rápidamente..."
```

### **Caso 3: Ambos**

```python
# Dato con problema actual Y tendencia de empeoramiento

resultado = sistema.analizar_dato(dato)

# Resultado:
# anomalia_actual: True
# prediccion.riesgo_futuro: True
# tipo_problema: ['CONSUMO EXCESIVO', 'RIESGO CRÍTICO FUTURO']
# severidad: 'CRÍTICA'
```

---

## 📈 Ventajas del Sistema

### **1. Detección Inmediata:**
- ✅ Detecta problemas cuando ocurren
- ✅ Alertas instantáneas
- ✅ Clasificación automática del tipo

### **2. Predicción Preventiva:**
- 🔮 Predice problemas antes de que ocurran
- 🔮 Analiza tendencias históricas
- 🔮 Permite acción preventiva

### **3. Alertas Descriptivas:**
- 📊 Explica qué está pasando
- 📊 Indica qué tipo de problema es
- 📊 Proporciona contexto y tendencias

---

## 🔧 Integración

### **Con Sistema de Alertas:**

```python
from sistema_prediccion_problemas import SistemaPrediccionProblemas

sistema = SistemaPrediccionProblemas('modelo_multi_empresa.pkl')
sistema.iniciar()

# Procesar dato nuevo
resultado = sistema.analizar_dato(nuevo_dato)

# La alerta se genera automáticamente si:
# - Hay anomalía actual, O
# - Hay riesgo futuro
```

---

## 📊 Resumen

| Característica | Detección Actual | Predicción Futura |
|----------------|------------------|-------------------|
| **¿Qué detecta?** | Problemas que ya ocurren | Problemas que podrían ocurrir |
| **Basado en** | Datos actuales | Tendencias históricas |
| **Mensaje** | "🚨 HAY UNA ANOMALÍA" | "🔮 PREDICCIÓN: [riesgo]" |
| **Acción** | Resolver problema actual | Prevenir problema futuro |

---

## 🎯 Respuesta Directa

**SÍ, el sistema hace ambas cosas:**

1. **Detecta anomalías actuales:** "🚨 HAY UNA ANOMALÍA" cuando hay un problema ahora
2. **Predice problemas futuros:** "🔮 PREDICCIÓN: [riesgo]" analizando tendencias

**Ejecuta:**
```bash
python sistema_prediccion_problemas.py
```

**Y obtendrás:**
- ✅ Detección de problemas actuales
- 🔮 Predicción de problemas futuros
- 📊 Clasificación del tipo de problema
- 🚨 Alertas descriptivas con recomendaciones

**✅ Sistema completo de detección y predicción listo!**

