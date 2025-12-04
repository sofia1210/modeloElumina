from entrenar_multiples_empresas import MultiCompanyTrainer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Archivos de diferentes empresas
company_files = [
    '22_10_NEUROCIENCIAS_01012023-01012024.csv',
    '22_10_NEUROCIENCIAS_01012024-01012025.csv',
    '22_10_NEUROCIENCIAS_01012025-01122025.csv',
    '23_04_BIOPETROL_BEREA_01012023-31122024.csv',
    '23_04_BIOPETROL_BEREA_01012024-01012025.csv',
    '23_04_BIOPETROL_BEREA_01012025-01122025.csv',
    '23_04_Rod_Peel_01012023-01012024.csv',
    '23_04_Rod_Peel_01012024-01012025.csv',
    '23_04_Rod_Peel_01012025-01122025.csv',
    '23_04_SHOPPING_BOLIVAR_214_01012023-01012024.csv',
    '23_04_SHOPPING_BOLIVAR_214_01012024-01012025.csv',
    '23_06_UNION_AGRONEGOCIOS_01012023-31122024.csv',
    '23_06_UNION_AGRONEGOCIOS_01012024-01012025.csv',
    '23_06_UNION_AGRONEGOCIOS_01012025-01122025.csv',
    '23_12_BANCO_ECONOMICO_MUTUALISTA_01012023-01012024.csv',
    '23_12_BANCO_ECONOMICO_MUTUALISTA_01012024-01012025.csv',
    '23_12_BANCO_ECONOMICO_MUTUALISTA_01012025-01122025.csv',
    '24_02_ALVARO_AITKEN_01012024-01012025.csv',
    '24_02_ALVARO_AITKEN_01012025-01122025.csv',
    '24_04_MEGALABS_CEDIS_01012022-31122024.csv',
    '24_04_MEGALABS_CEDIS_01012024-01012025.csv',
    '24_04_AVICOLA_PAURITO_01012024-01012025.csv',
    '24_09_ALIANZA_01122024-01122025.csv',
    '24_09_FAMILIA_EDWIN_VARGAS_01012023-01012024.csv',
    '24_09_FAMILIA_EDWIN_VARGAS_01012024-01012025.csv',
    '25_08_MULTICENTER_01012025-01122025.csv',
    'EDUARDO_LOZADA_01012023-01012024.csv',
    'EDUARDO_LOZADA_01012024-01012025.csv',
    'EDUARDO_LOZADA_01012025-01122025.csv'
]

print("=" * 60)
print("🏢 ENTRENAMIENTO MULTI-EMPRESA CON GRÁFICOS")
print("=" * 60)

# Entrenar con normalización global
trainer = MultiCompanyTrainer(normalization_method='global')
detector = trainer.train_multi_company(
    company_files,
    contamination=0.10,
    save_model='modelo_multi_empresa.pkl'
)

if detector is not None:
    print("\n" + "=" * 60)
    print("📊 GENERANDO GRÁFICOS DE ANÁLISIS")
    print("=" * 60)
    
    # Cargar datos combinados para visualización
    print("\n📂 Cargando datos para visualización...")
    company_dfs = []
    for filepath in company_files:
        if os.path.exists(filepath):
            df = trainer.load_company_data(filepath)
            if df is not None:
                company_dfs.append(df)
    
    if company_dfs:
        # Combinar datos
        combined_df = trainer.combine_companies(company_dfs)
        normalized_df = trainer.normalize_data(combined_df)
        
        # Obtener predicciones
        print("\n🔍 Detectando anomalías en datos combinados...")
        results = detector.get_anomalies(normalized_df)
        
        # Guardar resultados
        results.to_csv('resultados_multi_empresa.csv', index=False, encoding='utf-8')
        print("✅ Resultados guardados en: resultados_multi_empresa.csv")
        
        # ========== GRÁFICO 1: Distribución por Empresa ==========
        print("\n📊 Generando gráfico 1: Distribución por Empresa...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Análisis Multi-Empresa: Distribución y Anomalías', fontsize=16, fontweight='bold')
        
        # Gráfico 1.1: Registros por empresa
        ax1 = axes[0, 0]
        company_counts = results['Empresa'].value_counts()
        ax1.barh(range(len(company_counts)), company_counts.values, color='steelblue')
        ax1.set_yticks(range(len(company_counts)))
        ax1.set_yticklabels(company_counts.index, fontsize=8)
        ax1.set_xlabel('Número de Registros')
        ax1.set_title('Registros por Empresa')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Gráfico 1.2: Anomalías por empresa
        ax2 = axes[0, 1]
        anomalies_by_company = results[results['Es_Anomalia'] == True]['Empresa'].value_counts()
        if len(anomalies_by_company) > 0:
            ax2.barh(range(len(anomalies_by_company)), anomalies_by_company.values, color='crimson')
            ax2.set_yticks(range(len(anomalies_by_company)))
            ax2.set_yticklabels(anomalies_by_company.index, fontsize=8)
            ax2.set_xlabel('Número de Anomalías')
            ax2.set_title('Anomalías Detectadas por Empresa')
            ax2.grid(True, alpha=0.3, axis='x')
        else:
            ax2.text(0.5, 0.5, 'No se detectaron anomalías', 
                   ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Anomalías Detectadas por Empresa')
        
        # Gráfico 1.3: Tasa de anomalías por empresa
        ax3 = axes[1, 0]
        anomaly_rates = []
        company_names = []
        for company in results['Empresa'].unique():
            company_data = results[results['Empresa'] == company]
            rate = company_data['Es_Anomalia'].mean() * 100
            anomaly_rates.append(rate)
            company_names.append(company)
        
        if len(anomaly_rates) > 0:
            colors = ['red' if r > 5 else 'orange' if r > 2 else 'green' for r in anomaly_rates]
            ax3.barh(range(len(company_names)), anomaly_rates, color=colors)
            ax3.set_yticks(range(len(company_names)))
            ax3.set_yticklabels(company_names, fontsize=8)
            ax3.set_xlabel('Tasa de Anomalías (%)')
            ax3.set_title('Tasa de Anomalías por Empresa')
            ax3.axvline(5, color='red', linestyle='--', alpha=0.5, label='5%')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='x')
        
        # Gráfico 1.4: Resumen general
        ax4 = axes[1, 1]
        total_normal = (results['Es_Anomalia'] == False).sum()
        total_anomalies = results['Es_Anomalia'].sum()
        total = len(results)
        
        ax4.pie([total_normal, total_anomalies], 
               labels=[f'Normal\n({total_normal})', f'Anomalías\n({total_anomalies})'],
               autopct='%1.1f%%', 
               colors=['green', 'red'],
               startangle=90)
        ax4.set_title(f'Resumen General\nTotal: {total} registros')
        
        plt.tight_layout()
        plt.savefig('grafico_distribucion_empresas.png', dpi=300, bbox_inches='tight')
        print("✅ Gráfico guardado: grafico_distribucion_empresas.png")
        plt.close()
        
        # ========== GRÁFICO 2: Generación vs Consumo Multi-Empresa ==========
        print("📊 Generando gráfico 2: Generación vs Consumo...")
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Colores diferentes por empresa
        unique_companies = results['Empresa'].unique()
        colors_map = plt.cm.tab20(np.linspace(0, 1, len(unique_companies)))
        company_colors = dict(zip(unique_companies, colors_map))
        
        # Plotear normales
        normal_data = results[results['Es_Anomalia'] == False]
        for company in unique_companies:
            company_normal = normal_data[normal_data['Empresa'] == company]
            if len(company_normal) > 0:
                ax.scatter(company_normal['Consumo total'], 
                          company_normal['Generación total'],
                          alpha=0.4, s=30, color=company_colors[company],
                          label=f'{company} (Normal)' if len(company_normal) > 0 else '')
        
        # Plotear anomalías
        anomalies_data = results[results['Es_Anomalia'] == True]
        if len(anomalies_data) > 0:
            for company in anomalies_data['Empresa'].unique():
                company_anomalies = anomalies_data[anomalies_data['Empresa'] == company]
                ax.scatter(company_anomalies['Consumo total'],
                          company_anomalies['Generación total'],
                          alpha=0.9, s=150, color='red', marker='x',
                          linewidths=2, label='Anomalías' if company == anomalies_data['Empresa'].iloc[0] else '')
        
        ax.set_xlabel('Consumo Total (Wh)', fontsize=12)
        ax.set_ylabel('Generación Total (Wh)', fontsize=12)
        ax.set_title('Generación vs Consumo - Todas las Empresas', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('grafico_generacion_consumo_multi_empresa.png', dpi=300, bbox_inches='tight')
        print("✅ Gráfico guardado: grafico_generacion_consumo_multi_empresa.png")
        plt.close()
        
        # ========== GRÁFICO 3: Scores de Anomalía ==========
        print("📊 Generando gráfico 3: Distribución de Scores...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histograma de scores
        ax1 = axes[0]
        ax1.hist(results['Score_Anomalia'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        anomaly_scores = results[results['Es_Anomalia'] == True]['Score_Anomalia']
        if len(anomaly_scores) > 0:
            ax1.axvline(anomaly_scores.min(), color='red', linestyle='--', linewidth=2, label='Umbral anomalías')
        ax1.set_xlabel('Score de Anomalía')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Scores de Anomalía')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot por empresa (top 10 empresas con más datos)
        ax2 = axes[1]
        top_companies = results['Empresa'].value_counts().head(10).index
        top_data = results[results['Empresa'].isin(top_companies)]
        
        if len(top_data) > 0:
            box_data = [top_data[top_data['Empresa'] == company]['Score_Anomalia'].values 
                       for company in top_companies]
            bp = ax2.boxplot(box_data, labels=top_companies, vert=True, patch_artist=True)
            
            # Colorear cajas
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax2.set_ylabel('Score de Anomalía')
            ax2.set_title('Distribución de Scores por Empresa (Top 10)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('grafico_scores_anomalia.png', dpi=300, bbox_inches='tight')
        print("✅ Gráfico guardado: grafico_scores_anomalia.png")
        plt.close()
        
        # ========== GRÁFICO 4: Resumen Estadístico ==========
        print("📊 Generando gráfico 4: Resumen Estadístico...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Resumen Estadístico Multi-Empresa', fontsize=16, fontweight='bold')
        
        # Estadísticas por empresa
        stats_data = []
        for company in results['Empresa'].unique():
            company_data = results[results['Empresa'] == company]
            stats_data.append({
                'Empresa': company,
                'Registros': len(company_data),
                'Anomalías': company_data['Es_Anomalia'].sum(),
                'Tasa (%)': company_data['Es_Anomalia'].mean() * 100,
                'Score Promedio': company_data['Score_Anomalia'].mean(),
                'Generación Promedio': company_data['Generación total'].mean() if 'Generación total' in company_data.columns else 0,
                'Consumo Promedio': company_data['Consumo total'].mean() if 'Consumo total' in company_data.columns else 0
            })
        
        stats_df = pd.DataFrame(stats_data).sort_values('Registros', ascending=False)
        
        # Gráfico 4.1: Top empresas por registros
        ax1 = axes[0, 0]
        top_10 = stats_df.head(10)
        ax1.barh(range(len(top_10)), top_10['Registros'].values, color='steelblue')
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels(top_10['Empresa'].values, fontsize=9)
        ax1.set_xlabel('Número de Registros')
        ax1.set_title('Top 10 Empresas por Cantidad de Datos')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Gráfico 4.2: Tasa de anomalías
        ax2 = axes[0, 1]
        top_10_anomalies = stats_df.nlargest(10, 'Anomalías')
        if len(top_10_anomalies) > 0:
            ax2.barh(range(len(top_10_anomalies)), top_10_anomalies['Tasa (%)'].values, color='crimson')
            ax2.set_yticks(range(len(top_10_anomalies)))
            ax2.set_yticklabels(top_10_anomalies['Empresa'].values, fontsize=9)
            ax2.set_xlabel('Tasa de Anomalías (%)')
            ax2.set_title('Top 10 Empresas con Más Anomalías')
            ax2.grid(True, alpha=0.3, axis='x')
        
        # Gráfico 4.3: Score promedio por empresa
        ax3 = axes[1, 0]
        top_10_scores = stats_df.nsmallest(10, 'Score Promedio')  # Más negativos = más anómalos
        ax3.barh(range(len(top_10_scores)), top_10_scores['Score Promedio'].values, color='orange')
        ax3.set_yticks(range(len(top_10_scores)))
        ax3.set_yticklabels(top_10_scores['Empresa'].values, fontsize=9)
        ax3.set_xlabel('Score Promedio (más negativo = más anómalo)')
        ax3.set_title('Top 10 Empresas con Scores Más Negativos')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Gráfico 4.4: Resumen numérico
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
        RESUMEN GENERAL
        
        Total Empresas: {len(stats_df)}
        Total Registros: {len(results):,}
        Anomalías Detectadas: {results['Es_Anomalia'].sum():,}
        Tasa Global: {results['Es_Anomalia'].mean()*100:.2f}%
        
        Empresa con más datos:
        {stats_df.iloc[0]['Empresa']}
        ({stats_df.iloc[0]['Registros']} registros)
        
        Empresa con más anomalías:
        {stats_df.nlargest(1, 'Anomalías').iloc[0]['Empresa']}
        ({stats_df.nlargest(1, 'Anomalías').iloc[0]['Anomalías']} anomalías)
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=11, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('grafico_resumen_estadistico.png', dpi=300, bbox_inches='tight')
        print("✅ Gráfico guardado: grafico_resumen_estadistico.png")
        plt.close()
        
        print("\n" + "=" * 60)
        print("✅ TODOS LOS GRÁFICOS GENERADOS")
        print("=" * 60)
        print("\n📊 Gráficos creados:")
        print("   1. grafico_distribucion_empresas.png")
        print("   2. grafico_generacion_consumo_multi_empresa.png")
        print("   3. grafico_scores_anomalia.png")
        print("   4. grafico_resumen_estadistico.png")
        print("\n💾 Resultados guardados en: resultados_multi_empresa.csv")
        print("\n✅ Proceso completado exitosamente!")
        
    else:
        print("⚠️ No se pudieron cargar datos para visualización")
else:
    print("❌ Error: No se pudo entrenar el modelo")