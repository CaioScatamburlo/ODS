import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("Referência: Aquecimento Ideal de Fluido (Sem Perdas)")
st.write("Este aplicativo demonstra o tempo de aquecimento de um fluido considerando apenas a potência da bomba, sem quaisquer perdas de calor para o ambiente.")

# --- Parâmetros Fixos para a Referência ---
st.header("Parâmetros do Cenário de Referência")

ref_pump_power_kw = 69.0  # Potência da bomba em kW
st.write(f"Potência de Aquecimento: **{ref_pump_power_kw} kW** (equivalente a uma bomba de 69kW)")

ref_total_volume_m3 = 10.0  # Volume de fluido em m³
st.write(f"Volume de Fluido: **{ref_total_volume_m3} m³**")

# Propriedades do fluido KRD MAX 225 (conforme o código principal)
ref_fluid_type = "Óleo KRD MAX 225"
ref_fluid_density = 850.0  # kg/m³
ref_fluid_cp = 2000.0  # J/(kg·K)
st.write(f"Fluido: **{ref_fluid_type}** (Densidade: {ref_fluid_density} kg/m³, Calor Específico: {ref_fluid_cp} J/(kg·K))")

ref_T_initial_C = 25.0  # Temperatura inicial em °C
st.write(f"Temperatura Inicial: **{ref_T_initial_C}°C**")

ref_T_target_C = 80.0  # Temperatura alvo em °C
st.write(f"Temperatura Alvo: **{ref_T_target_C}°C**")

# --- Cálculos de Aquecimento ---
st.header("Resultados do Aquecimento Ideal")

ref_power_watts = ref_pump_power_kw * 1000  # Converter kW para Watts
ref_fluid_mass_kg = ref_total_volume_m3 * ref_fluid_density  # Massa total do fluido em kg

# Taxa de aquecimento (dT/dt) = Potência / (massa * calor_específico)
ref_heating_rate_Cs = ref_power_watts / (ref_fluid_mass_kg * ref_fluid_cp)

# Variação total da temperatura
ref_delta_T = ref_T_target_C - ref_T_initial_C

# Tempo necessário para atingir a temperatura alvo
ref_time_seconds = ref_delta_T / ref_heating_rate_Cs
ref_time_minutes = ref_time_seconds / 60
ref_time_hours = ref_time_seconds / 3600

st.write(f"Massa do Fluido: **{ref_fluid_mass_kg:.2f} kg**")
st.write(f"Taxa de Aquecimento Teórica (dT/dt): **{ref_heating_rate_Cs:.4f} °C/s**")

st.markdown(f"**Tempo Estimado para Aquecer de {ref_T_initial_C}°C a {ref_T_target_C}°C:**")
st.write(f"- **{ref_time_seconds:.2f} segundos**")
st.write(f"- **{ref_time_minutes:.2f} minutos**")
st.write(f"- **{ref_time_hours:.2f} horas**")

# --- Geração da Curva de Aquecimento ---
st.header("Curva de Aquecimento Ideal (Sem Perdas)")

# Gerar pontos de tempo para o gráfico
# Estendemos um pouco o tempo total para ver a linearidade clara
time_points_s = np.linspace(0, ref_time_seconds * 1.1, 100)
temp_points_C = ref_T_initial_C + ref_heating_rate_Cs * time_points_s

fig_ref = go.Figure()
fig_ref.add_trace(go.Scatter(x=time_points_s / 3600, y=temp_points_C, mode='lines', 
                             name='Temperatura do Fluido', 
                             line=dict(color='green', width=3)))

# Adicionar linha da temperatura inicial
fig_ref.add_trace(go.Scatter(x=[0, time_points_s[-1] / 3600], y=[ref_T_initial_C, ref_T_initial_C], 
                             mode='lines', name=f'Temp. Inicial: {ref_T_initial_C}°C', 
                             line=dict(dash='dot', color='blue')))

# Adicionar linha da temperatura alvo
fig_ref.add_trace(go.Scatter(x=[0, time_points_s[-1] / 3600], y=[ref_T_target_C, ref_T_target_C], 
                             mode='lines', name=f'Temp. Alvo: {ref_T_target_C}°C', 
                             line=dict(dash='dot', color='red')))

fig_ref.update_layout(
    title="Curva de Aquecimento Ideal: Temperatura vs. Tempo",
    xaxis_title="Tempo (horas)",
    yaxis_title="Temperatura (°C)",
    hovermode="x unified" # Melhora a interatividade do gráfico
)
# Ajusta o range do Y para visualização
fig_ref.update_yaxes(range=[ref_T_initial_C - 5, ref_T_target_C + 5]) 

st.plotly_chart(fig_ref)

st.markdown("""
---
**Nota:** Este é um modelo de referência **simplificado**. Na prática, perdas de calor para o ambiente e variações nas propriedades do fluido afetariam a curva de aquecimento.
""")
