import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve # Importar fsolve aqui

# --- Constantes Físicas ---
SIGMA = 5.67e-8  # Constante de Stefan-Boltzmann [W/(m^2 * K^4)]

# --- Funções Auxiliares ---
def calculate_h_rad(emissivity, T_surface_K, T_ambient_K):
    """Calcula o coeficiente de transferência de calor por radiação."""
    T_surface_K = float(T_surface_K)
    T_ambient_K = float(T_ambient_K)
    
    if T_surface_K < 0 or T_ambient_K < 0:
        return 0.0
        
    return emissivity * SIGMA * (T_surface_K + T_ambient_K) * (T_surface_K**2 + T_ambient_K**2)

# Streamlit App Title
st.title("Pump Heat Simulation Tool")
st.write("Enter your pump, fluid, piping, and insulation parameters below:")

# === System Data ===
st.header("System Data")
total_volume_m3 = st.number_input("Total fluid volume in system (m³):", min_value=0.1, value=10.0)
T_ambient = st.number_input("Ambient temperature (°C):", value=25.0)
target_mu = st.number_input("Target Viscosity (cP):", value=25.0)
max_mu = target_mu * 1.1 / 1000 # Convert cP to Pa.s
min_mu = target_mu * 0.9 / 1000 # Convert cP to Pa.s

# === Fluid Data ===
st.header("Fluid Data")
use_manual_input = st.checkbox("Manually input fluid properties")

if use_manual_input:
    rho = st.number_input("Fluid density (kg/m³):", min_value=100.0, value=850.0)
    cp_fluid = st.number_input("Fluid specific heat capacity (J/kg·K):", min_value=0.1, value=2000.0)
    k_fluid = st.number_input("Fluid thermal conductivity (W/m·K):", min_value=0.01, value=0.12)
    mu_constant = st.number_input("Fluid dynamic viscosity (Pa·s):", min_value=0.001, value=0.3)
    viscosity_model = lambda Tf: mu_constant
    fluid_choice = "Manual Input" # Define fluid_choice even for manual input
else:
    fluid_choice = st.selectbox("Select fluid from library:", [
        "KRD MAX 225 (11.4 - 40.8 cP)",
        "KRD MAX 2205 (82.5 - 402 cP)",
        "KRD MAX 685 (68.2 - 115.6 cP)",
        "KRD MAX 55 (2.4 - 4.64 cP)",
        "Tellus S2 V32 (16,4 - 77.7 cP)",
        "Tellus S2 V100 (44.4 - 277.8 cP)",
        "Tellus S2 M32 (14 - 86.1 cP)",
        "Tellus S2 M100 (35.3 -  315 cP)" 
    ])

    # Fluid properties (common for library fluids)
    rho = 850.0  # kg/m³
    cp_fluid = 2000.0  # J/kg·K
    k_fluid = 0.12  # W/m·K

    # viscosity models by fluid (ensure they return Pa.s)
    if fluid_choice == "KRD MAX 225 (11.4 - 40.8 cP)":
        viscosity_model = lambda Tf: 0.1651 * np.exp(-0.046 * Tf)
    elif fluid_choice == "KRD MAX 2205 (82.5 - 402 cP)":
        viscosity_model = lambda Tf: 1.9133 * np.exp(-0.053 * Tf)
    elif fluid_choice == "KRD MAX 685 (68.2 - 115.6 cP":
        viscosity_model = lambda Tf: 0.5933 * np.exp(-0.054 * Tf)
    elif fluid_choice == "KRD MAX 55 (2.4 - 4.64 cP)":
        viscosity_model = lambda Tf: -9e-08 * Tf**3 + 1e-05 * Tf**2 - 0.0007 * Tf + 0.0165
    
    # NOVAS ADIÇÕES PARA OS ÓLEOS TELLUS S2
    elif fluid_choice == "Tellus S2 V32":
        viscosity_model = lambda Tf: 0.1661 * np.exp(-0.0456 * Tf)
    elif fluid_choice == "Tellus S2 V100":
        viscosity_model = lambda Tf: 0.8123 * np.exp(-0.0458 * Tf)
    elif fluid_choice == "Tellus S2 M32":
        viscosity_model = lambda Tf: 0.2241 * np.exp(-0.0385 * Tf)
    elif fluid_choice == "Tellus S2 M100":
        viscosity_model = lambda Tf: 1.0963 * np.exp(-0.0388 * Tf)


# === Pump Data ===
st.header("Pump Data")
pump_heat_factor = st.number_input(
    "Pump Heat Factor:",
    min_value=0.0,
    value=1.0,
    step=0.1,
    help="A multiplier applied to the pump's hydraulic power to calculate the heat added to the fluid."
)

st.header("Heating Phase Pump Config")
pump_power_kw = st.number_input("Nominal power per heating pump (kW):", min_value=0.1, value=69.0)
pump_hydraulic_power = st.number_input("Hydraulic Power per heating pump (kW):", min_value=0.1, value=40.0)
pump_flow_m3h = st.number_input("Flow rate per heating pump (m³/h):", min_value=0.1, value=550.0)
pump_eff = st.number_input("Pump efficiency (%) for heating:", min_value=1.0, max_value=100.0, value=58.0)
num_pumps = st.number_input("Number of heating pumps operating in parallel:", min_value=1, step=1, value=1)
pump_surface_area_m2 = st.number_input("Pump Surface Area (m²) for heat loss per pump:", min_value=0.0, value=1.5)


st.header("Calibration Phase Pump Config")
calib_pump_power_kw = st.number_input("Nominal power per calibration pump (kW):", min_value=0.1, value=69.0)
calib_pump_hydraulic_power = st.number_input("Hydraulic Power per calibration pump (kW):", min_value=0.1, value=40.0)
calib_pump_flow_m3h = st.number_input("Flow rate per calibration pump (m³/h):", min_value=0.1, value=550.0)
calib_pump_eff = st.number_input("Pump efficiency (%) for calibration:", min_value=1.0, max_value=100.0, value=58.0)
calib_num_pumps = st.number_input("Number of calibration pumps operating in parallel:", min_value=1, step=1, value=1)
calib_pump_surface_area_m2 = st.number_input("Calibration Pump Surface Area (m²) for heat loss per pump:", min_value=0.0, value=1.5)


# === Piping Data ===
st.header("Piping Data")
d = st.number_input("Inner pipe diameter (m):", min_value=0.01, value=0.25716)
D = st.number_input("Outer pipe diameter (m):", min_value=0.01, value=0.3238)
L = st.number_input("Pipe length (m):", min_value=1.0, value=40.0)

# Insulation option
use_insulation = st.checkbox("Use pipe insulation?", value=False)

insulation_thickness = 0.0
D_insul = D
k_insul = 1.0 

if use_insulation:
    insulation_thickness = st.number_input("Insulation thickness (m):", min_value=0.001, value=0.01)
    D_insul = D + 2 * insulation_thickness
    st.write(f"Outer diameter with insulation: {D_insul:.3f} m")
    k_insul = st.number_input("Insulation thermal conductivity (W/m·K):", min_value=0.01, value=0.04)

# === Tank Data ===
st.header("Tank Data")
num_tanks = st.number_input("Number of Tanks:", min_value=0, value=1, step=1)
tank_surface_area_m2_per_unit = 0.0
tank_wall_thickness_m = 0.001 
tank_k_material = 1.0 

if num_tanks > 0:
    tank_type = st.selectbox("Tank Geometry/Type:", ["Cylindrical (vertical)", "Manual Exposed Surface Area"])
    
    if tank_type == "Cylindrical (vertical)":
        tank_diameter_m = st.number_input("Tank Diameter (m):", min_value=0.1, value=2.0)
        tank_height_m = st.number_input("Tank Height (m):", min_value=0.1, value=2.0)
        
        # Calculate and display exposed surface area (side + top)
        tank_surface_area_m2_per_unit = (np.pi * tank_diameter_m * tank_height_m) + (np.pi * (tank_diameter_m / 2)**2)
        st.write(f"Calculated exposed surface area per tank: {tank_surface_area_m2_per_unit:.2f} m² (Side + Top)")
        
        # NOVO: Calcular e exibir o volume do tanque
        tank_volume_calculated_m3 = (np.pi * (tank_diameter_m / 2)**2) * tank_height_m
        st.write(f"Calculated volume per tank: {tank_volume_calculated_m3:.2f} m³")

    else: # Manual Surface Area
        tank_surface_area_m2_per_unit = st.number_input("Exposed Surface Area per Tank (m²):", min_value=0.1, value=10.0)

    tank_wall_thickness_m = st.number_input("Tank Wall Thickness (m):", min_value=0.001, value=0.005)
    tank_k_material = st.number_input("Tank Wall Thermal Conductivity (W/m·K):", min_value=0.1, value=50.0)
    
    st.markdown("*(Note: Tank insulation is not considered in this model.)*")


t_max_h = st.number_input("Total simulation time (h):", min_value=0.1, value=10.0)


# === Run Simulation ===
if st.button("Run Simulation"):
    # --- Power Validation ---
    error_margin = 0.02 # 2% error margin
    
    # Validation for Heating Phase
    if pump_eff > 0:
        expected_nominal_heating = (pump_hydraulic_power / (pump_eff / 100))
        if pump_power_kw > 0 and not (abs(expected_nominal_heating - pump_power_kw) / pump_power_kw <= error_margin):
            st.warning(f"**Warning (Heating Phase):** 'Nominal Power' ({pump_power_kw:.2f} kW) does not match (Hydraulic Power / Efficiency) = ({expected_nominal_heating:.2f} kW) within a 2% margin. "
                       "Please ensure 'Nominal Power' represents the **Pump Axle Power**. "
                       "The calculation will proceed using the provided Nominal Power.")
    else:
        st.warning("**Warning (Heating Phase):** Pump efficiency cannot be zero. Calculation may be inaccurate.")
    
    # Validation for Calibration Phase
    if calib_pump_eff > 0:
        expected_nominal_calib = (calib_pump_hydraulic_power / (calib_pump_eff / 100))
        if calib_pump_power_kw > 0 and not (abs(expected_nominal_calib - calib_pump_power_kw) / calib_pump_power_kw <= error_margin):
            st.warning(f"**Warning (Calibration Phase):** 'Nominal Power' ({calib_pump_power_kw:.2f} kW) does not match (Hydraulic Power / Efficiency) = ({expected_nominal_calib:.2f} kW) within a 2% margin. "
                       "Please ensure 'Nominal Power' represents the **Pump Axle Power**. "
                       "The calculation will proceed using the provided Nominal Power.")
    else:
        st.warning("**Warning (Calibration Phase):** Calibration pump efficiency cannot be zero. Calculation may be inaccurate.")
    
    # --- Start Simulation Calculations ---
    dWp_dt = pump_power_kw * 1000 * num_pumps  # W (Total heat generated by pump(s) from shaft power)
    F = (pump_flow_m3h / 3600) * num_pumps  # m³/s (Total flow rate)

    m = total_volume_m3 * rho  # kg (Total fluid mass)
    k_pipe = 45  # W/m.K (Thermal conductivity of pipe material)
    h_out_convection = 25  # W/m2.K (External convection coefficient, ONLY CONVECTION)
    emiss = 0.95 # Emissivity for painted steel pipe.
    
    n = 0.33 # Exponent for Nusselt correlation (Dittus-Boelter)

    T_ambient_K = T_ambient + 273.15

    # --- Euler Simulation Setup ---
    dt = 0.1 # Time step in seconds
    t_max = t_max_h * 3600 # Max simulation time in seconds
    time = np.arange(0, t_max + dt, dt) # Adjusted to ensure final point inclusion
    Tf = np.zeros_like(time)
    Tf[0] = T_ambient # Initial fluid temperature (in Celsius)

    # Calculate constant thermal resistances for the pipe once outside the loop
    R_cond_pipe = np.log(D / d) / (2 * np.pi * k_pipe * L)
    R_cond_insul = np.log(D_insul / D) / (2 * np.pi * k_insul * L) if use_insulation else 0
    outer_diameter_for_loss = D_insul if use_insulation else D # Diameter for external heat transfer area of pipe


    # --- Euler Simulation Loop ---
    for i in range(1, len(time)):
        current_T_C = Tf[i-1] # Fluid temperature in Celsius from previous step
        current_T_K = current_T_C + 273.15 # Convert to Kelvin for h_rad and energy balance

        # 1. Recalculate h_in (depends on fluid temperature via viscosity)
        mu_t = viscosity_model(current_T_C)
        if mu_t <= 0:
            st.warning(f"Viscosity became non-positive at T={current_T_C:.1f}°C. Stopping simulation.")
            Tf = Tf[:i]
            time = time[:i]
            break

        Re = (4 * F * rho) / (np.pi * d * mu_t)
        
        if k_fluid == 0:
            st.error("Fluid thermal conductivity (k_fluid) cannot be zero. Please check your fluid data.")
            st.stop()

        Pr = (mu_t * cp_fluid) / k_fluid
        Nu = 0.023 * Re**0.8 * Pr**n
        
        if d == 0 or Nu == 0:
            h_in = float('inf')
        else:
            h_in = Nu * k_fluid / d

        if h_in == 0 or d == 0 or L == 0:
            R_conv_in = float('inf')
        else:
            R_conv_in = 1 / (h_in * np.pi * d * L)

        # 2. Calculate h_rad and R_equiv_ext (for PIPE) dynamically
        h_rad_pipe = calculate_h_rad(emiss, current_T_K, T_ambient_K) # h_rad for pipe
        total_h_external_pipe = (h_out_convection + h_rad_pipe)
        
        if total_h_external_pipe == 0 or outer_diameter_for_loss == 0 or L == 0:
            R_equiv_ext_pipe = float('inf')
        else:
            R_equiv_ext_pipe = 1 / (total_h_external_pipe * np.pi * outer_diameter_for_loss * L)

        # 3. Calculate h_rad and R_equiv_ext (for PUMP) dynamically
        h_rad_pump = calculate_h_rad(emiss, current_T_K, T_ambient_K) # h_rad for pump
        total_h_external_pump = (h_out_convection + h_rad_pump)
        
        if total_h_external_pump == 0 or pump_surface_area_m2 == 0:
            R_equiv_ext_pump = float('inf')
        else:
            R_equiv_ext_pump = 1 / (total_h_external_pump * pump_surface_area_m2 * num_pumps)

        # 4. Calculate Tank Heat Loss Resistances (no insulation for tanks)
        conductance_tank = 0 # Default to 0 if no tanks
        if num_tanks > 0:
            # Tank Wall Conduction Resistance (simplified flat plate approx)
            if tank_k_material == 0 or tank_surface_area_m2_per_unit == 0:
                R_cond_tank_wall_approx = float('inf')
            else:
                R_cond_tank_wall_approx = tank_wall_thickness_m / (tank_k_material * tank_surface_area_m2_per_unit)
            
            R_cond_tank_insulation_approx = 0 # No insulation for tanks
            
            R_total_tank_conductive = R_cond_tank_wall_approx + R_cond_tank_insulation_approx

            # External Convection and Radiation for Tank
            h_rad_tank = calculate_h_rad(emiss, current_T_K, T_ambient_K)
            total_h_external_tank_combined = (h_out_convection + h_rad_tank)
            
            if total_h_external_tank_combined == 0 or tank_surface_area_m2_per_unit == 0:
                R_equiv_ext_tank = float('inf')
            else:
                R_equiv_ext_tank = 1 / (total_h_external_tank_combined * tank_surface_area_m2_per_unit * num_tanks)
            
            # Total resistance from fluid inside tank to ambient (assuming perfect internal mixing)
            R_total_tank_current_step = R_total_tank_conductive + R_equiv_ext_tank
            conductance_tank = 1 / R_total_tank_current_step if R_total_tank_current_step != float('inf') else 0

        # 5. Calculate Total Thermal Conductance for the current step (PIPE + PUMP + TANK in parallel)
        conductance_pipe = 1 / (R_conv_in + R_cond_pipe + R_cond_insul + R_equiv_ext_pipe)
        conductance_pump = 1 / R_equiv_ext_pump if R_equiv_ext_pump != float('inf') else 0
        
        total_conductance_loss = conductance_pipe + conductance_pump + conductance_tank
        
        if total_conductance_loss == 0:
            R_total_system_for_loss = float('inf')
        else:
            R_total_system_for_loss = 1 / total_conductance_loss

        if R_total_system_for_loss == 0:
            loss_term = float('inf') * np.sign(current_T_K - T_ambient_K)
        else:
            loss_term = (current_T_K - T_ambient_K) / R_total_system_for_loss

        # 6. Calculate dT/dt and update fluid temperature
        if m * cp_fluid == 0:
            dT_dt = 0
        else:
            dT_dt = (dWp_dt - loss_term) / (m * cp_fluid)
        
        Tf[i] = Tf[i-1] + dT_dt * dt

    # --- Calculations related to Viscosity-Temperature relationship ---
    T_90 = T_110 = T_target_visc = None
    
    if fluid_choice == "KRD MAX 225 (11.4 - 40.8 cP)":
        T_90 = -1 / 0.046 * np.log(min_mu / 0.1651)
        T_110 = -1 / 0.046 * np.log(max_mu / 0.1651)
        T_target_visc = -1 / 0.046 * np.log(target_mu / 0.1651)

    elif fluid_choice == "KRD MAX 2205 (82.5 - 402 cP)":
        T_90 = -1 / 0.053 * np.log(min_mu / 1.9133)
        T_110 = -1 / 0.053 * np.log(max_mu / 1.9133)
        T_target_visc = -1 / 0.053 * np.log(target_mu / 1.9133)

    elif fluid_choice == "KRD MAX 685 (68.2 - 115.6 cP":
        T_90 = -1 / 0.054 * np.log(min_mu / 0.5933)
        T_110 = -1 / 0.054 * np.log(max_mu / 0.5933)
        T_target_visc = -1 / 0.054 * np.log(target_mu / 0.5933)

    elif fluid_choice == "KRD MAX 55 (2.4 - 4.64 cP)":
        def inverse_viscosity(mu_target_PaS):
            func = lambda T: (-9e-08 * T**3 + 1e-05 * T**2 - 0.0007 * T + 0.0165) - mu_target_PaS
            try:
                guesses = [0, 25, 50, 75, 100]
                for x0_guess in guesses:
                    sol = fsolve(func, x0=x0_guess)
                    if len(sol) > 0 and np.isreal(sol[0]):
                        return sol[0]
                return np.nan
            except Exception:
                return np.nan

        T_90 = inverse_viscosity(min_mu)
        T_110 = inverse_viscosity(max_mu)
        T_target_visc = inverse_viscosity(target_mu)
    
    # NOVAS ADIÇÕES PARA O CÁLCULO INVERSO PARA TEMPERATURA (T) DADO VISCOSIDADE (mu_target_PaS)
    # T = -1/B * log(mu_target_PaS / A)
    elif fluid_choice == "Tellus S2 V32":
        A_v32, B_v32 = 0.1661, 0.0456
        T_90 = -1 / B_v32 * np.log(min_mu / A_v32)
        T_110 = -1 / B_v32 * np.log(max_mu / A_v32)
        T_target_visc = -1 / B_v32 * np.log(target_mu / A_v32)
    elif fluid_choice == "Tellus S2 V100":
        A_v100, B_v100 = 0.8123, 0.0458
        T_90 = -1 / B_v100 * np.log(min_mu / A_v100)
        T_110 = -1 / B_v100 * np.log(max_mu / A_v100)
        T_target_visc = -1 / B_v100 * np.log(target_mu / A_v100)
    elif fluid_choice == "Tellus S2 M32":
        A_m32, B_m32 = 0.2241, 0.0385
        T_90 = -1 / B_m32 * np.log(min_mu / A_m32)
        T_110 = -1 / B_m32 * np.log(max_mu / A_m32)
        T_target_visc = -1 / B_m32 * np.log(target_mu / A_m32)
    elif fluid_choice == "Tellus S2 M100":
        A_m100, B_m100 = 1.0963, 0.0388
        T_90 = -1 / B_m100 * np.log(min_mu / A_m100)
        T_110 = -1 / B_m100 * np.log(max_mu / A_m100)
        T_target_visc = -1 / B_m100 * np.log(target_mu / A_m100)
    else:
        pass

    # --- Find 110% time (Heating Phase) ---
    t_110_h = None
    T_110_actual = None
    if T_110 is not None and not np.isnan(T_110):
        idx_110 = np.where(Tf >= T_110)[0]
        if len(idx_110) > 0:
            t_110_h = time[idx_110[0]] / 3600
            T_110_actual = Tf[idx_110[0]]
        else:
            st.warning(f"Heating phase did not reach 110% viscosity target temperature ({T_110:.1f}°C) within simulation time.")

    # --- Calibration Phase Simulation ---
    if t_110_h is not None:
        idx_110_heating = np.where(time <= t_110_h * 3600)[0]
        time_heating_truncated = time[idx_110_heating]
        Tf_heating_truncated = Tf[idx_110_heating]

        time_calib = np.arange(t_110_h * 3600, t_max + dt, dt)
        if len(time_calib) > 0:
            Tf_calib = np.zeros_like(time_calib)
            Tf_calib[0] = T_110_actual
        else:
            Tf_calib = np.array([])
            time_calib = np.array([])
            st.warning("Calibration phase has no time steps available after heating phase ends.")

        dWp_dt_calib = calib_pump_power_kw * 1000 * calib_num_pumps
        F_calib = (calib_pump_flow_m3h / 3600) * calib_num_pumps

        for i in range(1, len(time_calib)):
            current_T_C_calib = Tf_calib[i-1]
            current_T_K_calib = current_T_C_calib + 273.15

            mu_t_calib = viscosity_model(current_T_C_calib)
            if mu_t_calib <= 0:
                st.warning(f"Viscosity became non-positive during calibration at T={current_T_C_calib:.1f}°C. Stopping simulation.")
                Tf_calib = Tf_calib[:i]
                time_calib = time_calib[:i]
                break

            Re_calib = (4 * F_calib * rho) / (np.pi * d * mu_t_calib)
            Pr_calib = (mu_t_calib * cp_fluid) / k_fluid
            Nu_calib = 0.023 * Re_calib**0.8 * Pr_calib**n
            
            if d == 0 or Nu_calib == 0:
                h_in_calib = float('inf')
            else:
                h_in_calib = Nu_calib * k_fluid / d
            
            if h_in_calib == 0 or d == 0 or L == 0:
                R_conv_in_calib = float('inf')
            else:
                R_conv_in_calib = 1 / (h_in_calib * np.pi * d * L)

            h_rad_pipe_calib = calculate_h_rad(emiss, current_T_K_calib, T_ambient_K)
            total_h_external_pipe_calib = (h_out_convection + h_rad_pipe_calib)
            
            if total_h_external_pipe_calib == 0 or outer_diameter_for_loss == 0 or L == 0:
                R_equiv_ext_pipe_calib = float('inf')
            else:
                R_equiv_ext_pipe_calib = 1 / (total_h_external_pipe_calib * np.pi * outer_diameter_for_loss * L)

            h_rad_pump_calib = calculate_h_rad(emiss, current_T_K_calib, T_ambient_K)
            total_h_external_pump_calib = (h_out_convection + h_rad_pump_calib)
            
            if total_h_external_pump_calib == 0 or calib_pump_surface_area_m2 == 0:
                R_equiv_ext_pump_calib = float('inf')
            else:
                R_equiv_ext_pump_calib = 1 / (total_h_external_pump_calib * calib_pump_surface_area_m2 * calib_num_pumps)

            # Calculate Tank Heat Loss Resistances (no insulation for tanks) for calibration phase
            conductance_tank_calib = 0 # Default if no tanks
            if num_tanks > 0: # Ensure there are tanks to calculate loss from
                if tank_k_material == 0 or tank_surface_area_m2_per_unit == 0:
                    R_cond_tank_wall_approx_calib = float('inf')
                else:
                    R_cond_tank_wall_approx_calib = tank_wall_thickness_m / (tank_k_material * tank_surface_area_m2_per_unit)
                
                R_cond_tank_insulation_approx_calib = 0 # No insulation for tanks
                
                R_total_tank_conductive_calib = R_cond_tank_wall_approx_calib + R_cond_tank_insulation_approx_calib

                h_rad_tank_calib = calculate_h_rad(emiss, current_T_K_calib, T_ambient_K)
                total_h_external_tank_combined_calib = (h_out_convection + h_rad_tank_calib)
                
                if total_h_external_tank_combined_calib == 0 or tank_surface_area_m2_per_unit == 0:
                    R_equiv_ext_tank_calib = float('inf')
                else:
                    R_equiv_ext_tank_calib = 1 / (total_h_external_tank_combined_calib * tank_surface_area_m2_per_unit * num_tanks)
                
                R_total_tank_calib_current_step = R_total_tank_conductive_calib + R_equiv_ext_tank_calib
                conductance_tank_calib = 1 / R_total_tank_calib_current_step if R_total_tank_calib_current_step != float('inf') else 0

            conductance_pipe_calib = 1 / (R_conv_in_calib + R_cond_pipe + R_cond_insul + R_equiv_ext_pipe_calib)
            conductance_pump_calib = 1 / R_equiv_ext_pump_calib if R_equiv_ext_pump_calib != float('inf') else 0

            total_conductance_loss_calib = conductance_pipe_calib + conductance_pump_calib + conductance_tank_calib
            
            if total_conductance_loss_calib == 0:
                R_total_system_for_loss_calib = float('inf')
            else:
                R_total_system_for_loss_calib = 1 / total_conductance_loss_calib

            if R_total_system_for_loss_calib == 0:
                loss_term_calib = float('inf') * np.sign(current_T_K_calib - T_ambient_K)
            else:
                loss_term_calib = (current_T_K_calib - T_ambient_K) / R_total_system_for_loss_calib

            if m * cp_fluid == 0:
                dT_dt_calib = 0
            else:
                dT_dt_calib = (dWp_dt_calib - loss_term_calib) / (m * cp_fluid)
            
            Tf_calib[i] = Tf_calib[i-1] + dT_dt_calib * dt

        idx_90_calib = np.where(Tf_calib >= T_90)[0] if T_90 is not None and not np.isnan(T_90) else []
        if len(idx_90_calib) > 0:
            t_90_h = time_calib[idx_90_calib[0]] / 3600
            T_90_actual = Tf_calib[idx_90_calib[0]]
        else:
            t_90_h = None
            T_90_actual = None
            st.warning(f"Calibration phase did not reach 90% viscosity target temperature ({T_90:.1f}°C) within simulation time.")

        if R_total_system_for_loss_calib != 0:
            T_eq = T_ambient + dWp_dt_calib * R_total_system_for_loss_calib
        else:
            T_eq = T_ambient
            st.warning("Resistance for equilibrium temperature calculation is zero or infinite. Check parameters.")
        
        t_110_hours = int(t_110_h) if t_110_h is not None else 0
        t_110_minutes = int((t_110_h - t_110_hours) * 60) if t_110_h is not None else 0

        st.write(f"Calibration Phase starting after {t_110_hours:.0f}h{t_110_minutes:.0f}min at temperature {T_110_actual:.1f}°C")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔥 Heating Phase Configuration")
            st.write(f"💧 **Total Flow Rate**: {pump_flow_m3h * num_pumps:.2f} m³/h")
            st.write(f"🔋 **Number of Pumps**: {num_pumps}")
            st.write(f"⚡ **Total Power**: {pump_power_kw * num_pumps:.2f} kW")

        with col2:
            st.markdown("### 🧪 Calibration Phase Configuration")
            st.write(f"💧 **Total Flow Rate**: {calib_pump_flow_m3h * calib_num_pumps:.2f} m³/h")
            st.write(f"🔋 **Number of Pumps**: {calib_num_pumps}")
            st.write(f"⚡ **Total Power**: {calib_pump_power_kw * calib_num_pumps:.2f} kW")

        st.write(f"### System Info")
        st.write(f"🛢️ **Selected Fluid**: {fluid_choice}")
        st.write(f"📦 **Total Fluid Volume**: {total_volume_m3} m³")
        st.write(f"🎯 **Target Viscosity**: {target_mu:.2f} cP")
        
        if t_110_h is not None:
             st.write(f"⏱️ **Heating time**: {t_110_hours} h {t_110_minutes} min")
        else:
             st.write("⏱️ **Heating time**: Not reached 110% viscosity target.")

        if t_90_h is not None and t_110_h is not None:
            calibration_time_h = t_90_h - t_110_h
            hours = int(calibration_time_h)
            minutes = int((calibration_time_h - hours) * 60)
            st.write(f"📏 **Available Calibration Time Window**: {hours} h {minutes} min")
        else:
            st.write("📏 **Available Calibration Time Window**: Not available (target not reached or 110% not defined).")
        
        # Create plot of Temperature over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_heating_truncated/3600, y=Tf_heating_truncated, mode='lines', name='Heating Phase', line=dict(color='red')))
        
        if len(time_calib) > 0:
            fig.add_trace(go.Scatter(x=time_calib/3600, y=Tf_calib, mode='lines', name='Calibration Phase', line=dict(color='blue')))
        
        fig.update_layout(title="Temperature vs Time", xaxis_title="Time (hours)", yaxis_title="Temperature (°C)")

        if T_eq is not None:
            fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_eq, T_eq], mode='lines',
                                     name=f'Equilibrium Temp: {T_eq:.1f} °C',
                                     line=dict(color='red', dash='dash')))

        if T_90 is not None and not np.isnan(T_90):
            fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_90, T_90], mode='lines',
                                     name=f'90% Viscosity Temp: {T_90:.1f} °C',
                                     line=dict(color='green', dash='dot')))

            if t_90_h is not None:
                fig.add_trace(go.Scatter(x=[t_90_h, t_90_h], y=[Tf.min() - 5, Tf.max() + 5], mode='lines',
                                         name=f'Time to reach 90% Viscosity ≈ {t_90_h:.2f} h',
                                         line=dict(color='green', dash='dot')))

                if T_90_actual is not None:
                    fig.add_trace(go.Scatter(x=[t_90_h], y=[T_90_actual], mode='markers',
                                             marker=dict(color='green', size=7),
                                             name='90% Viscosity Point'))

        if T_110 is not None and not np.isnan(T_110):
            fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_110, T_110], mode='lines',
                                     name=f'110% Viscosity Temp: {T_110:.1f} °C',
                                     line=dict(color='purple', dash='dot')))

            if t_110_h is not None:
                fig.add_trace(go.Scatter(x=[t_110_h, t_110_h], y=[Tf.min() - 5, Tf.max() + 5], mode='lines',
                                         name=f'Time to reach 110% Viscosity ≈ {t_110_h:.2f} h',
                                         line=dict(color='purple', dash='dot')))

                if T_110_actual is not None:
                    fig.add_trace(go.Scatter(x=[t_110_h], y=[T_110_actual], mode='markers',
                                             marker=dict(color='purple', size=7),
                                             name='110% Viscosity Point'))

        st.plotly_chart(fig)
    else:
        st.warning("The heating phase did not reach the 110% viscosity target temperature within the specified simulation time. Calibration phase could not be started.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time/3600, y=Tf, mode='lines', name='Heating Phase', line=dict(color='red')))
        fig.update_layout(title="Temperature vs Time (Heating Phase Only)", xaxis_title="Time (hours)", yaxis_title="Temperature (°C)")
        
        if T_90 is not None and not np.isnan(T_90):
            fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_90, T_90], mode='lines',
                                     name=f'90% Viscosity Temp: {T_90:.1f} °C',
                                     line=dict(color='green', dash='dot')))
        if T_110 is not None and not np.isnan(T_110):
            fig.add_trace(go.Scatter(x=[0, t_max_h], y=[T_110, T_110], mode='lines',
                                     name=f'110% Viscosity Temp: {T_110:.1f} °C',
                                     line=dict(color='purple', dash='dot')))
        
        st.plotly_chart(fig)
