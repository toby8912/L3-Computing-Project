#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G, h, m_e, pi, c, hbar
from scipy.optimize import curve_fit
solar_mass = 1.98847e30
m_n = 1.660539e-27 #average nucleon mass for carbon 12
m_neutron = 1.67492749804e-27 #neutron mass
Z_over_A = 0.5     # for carbon-12 
K_non_rel = ((h/(2*pi))**2 / (15* pi**2 * m_e)) * ((3*pi**2 * Z_over_A) / (m_n * c**2))**(5/3)
K_rel = (h*c/(24*pi**3))*((3*pi**2 * Z_over_A)/(m_n * c**2))**(4/3)
K_non_rel_neutron = ((h/(2*pi))**2 / (15* pi**2 * m_neutron)) * ((3*pi**2) / (m_neutron * c**2))**(5/3)
K_rel_neutron = 1/3


#Milestone functions
def white_dwarf_structure_non_rel(p_central):
    '''
    Solve white dwarf structure using the ODEs:
    dp/dr = -G*M*rho/r²
    dM/dr = 4*π*r²*rho
    
    Parameters:
    p_central: Central pressure in Pa
    dense_output: If True, return full solution object; if False, return [R, M, p]
    '''
    
    def stellar_structure_equations(r, y):
        '''
        Stellar structure equations:
        y[0] = p (pressure)
        y[1] = M (mass enclosed)
        '''
        p, M = y
        
        # Equation of state: rho = (p/K)^(3/5)
        if p > 0:
            rho = 1/c**2 * (p/K_non_rel)**(3/5)
        else:
            rho = 0
            return [0, 0]  # Stop if pressure becomes zero or negative
        
        # Stellar structure equations
        dpdr = -G * M * rho / r**2  # Hydrostatic equilibrium
        dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Integration parameters (optimized for typical white dwarf sizes)
    r_start = 1.0     # meters
    r_max = 2e7       # 20,000 km (more realistic upper bound)
    
    # Initial mass within r_start
    rho_central = 1/c**2 *(p_central / K_non_rel)**(3/5)
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    '''
    print(f"Central density: {rho_central:.2e} kg/m³")
    print(f"\nInitial conditions:")
    print(f"Central pressure: {p_central:.2e} Pa")
    print(f"Central density: {rho_central:.2e} kg/m³") 
    print(f"Starting radius: {r_start} m")
    print(f"Starting mass: {M_start:.3e} kg ({M_start/solar_mass:.6f} M☉)")
    '''
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0] - 1e5  # Stop when pressure drops to 10^5 Pa (near vacuum)
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve the ODEs
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',          # Good balance of speed and accuracy
        dense_output=True,     # Disabled - saves memory and computation
        rtol=1e-6,             # Relaxed tolerance for speed
        atol=1e-10,            # Sufficient precision for stellar structure
        max_step=1000           # Larger steps for faster integration
    )
 
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Surface found via event
        R_surface = sol.t_events[0][0]
        M_total = sol.y_events[0][0][1]
        p_surface = sol.y_events[0][0][0]
        #print(f"\n Surface found!")
        #print(f"Radius: {R_surface/1000:.1f} km")
        #print(f"Total mass: {M_total/solar_mass:.4f} M☉")
        #print(f"Surface pressure: {p_surface:.2e} Pa")
        return [R_surface, M_total, p_surface]

    else:
        # Integration went to maximum radius
        if len(sol.t) > 0:
            R_final = sol.t[-1]
            M_final = sol.y[1, -1]
            p_final = sol.y[0, -1]            
            #print(f"Final radius: {R_final/1000:.1f} km")
            #print(f"Final mass: {M_final/solar_mass:.4f} M☉")
            #print(f"Final pressure: {p_final:.2e} Pa")
            print(f"\nReached maximum radius")
            return [R_final, M_final, p_final]
        else:
            print("Integration failed!")
            return None, None, None

    #print(f'Radius: {white_dwarf_structure(p_central)[0]/1000:.1f} km, Mass: {white_dwarf_structure(p_central)[1]/solar_mass:.4f} M☉')

def white_dwarf_structure_rel(p_central):
    
    #print(f"Polytropic constant K = {K_rel:.3e} Pa⋅m^5⋅kg^(-4/3)")

    def stellar_structure_equations(r, y):
        p, M = y
        
        # Equation of state: rho = (p/K)^(3/4)
        if p > 0:
            rho = 1/c**2 * (p/K_rel)**(3/4)
        else:
            rho = 0
            return [0, 0]  # Stop if pressure becomes zero or negative
        
        # Stellar structure equations
        dpdr = -G * M * rho / r**2  # Hydrostatic equilibrium
        dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Integration parameters (optimized for typical white dwarf sizes)
    r_start = 0.001     # meters
    r_max = 2e7       # 20,000 km (more realistic upper bound)
    
    # Initial mass within r_start
    rho_central = 1/c**2 *(p_central / K_rel)**(3/4)
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0]  # Stop when pressure drops to 0 Pa (near vacuum)
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve the ODEs
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',          # Good balance of speed and accuracy
        dense_output=False,     # Disabled - saves memory and computation
        rtol=1e-6,             # Relaxed tolerance for speed
        atol=1e-10,            # Sufficient precision for stellar structure
        max_step=1000           # Larger steps for faster integration
    )
 
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Surface found via event
        R_surface = sol.t_events[0][0]
        M_total = sol.y_events[0][0][1]
        p_surface = sol.y_events[0][0][0]
        #print(f"\n Surface found!")
        #print(f"Radius: {R_surface/1000:.1f} km")
        #print(f"Total mass: {M_total/solar_mass:.4f} M☉")
        #print(f"Surface pressure: {p_surface:.2e} Pa")
        return [R_surface, M_total, p_surface]

    else:
        # Integration went to maximum radius
        if len(sol.t) > 0:
            R_final = sol.t[-1]
            M_final = sol.y[1, -1]
            p_final = sol.y[0, -1]            
            #print(f"Final radius: {R_final/1000:.1f} km")
            #print(f"Final mass: {M_final/solar_mass:.4f} M☉")
            #print(f"Final pressure: {p_final:.2e} Pa")
            print(f"\nReached maximum radius")
            return [R_final, M_final, p_final]
        else:
            print("Integration failed!")
            return None, None, None

def graph_mass_radius():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    p_central_values_non_rel = np.logspace(19, 24, num=50)  # Central pressures from 10^19 to 10^24 Pa
    radii_non_rel = []
    masses_non_rel = []
    for p_central in p_central_values_non_rel:
        R, M, _ = white_dwarf_structure_non_rel(p_central)
        radii_non_rel.append(R / 1000)  # Convert to km
        masses_non_rel.append(M / solar_mass)  # Convert to solar masses

    p_central_values_rel = 1e30  # Central pressures from 10^23 to 10^28 Pa
    radii_rel = np.linspace(0, 20000, num=50)
    R, M, _ = white_dwarf_structure_rel(p_central_values_rel)
    masses_rel = [M/ solar_mass]*50
    # force y-axis to start at 0 with no extra lower margin
    
    
    masses_non_rel = np.array(masses_non_rel)
    radii_non_rel = np.array(radii_non_rel)
    sort_indices = np.argsort(masses_non_rel)
    masses_non_rel = masses_non_rel[sort_indices]
    radii_non_rel = radii_non_rel[sort_indices]
    
    # Define realistic carbon white dwarf mass range
    min_carbon_mass = 0.5  # Solar masses
    max_carbon_mass = 1.4  # Solar masses
    
    
    min_idx = np.argmax(masses_non_rel >= min_carbon_mass)
    max_idx = np.argmax(masses_non_rel > max_carbon_mass)
    if max_idx == 0:  # If no masses exceed max_carbon_mass
        max_idx = len(masses_non_rel)
    if min_idx > 0:
        low_end = min(min_idx + 1, len(masses_non_rel))
        plt.plot(masses_non_rel[0:low_end], radii_non_rel[0:low_end], 
                color='lightskyblue', linestyle='--', alpha=0.7)
    if max_idx > min_idx:
        plt.plot(masses_non_rel[min_idx:max_idx], radii_non_rel[min_idx:max_idx], 
                marker='', label='Non-relativistic EOS (Typical Observed Range)', color='lightskyblue', linestyle='-')
    if max_idx < len(masses_non_rel):
        high_start = max(max_idx - 1, 0)
        plt.plot(masses_non_rel[high_start:], radii_non_rel[high_start:], 
                color='lightskyblue', linestyle='--', alpha=0.7)
    
    plt.plot([], [], color='lightskyblue', linestyle='--', alpha=0.7, label='Non-relativistic EOS (Unphysical)')
    plt.plot(masses_rel, radii_rel, marker='', label='Relativistic EOS', color='indianred')
    
    
    # Apply a dark theme: black background, white axes/labels/ticks/legend
    fig = plt.gcf()
    fig.patch.set_facecolor('black')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'legend.facecolor': 'black',
        'legend.edgecolor': 'white'
    })
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.tick_params(colors='white', which='both')
    for spine in ax.spines.values():
        spine.set_color('white')
        
        
    chandrasekhar_mass = M / solar_mass 
    #plt.text(chandrasekhar_mass, -110, 'CM', ha='center', va='top', fontsize=7)
    
    plt.xlabel('Mass (M☉)', fontsize=14)
    plt.ylabel('Radius (km)', fontsize=14)
    plt.title('White Dwarf Mass-Radius Relation', fontsize=16)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    print(f'Chandrasekhar Mass Limit : {chandrasekhar_mass}M☉')  # in solar masses

def plot_mass_pressure_vs_radius(p_central):
    """
    Plot both mass enclosed and pressure against radius for a single white dwarf star
    Both lines on the same plot with dual y-axes
    
    Parameters:
    p_central: Central pressure in Pa
    """
    import matplotlib.pyplot as plt
    
    def stellar_structure_equations(r, y):
        """Stellar structure equations"""
        p, M = y
        
        if p > 0:
            rho = 1/c**2 * (p/K_non_rel)**(3/5)
        else:
            return [0, 0]
        
        dpdr = -G * M * rho / r**2
        dMdr = 4 * pi * r**2 * rho
        
        return [dpdr, dMdr]
    
    # Integration setup
    r_start = 1.0
    r_max = 2e7
    
    rho_central = 1/c**2 * (p_central / K_non_rel)**(3/5)
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function to stop at surface
    def surface_condition(r, y):
        return y[0]
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve with dense output
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',
        dense_output=True,
        rtol=1e-6,
        atol=1e-10,
        max_step=1000
    )
    
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Get detailed radial structure
        r_surface = sol.t_events[0][0]
        r_detailed = np.linspace(r_start, r_surface, 200)
        y_detailed = sol.sol(r_detailed)
        
        # Convert to convenient units
        radii_km = r_detailed / 1000  # Convert to km
        masses_enclosed = y_detailed[1] / solar_mass  # Convert to solar masses
        pressures = y_detailed[0] / 1e22  # Convert to units of 10^22 Pa for better display
        
        # Create plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot pressure on left y-axis
        
        ax1.set_xlabel('Radius (km)', fontsize=14)
        ax1.set_ylabel('Pressure (10²² Pa)', fontsize=14)
        line1 = ax1.plot(radii_km, pressures, color='indianred', linewidth=2, label='Pressure')
        ax1.tick_params(axis='y', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        # Remove scientific notation from tick labels
        ax1.ticklabel_format(style='plain', axis='y')
        
        # Create second y-axis for mass enclosed
        ax2 = ax1.twinx()
        ax2.set_ylabel('Mass Enclosed (M☉)', fontsize=14)
        line2 = ax2.plot(radii_km, masses_enclosed, color='lightskyblue', linewidth=2, label='Mass Enclosed')
        ax2.tick_params(axis='y', labelsize=12)
        ax2.set_ylim(bottom=0)
        
        fig = plt.gcf()
    fig.patch.set_facecolor('black')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'legend.facecolor': 'black',
        'legend.edgecolor': 'white'
    })
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.tick_params(colors='white', which='both')
    for spine in ax.spines.values():
        spine.set_color('white')
        
        
        # Add title and legend
        ax1.set_title(f'White Dwarf: Mass Enclosed and Pressure against Radius', fontsize=16)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        # Print final values
        final_mass = masses_enclosed[-1]
        final_radius = radii_km[-1]
        final_pressure = pressures[-1] * 1e22  # Convert back to Pa for display
        print(f"Final Results:")
        print(f"  Total Mass: {final_mass:.3f} M☉")
        print(f"  Total Radius: {final_radius:.1f} km")
        print(f"  Surface Pressure: {final_pressure:.2e} Pa")
        
        #return radii_km, masses_enclosed, pressures * 1e22 
        
    else:
        print("Integration failed to find surface")
        return None, None, None

#TOV GR correction functions

def white_dwarf_structure_non_rel_TOV(p_central):
    
    def stellar_structure_equations(r, y):
        
        p, M = y
        
        # Equation of state: rho = (p/K)^(3/5)
        if p > 0:
            rho = 1/c**2 * (p/K_non_rel)**(3/5)
        else:
            rho = 0
            return [0, 0]  # Stop if pressure becomes zero or negative

        epsilson = rho * c**2  # Energy density
    
        # Stellar structure equations
        dpdr = -(G * M * rho / r**2)*(1+p/epsilson)*(1+(4*pi*(r**3)*p)/(M*(c**2)))*(1-(2*G*M)/((c**2)*r))**(-1)  # TOV
        dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Integration parameters (optimized for typical white dwarf sizes)
    r_start = 1.0     # meters
    r_max = 2e7       # 20,000 km (more realistic upper bound)
    
    # Initial mass within r_start
    rho_central = 1/c**2 *(p_central / K_non_rel)**(3/5)
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0] - 1e5  # Stop when pressure drops to 10^5 Pa (near vacuum)
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve the ODEs
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',          # Good balance of speed and accuracy
        dense_output=True,     # Disabled - saves memory and computation
        rtol=1e-6,             # Relaxed tolerance for speed
        atol=1e-10,            # Sufficient precision for stellar structure
        max_step=1000           # Larger steps for faster integration
    )
 
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Surface found via event
        R_surface = sol.t_events[0][0]
        M_total = sol.y_events[0][0][1]
        p_surface = sol.y_events[0][0][0]
        #print(f"\n Surface found!")
        #print(f"Radius: {R_surface/1000:.1f} km")
        #print(f"Total mass: {M_total/solar_mass:.4f} M☉")
        #print(f"Surface pressure: {p_surface:.2e} Pa")
        return [R_surface, M_total, p_surface]

    else:
        # Integration went to maximum radius
        if len(sol.t) > 0:
            R_final = sol.t[-1]
            M_final = sol.y[1, -1]
            p_final = sol.y[0, -1]            
            #print(f"Final radius: {R_final/1000:.1f} km")
            #print(f"Final mass: {M_final/solar_mass:.4f} M☉")
            #print(f"Final pressure: {p_final:.2e} Pa")
            print(f"\nReached maximum radius")
            return [R_final, M_final, p_final]
        else:
            print("Integration failed!")
            return None, None, None

    #print(f'Radius: {white_dwarf_structure(p_central)[0]/1000:.1f} km, Mass: {white_dwarf_structure(p_central)[1]/solar_mass:.4f} M☉')

def white_dwarf_structure_rel_TOV(p_central):
    
    #print(f"Polytropic constant K = {K_rel:.3e} Pa⋅m^5⋅kg^(-4/3)")

    def stellar_structure_equations(r, y):
        p, M = y
        
        # Equation of state: rho = (p/K)^(3/4)
        if p > 0:
            rho = 1/c**2 * (p/K_rel)**(3/4)
        else:
            rho = 0
            return [0, 0]  # Stop if pressure becomes zero or negative
        
        epsilson = rho * c**2  # Energy density
        
        # Stellar structure equations
        dpdr = -(G * M * rho / r**2)*(1+p/epsilson)*(1+(4*pi*(r**3)*p)/(M*(c**2)))*1/(1-(2*G*M)/((c**2)*r))  # TOV
        dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Integration parameters (optimized for typical white dwarf sizes)
    r_start = 0.001     # meters
    r_max = 2e7       # 20,000 km (more realistic upper bound)
    
    # Initial mass within r_start
    rho_central = 1/c**2 *(p_central / K_rel)**(3/4)
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0]  # Stop when pressure drops to 0 Pa (near vacuum)
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve the ODEs
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',          # Good balance of speed and accuracy
        dense_output=False,     # Disabled - saves memory and computation
        rtol=1e-6,             # Relaxed tolerance for speed
        atol=1e-10,            # Sufficient precision for stellar structure
        max_step=500           # Larger steps for faster integration
    )
 
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Surface found via event
        R_surface = sol.t_events[0][0]
        M_total = sol.y_events[0][0][1]
        p_surface = sol.y_events[0][0][0]
        #print(f"\n Surface found!")
        #print(f"Radius: {R_surface/1000:.1f} km")
        #print(f"Total mass: {M_total/solar_mass:.4f} M☉")
        #print(f"Surface pressure: {p_surface:.2e} Pa")
        return [R_surface, M_total, p_surface]

    else:
        # Integration went to maximum radius
        if len(sol.t) > 0:
            R_final = sol.t[-1]
            M_final = sol.y[1, -1]
            p_final = sol.y[0, -1]            
            #print(f"Final radius: {R_final/1000:.1f} km")
            #print(f"Final mass: {M_final/solar_mass:.4f} M☉")
            #print(f"Final pressure: {p_final:.2e} Pa")
            print(f"\nReached maximum radius")
            return [R_final, M_final, p_final]
        else:
            print("Integration failed!")
            return None, None, None

def graph_mass_radius_TOV():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    p_central_values_non_rel = np.logspace(19, 24, num=50)#50  # Central pressures from 10^19 to 10^24 Pa
    radii_non_rel = []
    masses_non_rel = []
    for p_central in p_central_values_non_rel:
        R, M, _ = white_dwarf_structure_non_rel_TOV(p_central)
        radii_non_rel.append(R / 1000)  # Convert to km
        masses_non_rel.append(M / solar_mass)  # Convert to solar masses

    p_central_values_rel = np.logspace(22.7, 36, num=50)  # Central pressures from 10^23 to 10^28 Pa
    radii_rel = []
    masses_rel = []
    for p_central in p_central_values_rel:
        R, M, _ = white_dwarf_structure_rel_TOV(p_central)
        radii_rel.append(R / 1000)  # Convert to km
        masses_rel.append(M / solar_mass)  # Convert to solar masses
    # force y-axis to start at 0 with no extra lower margin
    
    
    masses_non_rel = np.array(masses_non_rel)
    radii_non_rel = np.array(radii_non_rel)
    sort_indices = np.argsort(masses_non_rel)
    masses_non_rel = masses_non_rel[sort_indices]
    radii_non_rel = radii_non_rel[sort_indices]
    
    # Define realistic carbon white dwarf mass range
    min_carbon_mass = 0.5  # Solar masses
    max_carbon_mass = 1.4  # Solar masses
    
    
    min_idx = np.argmax(masses_non_rel >= min_carbon_mass)
    max_idx = np.argmax(masses_non_rel > max_carbon_mass)
    if max_idx == 0:  # If no masses exceed max_carbon_mass
        max_idx = len(masses_non_rel)
    if min_idx > 0:
        low_end = min(min_idx + 1, len(masses_non_rel))
        plt.plot(masses_non_rel[0:low_end], radii_non_rel[0:low_end], 
                color='lightskyblue', linestyle='--', alpha=0.7)
    if max_idx > min_idx:
        plt.plot(masses_non_rel[min_idx:max_idx], radii_non_rel[min_idx:max_idx], 
                marker='', label='Non-relativistic EOS (Typical Observed Range)', color='lightskyblue', linestyle='-')
    if max_idx < len(masses_non_rel):
        high_start = max(max_idx - 1, 0)
        plt.plot(masses_non_rel[high_start:], radii_non_rel[high_start:], 
                color='lightskyblue', linestyle='--', alpha=0.7)
    
    plt.plot([], [], color='lightskyblue', linestyle='--', alpha=0.7, label='Non-relativistic EOS (Unphysical)')
    plt.plot(masses_rel, radii_rel, marker='', label='Relativistic EOS', color='indianred')
    
    
    # Apply a dark theme: black background, white axes/labels/ticks/legend
    fig = plt.gcf()
    fig.patch.set_facecolor('black')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'legend.facecolor': 'black',
        'legend.edgecolor': 'white'
    })
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.tick_params(colors='white', which='both')
    for spine in ax.spines.values():
        spine.set_color('white')
        
        
    plt.plot([white_dwarf_structure_rel(4e22)[1]/solar_mass]*2, [0, 20000], color='gray', linestyle='--', label='Chandrasekhar Limit without GR')
    
    plt.xlabel('Mass (M☉)', fontsize=14)
    plt.ylabel('Radius (km)', fontsize=14)
    plt.title('White Dwarf Mass-Radius Relation with General Relativitistic Corrections', fontsize=16)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Neutron star functions

def neutron_star_structure_non_rel_TOV(p_central):
    
    def stellar_structure_equations(r, y):
        
        p, M = y
        
        # Equation of state: rho = (p/K)^(3/5)
        if p > 0:
            rho = 1/c**2 * (p/K_non_rel_neutron)**(3/5)
        else:
            rho = 0
            return [0, 0]  # Stop if pressure becomes zero or negative

        epsilson = rho * c**2  # Energy density
    
        # Stellar structure equations
        dpdr = -(G * M * rho / r**2)*(1+p/epsilson)*(1+(4*pi*(r**3)*p)/(M*(c**2)))*(1-(2*G*M)/((c**2)*r))**(-1)  # TOV
        dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Integration parameters (optimized for typical white dwarf sizes)
    r_start = 0.01     # meters
    r_max = 2e4       # 20km (more realistic upper bound)
    
    # Initial mass within r_start
    rho_central = 1/c**2 *(p_central / K_non_rel_neutron)**(3/5)
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0] - 1e5  # Stop when pressure drops to 10^5 Pa (near vacuum)
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve the ODEs
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',          # Good balance of speed and accuracy
        dense_output=True,    # Disabled - saves memory and computation
        rtol=1e-6,             # Relaxed tolerance for speed
        atol=1e-10,            # Sufficient precision for stellar structure
        max_step=10           # Smaller steps for neutron stars
    )
 
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Surface found via event
        R_surface = sol.t_events[0][0]
        M_total = sol.y_events[0][0][1]
        p_surface = sol.y_events[0][0][0]
        #print(f"\n Surface found!")
        #print(f"Radius: {R_surface/1000:.1f} km")
        #print(f"Total mass: {M_total/solar_mass:.4f} M☉")
        #print(f"Surface pressure: {p_surface:.2e} Pa")
        return [R_surface, M_total, p_surface]

    else:
        # Integration went to maximum radius
        if len(sol.t) > 0:
            R_final = sol.t[-1]
            M_final = sol.y[1, -1]
            p_final = sol.y[0, -1]            
            #print(f"Final radius: {R_final/1000:.1f} km")
            #print(f"Final mass: {M_final/solar_mass:.4f} M☉")
            #print(f"Final pressure: {p_final:.2e} Pa")
            print(f"\nReached maximum radius")
            return [R_final, M_final, p_final]
        else:
            print("Integration failed!")
            return None, None, None

    #print(f'Radius: {white_dwarf_structure(p_central)[0]/1000:.1f} km, Mass: {white_dwarf_structure(p_central)[1]/solar_mass:.4f} M☉')

def neutron_star_structure_non_rel(p_central):
    
    def stellar_structure_equations(r, y):
        
        p, M = y
        
        # Equation of state: rho = (p/K)^(3/5)
        if p > 0:
            rho = 1/c**2 * (p/K_non_rel_neutron)**(3/5)
        else:
            rho = 0
            return [0, 0]  # Stop if pressure becomes zero or negative

        epsilson = rho * c**2  # Energy density
    
        # Stellar structure equations
        dpdr = -(G * M * rho / r**2)
        dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Integration parameters (optimized for typical white dwarf sizes)
    r_start = 0.01     # meters
    r_max = 2e4       # 20km (more realistic upper bound)
    
    # Initial mass within r_start
    rho_central = 1/c**2 *(p_central / K_non_rel_neutron)**(3/5)
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0] - 1e5  # Stop when pressure drops to 10^5 Pa (near vacuum)
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve the ODEs
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',          # Good balance of speed and accuracy
        dense_output=False,     # Disabled - saves memory and computation
        rtol=1e-6,             # Relaxed tolerance for speed
        atol=1e-10,            # Sufficient precision for stellar structure
        max_step=10           # Smaller steps for neutron stars
    )
 
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Surface found via event
        R_surface = sol.t_events[0][0]
        M_total = sol.y_events[0][0][1]
        p_surface = sol.y_events[0][0][0]
        #print(f"\n Surface found!")
        #print(f"Radius: {R_surface/1000:.1f} km")
        #print(f"Total mass: {M_total/solar_mass:.4f} M☉")
        #print(f"Surface pressure: {p_surface:.2e} Pa")
        return [R_surface, M_total, p_surface]

    else:
        # Integration went to maximum radius
        if len(sol.t) > 0:
            R_final = sol.t[-1]
            M_final = sol.y[1, -1]
            p_final = sol.y[0, -1]            
            #print(f"Final radius: {R_final/1000:.1f} km")
            #print(f"Final mass: {M_final/solar_mass:.4f} M☉")
            #print(f"Final pressure: {p_final:.2e} Pa")
            print(f"\nReached maximum radius")
            return [R_final, M_final, p_final]
        else:
            print("Integration failed!")
            return None, None, None

    #print(f'Radius: {white_dwarf_structure(p_central)[0]/1000:.1f} km, Mass: {white_dwarf_structure(p_central)[1]/solar_mass:.4f} M☉')

'''
def neutron_star_structure_rel_TOV(p_central):
    
    #print(f"Polytropic constant K = {K_rel:.3e} Pa⋅m^5⋅kg^(-4/3)")

    def stellar_structure_equations(r, y):
        p, M = y
        
        # Equation of state: rho = (p/K)
        if p > 0:
            rho = 1/c**2 * (p/K_rel_neutron)
        else:
            rho = 0
            return [0, 0]  # Stop if pressure becomes zero or negative
        
        epsilson = rho * c**2  # Energy density
        
        # Stellar structure equations
        dpdr = -(G * M * rho / r**2)*(1+p/epsilson)*(1+(4*pi*(r**3)*p)/(M*(c**2)))*1/(1-(2*G*M)/((c**2)*r))  # TOV
        dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Integration parameters (optimized for typical white dwarf sizes)
    r_start = 1     # meters
    r_max = 2e4       # 20,000 km (more realistic upper bound)
    
    # Initial mass within r_start
    rho_central = 1/c**2 *(p_central / K_rel_neutron)
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0] - 10**5 # Stop when pressure drops to 0 Pa (near vacuum)
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve the ODEs
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',          # Good balance of speed and accuracy
        dense_output=False,    # Disabled - saves memory and computation
        rtol=1e-6,             # Relaxed tolerance for speed
        atol=1e-10,            # Sufficient precision for stellar structure
        max_step=10           # Larger steps for faster integration
    )
 
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Surface found via event
        R_surface = sol.t_events[0][0]
        M_total = sol.y_events[0][0][1]
        p_surface = sol.y_events[0][0][0]
        #print(f"\n Surface found!")
        print(f"Radius: {R_surface/1000:.1f} km")
        print(f"Total mass: {M_total/solar_mass:.4f} M☉")
        print(f"Surface pressure: {p_surface:.2e} Pa")
        return [R_surface, M_total, p_surface]

    else:
        # Integration went to maximum radius
        if len(sol.t) > 0:
            R_final = sol.t[-1]
            M_final = sol.y[1, -1]
            p_final = sol.y[0, -1]            
            print(f"Final radius: {R_final/1000:.1f} km")
            print(f"Final mass: {M_final/solar_mass:.4f} M☉")
            print(f"Final pressure: {p_final:.2e} Pa")
            print(f"\nReached maximum radius")
            return [R_final, M_final, p_final]
        else:
            print("Integration failed!")
            return None, None, None
'''

def plot_p_vs_r_neutron_rel_TOV(p_central):
    """
    Plot pressure against radius for a single neutron star with relativistic TOV equation.
    
    Parameters:
    p_central: Central pressure in Pa
    """
    import matplotlib.pyplot as plt
    
    def stellar_structure_equations(r, y):
        p, M = y
        
        # Equation of state: rho = (p/K)
        if p > 0:
            rho = 1/c**2 * (p/K_rel_neutron)
        else:
            rho = 0
            return [0, 0]
        
        epsilson = rho * c**2  # Energy density
        
        # Stellar structure equations
        dpdr = -(G * M * rho / r**2)*(1+p/epsilson)*(1+(4*pi*(r**3)*p)/(M*(c**2)))*1/(1-(2*G*M)/((c**2)*r))  # TOV
        dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Integration parameters
    r_start = 1     # meters
    r_max = 2e9     # 20,000 km
    
    # Initial mass within r_start
    rho_central = 1/c**2 *(p_central / K_rel_neutron)
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0] - 10**5
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve the ODEs with dense output
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',
        dense_output=True,
        rtol=1e-6,
        atol=1e-10,
        max_step=10000
    )
    
    # Extract radius and pressure
    radii = sol.t / 1000  # Convert to km
    pressures = sol.y[0]  # Pressure is first component
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(radii, pressures, linewidth=2, color='cyan')
    
    # Apply dark theme
    fig = plt.gcf()
    fig.patch.set_facecolor('black')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
    })
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.tick_params(colors='white', which='both')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.xlabel('Radius (km)', fontsize=12)
    plt.ylabel('Pressure (Pa)', fontsize=12)
    plt.title(f'Neutron Star Pressure', fontsize=14)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def graph_neutron_star_mass_radius():
    """
    Plot mass-radius relation for neutron stars using non-relativistic equation of state with TOV corrections.
    Central pressures range from 1e33 to 1e50 Pa.
    
    dpdr= -(G * M * rho / r**2)*(1+p/epsilson)*(1+(4*pi*(r**3)*p)/(M*(c**2)))*(1-(2*G*M)/((c**2)*r))**(-1)
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    p_central_values = np.logspace(32.5, 48, num=100)  # Central pressures from 1e32 to 1e48 Pa
    radii = []
    masses = []
    
    for p_central in p_central_values:
        result = neutron_star_structure_non_rel_TOV(p_central)
        if result and result[0] is not None:
            R, M, _ = result
            radii.append(R / 1000)  # Convert to km
            masses.append(M / solar_mass)  # Convert to solar masses

    masses = np.array(masses)
    radii = np.array(radii)
    
    # Plot
    plt.plot(masses, radii, label='Neutron Star (Non-relativistic EOS + TOV)', 
             color='gold', linewidth=2)
    
    p_central_values = np.logspace(32.7, 35, num=100)  # Central pressures from 1e32 to 1e48 Pa
    radii = []
    masses = []
    
    for p_central in p_central_values:
        result = neutron_star_structure_non_rel(p_central)
        if result and result[0] is not None:
            R, M, _ = result
            radii.append(R / 1000)  # Convert to km
            masses.append(M / solar_mass)  # Convert to solar masses

    masses = np.array(masses)
    radii = np.array(radii)
    
    # Plot
    plt.plot(masses, radii, label='Neutron Star (Non-relativistic EOS Without TOV)', 
             color='green', linewidth=2)
    
    # Apply dark theme
    fig = plt.gcf()
    fig.patch.set_facecolor('black')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'legend.facecolor': 'black',
        'legend.edgecolor': 'white'
    })
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.tick_params(colors='white', which='both')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.xlabel('Mass (M☉)', fontsize=14)
    plt.ylabel('Radius (km)', fontsize=14)
    plt.title('Neutron Star Mass-Radius Relation', fontsize=16)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

#Neturon star combined EOS

def combined_eos_neutron():
    ps = []
    epsilons = []
    # Fermi wave number range (m^-1); spans NR to UR regimes around nuclear densities
    kf_values = np.linspace(0.5e15, 5e15, 1000)

    #pref = m_neutron**4 * c**5 / (8 * pi**2 * hbar**3) but not needed for fitting
    for kf in kf_values:
        # Dimensionless parameter: x = ħ k_F / (m c)
        x = (hbar * kf) / (m_neutron * c)
        # Pressure and energy density
        p = ((1/3) * x * (2 * x**2 - 3) * np.sqrt(1 + x**2) + np.arcsinh(x))
        epsilon = (x * (2 * x**2 + 1) * np.sqrt(1 + x**2) - np.arcsinh(x))
        ps.append(p)
        epsilons.append(epsilon)
    
    p_arr = np.array(ps)
    eps_arr = np.array(epsilons)
    
    # Model: epsilon(p) = Anr * p^(3/5) + Ar * p
    def model(p, Anr, Ar):
        return Anr * np.power(p, 3/5) + Ar * p

    # Use log-space fitting so both low-p (Anr) and high-p (Ar) regions contribute equally
    log_eps = np.log(eps_arr)
    def log_model(p, Anr, Ar):
        return np.log(Anr * np.power(p, 3/5) + Ar * p)

    params, cov = curve_fit(log_model, p_arr, log_eps, p0=(3, 3), 
                            bounds=(0, np.inf), maxfev=20000)
    Anr, Ar = params
    stderr = np.sqrt(np.diag(cov)) if cov is not None else np.array([np.nan, np.nan])

    #print(f"Fit results (log-space): Anr={Anr:.6e}, Ar={Ar:.6e}")
    #print(f"Std errors: σ_Anr={stderr[0]:.6e}, σ_Ar={stderr[1]:.6e}")

    # Overlay fitted model line on the EOS plot
    eps_fit = model(p_arr, Anr, Ar)
    otherfit = model(p_arr, 2.4216, 2.8663)  # From literature for comparison
    
    '''
    plt.figure()
    plt.plot(p_arr, eps_arr, color='cyan', linewidth=2, label='Direct Calculation')
    plt.plot(p_arr, eps_fit, color='magenta', linewidth=2, linestyle='--', label='Fit: ε=Anr·p^(3/5)+Ar·p')
    plt.plot(p_arr, otherfit, color='yellow', linewidth=2, linestyle=':', label='Literature Fit')
    plt.xlabel('Pressure (Pa)', fontsize=12)
    plt.ylabel('Energy Density (J/m³)', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    '''
    return Anr, Ar, stderr

def neutron_star_structure_combined(p_central):
    
    Anr_neutron, Ar_neutron, _ = combined_eos_neutron()
    
    # Unit prefactor to convert from dimensionless to SI units
    pref = m_neutron**4 * c**5 / (8 * pi**2 * hbar**3)

    def stellar_structure_equations(r, y):
        
        p, M = y
        # Energy density from fitted EOS (convert dimensionless quantities to SI)
        p_dimensionless = p / pref
        epsilon_dimensionless = Anr_neutron * (p_dimensionless)**(3/5) + Ar_neutron * p_dimensionless
        epsilon = epsilon_dimensionless * pref  # Energy density in SI units (J/m³)
        
        # Equation of state: rho = (p/K)^(3/5)
        if p > 0:
            rho = epsilon/c**2
        else:
            rho = 0
            return [0, 0]  # Stop if pressure becomes zero or negative

        # Stellar structure equations
        dpdr = -(G * M * rho / r**2)*(1+p/epsilon)*(1+(4*pi*(r**3)*p)/(M*(c**2)))*(1-(2*G*M)/((c**2)*r))**(-1)  # TOV
        dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Integration parameters (optimized for typical white dwarf sizes)
    r_start = 0.01     # meters
    r_max = 2e4       # 20km (more realistic upper bound)
    
    epsilon_central = Anr_neutron * (p_central)**(3/5) + Ar_neutron * p_central  # Central energy density
    # Initial mass within r_start
    rho_central = epsilon_central/c**2
    M_start = (4/3) * pi * r_start**3 * rho_central
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0] - 1e5  # Stop when pressure drops to 10^5 Pa (near vacuum)
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    # Solve the ODEs
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',          # Good balance of speed and accuracy
        dense_output=False,    # Disabled - saves memory and computation
        rtol=1e-6,             # Relaxed tolerance for speed
        atol=1e-10,            # Sufficient precision for stellar structure
        max_step=10           # Smaller steps for neutron stars
    )
 
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Surface found via event
        R_surface = sol.t_events[0][0]
        M_total = sol.y_events[0][0][1]
        p_surface = sol.y_events[0][0][0]
        #print(f"\n Surface found!")
        #print(f"Radius: {R_surface/1000:.1f} km")
        #print(f"Total mass: {M_total/solar_mass:.4f} M☉")
        #print(f"Surface pressure: {p_surface:.2e} Pa")
        
        '''
        plt.figure()
        plt.plot(sol.t/1000, sol.y[0], color='lightblue', linewidth=2, label='Pressure')
        #plt.yscale('log')
        plt.ylabel('Pressure (Pa)', fontsize=12)
        plt.xlabel('Radius (km)', fontsize=12)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(sol.t/1000, sol.y[1]/solar_mass, color='indianred', linewidth=2, label='Mass Enclosed')
        ax2.set_ylabel('Mass Enclosed (M☉)', fontsize=12)
        ax2.tick_params(axis='y')
        plt.legend(ax2.get_legend_handles_labels()[0] + ax1.get_legend_handles_labels()[0],
                   ax2.get_legend_handles_labels()[1] + ax1.get_legend_handles_labels()[1],
                   fontsize=11, loc='center right')
        plt.title(f'Neutron Star Pressure Profile (Combined EOS)', fontsize=14)
        '''    
        
        return [R_surface, M_total, p_surface]

    else:
        # Integration went to maximum radius
        if len(sol.t) > 0:
            R_final = sol.t[-1]
            M_final = sol.y[1, -1]
            p_final = sol.y[0, -1]            
            #print(f"Final radius: {R_final/1000:.1f} km")
            #print(f"Final mass: {M_final/solar_mass:.4f} M☉")
            #print(f"Final pressure: {p_final:.2e} Pa")
            print(f"\nReached maximum radius")
            return [R_final, M_final, p_final]
        else:
            print("Integration failed!")
            return None, None, None

def graph_neutron_star_mass_radius_combined():
    plt.figure(figsize=(10, 6))

    p_central_values = np.logspace(32.5, 42, num=100)  # Central pressures from 1e32 to 1e42 Pa. (34.5,34.6)
    radii = []
    masses = []
    pressures = []
    
    for p_central in p_central_values:
        result = neutron_star_structure_combined(p_central)
        if result and result[0] is not None:
            R, M, _ = result
            radii.append(R / 1000)  # Convert to km
            masses.append(M / solar_mass)  # Convert to solar masses
            pressures.append(p_central)

    masses = np.array(masses)
    radii = np.array(radii)
    pressures = np.array(pressures)
    
    ''' Find maximum mass and corresponding values
    max_mass_idx = np.argmax(masses)
    max_mass = masses[max_mass_idx]
    max_mass_radius = radii[max_mass_idx]
    max_mass_pressure = pressures[max_mass_idx]
    print(f"\n=== Maximum Mass Configuration ===")
    print(f"Maximum Mass: {max_mass:.4f} M☉")
    print(f"Corresponding Radius: {max_mass_radius:.2f} km")
    print(f"Central Pressure: {max_mass_pressure:.3e} Pa")
    print(f"==================================\n")
    '''
    
    # Plot
    plt.plot(masses, radii, label='Neutron Star (Combined EOS + TOV)', 
             color='lightskyblue', linewidth=2)
    
    
    ''' Apply dark theme
    fig = plt.gcf()
    fig.patch.set_facecolor('black')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'legend.facecolor': 'black',
        'legend.edgecolor': 'white'
    })
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.tick_params(colors='white', which='both')
    for spine in ax.spines.values():
        spine.set_color('white')
    '''
    
    plt.plot([0.7094, 0.7094], [0, 9.16], color='indianred', linestyle='--', alpha=0.7)
    plt.plot([0, 0.7094], [9.16, 9.16], color='indianred', linestyle='--', alpha=0.7)
    
    plt.xlabel('Mass (M☉)', fontsize=14)
    plt.ylabel('Radius (km)', fontsize=14)
    plt.title('Neutron Star Mass-Radius Relation for general EOS: $\epsilon = A_{NR}p^{3/5} + A_Rp$', fontsize=16)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    #plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    
    
     
#%%
#neutron_star_structure_combined(1e42)
#graph_mass_radius()
#plot_mass_pressure_vs_radius(4e22)
#print(white_dwarf_structure_rel(4e22)[1]/solar_mass)
#graph_mass_radius_TOV()
#print(white_dwarf_structure_rel_TOV(4e22)[1]/solar_mass)
#graph_neutron_star_mass_radius()
#plot_p_vs_r_neutron_rel_TOV(1e34)
#combined_eos_neutron()
graph_neutron_star_mass_radius_combined()
#%%
