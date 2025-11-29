#%%
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import G, h, m_e, pi, c
solar_mass = 1.98847e30
m_n = 1.660539e-27 #average nucleon mass for carbon 12
Z_over_A = 0.5     # for carbon-12 
K_non_rel = ((h/(2*pi))**2 / (15* pi**2 * m_e)) * ((3*pi**2 * Z_over_A) / (m_n * c**2))**(5/3)
K_rel = (h*c/(24*pi**3))*((3*pi**2 * Z_over_A)/(m_n * c**2))**(4/3)


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


#%%
#graph_mass_radius()
#plot_mass_pressure_vs_radius(4e22)
#print(white_dwarf_structure_rel(4e22)[1]/solar_mass)
graph_mass_radius_TOV()
#print(white_dwarf_structure_rel_TOV(4e22)[1]/solar_mass) 
#%%