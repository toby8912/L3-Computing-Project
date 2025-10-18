#%%
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import G, h, m_e, pi, c
solar_mass = 1.98847e30  # kg
m_n = 1.66054e-27  # atomic mass unit (kg)
Z_over_A = 0.5     # for carbon-12 
K = ((h/(2*pi))**2 / (15* pi**2 * m_e)) * ((3*pi**2 * Z_over_A) / (m_n * c**2))**(5/3)

def white_dwarf_structure(p_central):
    '''
    Solve white dwarf structure using the ODEs:
    dp/dr = -G*M*rho/r²
    dM/dr = 4*π*r²*rho

    print(f"WHITE DWARF STELLAR STRUCTURE SOLVER")
    print(f"Using stellar structure ODEs: dp/dr and dM/dr")
    print(f"Polytropic constant K = {K:.3e} Pa⋅m^5⋅kg^(-5/3)")
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
            rho = 1/c**2 * (p/K)**(3/5)
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
    rho_central = 1/c**2 *(p_central / K)**(3/5)
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

#p_central = 4e22 For sun like white dwarf
#p_central = 1e19 Minimum for graphing
#p_central = 1e24 Maximum for graphing

def graph_mass_radius():
    import matplotlib.pyplot as plt

    p_central_values = np.logspace(19, 24, num=50)  # Central pressures from 10^19 to 10^24 Pa
    radii = []
    masses = []

    for p_central in p_central_values:
        R, M, _ = white_dwarf_structure(p_central)
        radii.append(R / 1000)  # Convert to km
        masses.append(M / solar_mass)  # Convert to solar masses

    plt.figure(figsize=(10, 6))
    plt.plot(masses, radii, marker='')
    plt.xlabel('Mass (M☉)')
    plt.ylabel('Radius (km)')
    plt.title('White Dwarf Mass-Radius Relation')
    plt.show()

#print(f'Radius: {white_dwarf_structure(p_central)[0]/1000:.1f} km, Mass: {white_dwarf_structure(p_central)[1]/solar_mass:.4f} M☉')
#%%
graph_mass_radius()
# %%
