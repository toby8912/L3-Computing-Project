
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import G, h, m_e, pi , c

def white_dwarf_structure(p_central):
    """
    Solve white dwarf structure using the ODEs:
    dp/dr = -G*M*rho/r²
    dM/dr = 4*π*r²*rho
    """
    # Equation of state for non-relativistic degenerate electron gas
    # P = K * rho^(5/3) where K is the polytropic constant
    
    solar_mass = 1.98847e30  # kg
    m_n = 1.66054e-27  # atomic mass unit (kg)
    Z_over_A = 0.5     # for carbon-12 
    K = ((h/(2*pi))**2 / (15* pi**2 * m_e)) * ((3*pi**2 * Z_over_A) / (m_n * c**2))**(5/3)
    
    print(f"WHITE DWARF STELLAR STRUCTURE SOLVER")
    print(f"Using stellar structure ODEs: dp/dr and dM/dr")
    print(f"Polytropic constant K = {K:.3e} Pa⋅m^5⋅kg^(-5/3)")
    
    def stellar_structure_equations(r, y):
        """
        Stellar structure equations:
        y[0] = p (pressure)
        y[1] = M (mass enclosed)
        """
        p, M = y
        
        # Equation of state: rho = (p/K)^(3/5)
        if p > 0:
            rho = 1/c**2 * (p/K)**(3/5)
        else:
            rho = 0
            return [0, 0]  # Stop if pressure becomes zero or negative
        
        # Handle singularity at r = 0
        if r <= 0:
            dpdr = 0
            dMdr = 0
        else:
            # Stellar structure equations
            dpdr = -G * M * rho / r**2  # Hydrostatic equilibrium
            dMdr = 4 * pi * r**2 * rho  # Mass continuity
        
        return [dpdr, dMdr]
    
    # Start integration from small radius (not zero to avoid singularity)
    r_start = 1.0  # meters
    r_max = 1e8    # 100,000 km maximum
    
    # Initial mass within r_start
    rho_central = 1/c**2 *(p_central / K)**(3/5)
    M_start = (4/3) * pi * r_start**3 * rho_central
    print(f"Central density: {rho_central:.2e} kg/m³")
    print(f"\nInitial conditions:")
    print(f"Central pressure: {p_central:.2e} Pa")
    print(f"Central density: {rho_central:.2e} kg/m³") 
    print(f"Starting radius: {r_start} m")
    print(f"Starting mass: {M_start:.3e} kg ({M_start/solar_mass:.6f} M☉)")
    
    # Initial conditions
    y0 = [p_central, M_start]
    r_span = (r_start, r_max)
    
    # Event function: stop when pressure drops to near zero
    def surface_condition(r, y):
        return y[0] - 1e5  # Stop when pressure drops to 10^5 Pa (very close to zero for stellar scales)
    surface_condition.terminal = True
    surface_condition.direction = -1
    
    print("\nIntegrating stellar structure equations...")
    
    # Solve the ODEs
    sol = solve_ivp(
        stellar_structure_equations,
        r_span,
        y0,
        events=surface_condition,
        method='RK45',
        dense_output=True,
        rtol=1e-8,
        atol=1e-12,
        max_step=100
    )
    
    #print(f"Integration status: {sol.status}")
    
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Surface found via event
        R_surface = sol.t_events[0][0]
        M_total = sol.y_events[0][0][1]
        p_surface = sol.y_events[0][0][0]
        
        print(f"\n✓ Surface found!")
        print(f"Radius: {R_surface/1000:.1f} km")
        print(f"Total mass: {M_total/solar_mass:.4f} M☉")
        print(f"Surface pressure: {p_surface:.2e} Pa")
        
        return R_surface, M_total, p_central
        
    else:
        # Integration went to maximum radius
        if len(sol.t) > 0:
            R_final = sol.t[-1]
            M_final = sol.y[1, -1]
            p_final = sol.y[0, -1]
            
            print(f"\nReached maximum radius:")
            print(f"Final radius: {R_final/1000:.1f} km")
            print(f"Final mass: {M_final/solar_mass:.4f} M☉")
            print(f"Final pressure: {p_final:.2e} Pa")
            
            return R_final, M_final, p_central
        else:
            print("Integration failed!")
            return None, None, None

p_central = 4e22
white_dwarf_structure(p_central)
