
from scipy.constants import speed_of_light as c, Planck as h, G, elementary_charge as e, m_e, m_p, m_n
from scipy import integrate
import numpy as np
def integration():
    def equations(r, y):
        p, M = y
        
        # White dwarf equation of state (non-relativistic degenerate electron gas)
        # p = K * rho^(5/3) where K = (h^2/5m_e) * (3/8π)^(2/3) * (Z/A)^(5/3)
        # For C-12: Z/A = 0.5, so K ≈ 9.9e12 in SI units (Pa⋅m^5⋅kg^(-5/3))
        K = 9.9e12  
        
        if p <= 0:
            return [0, 0]  # Stop integration if pressure becomes non-positive
            
        rho = (p / K)**(3/5)  # Density from EOS
        
        # Stellar structure equations
        if r <= 0:
            dpdr = 0
            dMdr = 0
        else:
            dpdr = -G * M * rho / r**2 
            dMdr = 4 * np.pi * r**2 * rho
            
        return [dpdr, dMdr]

    # Integration parameters
    r_start = 1.0     # Start at 1 meter (to avoid r=0 singularity)
    r_end = 2e7       # Maximum radius (20,000 km)
    
    # Initial conditions - typical white dwarf values
    p_central = 5e22  # Central pressure (Pa) 
    K = 9.9e12
    rho_central = (p_central / K)**(3/5)
    M_start = 4/3 * np.pi * (r_start**3) * rho_central  # Mass within starting radius
    
    r_span = (r_start, r_end)
    y0 = [p_central, M_start]
    
    print(f"Starting integration with:")
    print(f"  Central pressure: {p_central:.2e} Pa")
    print(f"  Central density: {rho_central:.2e} kg/m³")
    print(f"  Starting radius: {r_start} m")
    print(f"  Initial mass: {M_start:.2e} kg")

    def stop_event(r, y):
        return y[0] - 1e18  # Stop when pressure drops significantly
    stop_event.terminal = True
    stop_event.direction = -1

    sol = integrate.solve_ivp(
        equations,
        r_span,
        y0,
        events=stop_event,
        method='RK45',
        max_step=1000,
        rtol=1e-6,
        atol=1e-10
    )

    solar_mass = 1.98847e30  # kg
    
    if sol.status == 1 and len(sol.t_events[0]) > 0:
        # Integration stopped due to event (pressure reached threshold)
        final_mass_solar = sol.y_events[0][0][1] / solar_mass
        final_radius_km = sol.t_events[0][0] / 1e3
        print(f"Integration successful!")
        print(f"Final mass M = {final_mass_solar:.4f} solar masses")
        print(f"Final radius R = {final_radius_km:.1f} km")
        print(f"Final pressure = {sol.y_events[0][0][0]:.2e} Pa")
    else:
        # Integration reached maximum radius or failed
        print(f"Integration status: {sol.status}")
        if len(sol.t) > 0:
            last_p = sol.y[0, -1]
            last_M = sol.y[1, -1]
            last_r = sol.t[-1]
            print(f"Last computed pressure p = {last_p:.3e} Pa")
            print(f"Last computed mass M = {last_M/solar_mass:.6f} solar masses") 
            print(f"Last computed radius r = {last_r/1e3:.4f} km")
            
            # Check if we're close to a solution
            if last_p < 1e12:  # If pressure is reasonably low
                print(f"Approximate solution: M ≈ {last_M/solar_mass:.4f} M☉, R ≈ {last_r/1e3:.1f} km")
integration()  # Example central pressure in Pascals
