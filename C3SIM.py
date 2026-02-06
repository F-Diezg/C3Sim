#C3Sim
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from scipy.optimize import newton

#Engine
#------------------#
#Constants 
G_REAL = 6.67430e-20
M_EARTH_REAL = 5.972e24
M_MOON_REAL = 7.348e22
DIST_TL_REAL = 384400.0
R_EARTH_REAL = 6371.0
R_MOON_REAL = 1737.0
MU_EARTH_REAL = G_REAL * M_EARTH_REAL

#3BP Normalization
L_STAR = DIST_TL_REAL
M_STAR = M_EARTH_REAL + M_MOON_REAL
MU = M_MOON_REAL / M_STAR
omega_real = np.sqrt(G_REAL * M_STAR / L_STAR**3)
T_STAR = 1 / omega_real
V_STAR = L_STAR / T_STAR 

#Normalized Radius 
R_EARTH_N = R_EARTH_REAL / L_STAR
R_MOON_N = R_MOON_REAL / L_STAR
R_SOI_M_N = (DIST_TL_REAL * (M_MOON_REAL / M_EARTH_REAL)**0.4) / L_STAR
R_SOI_E_N = (925000.0 / L_STAR) 

def pos_bodies_inertial(t):
    #bodies position
    xe = -MU * np.cos(t); ye = -MU * np.sin(t)
    xm = (1 - MU) * np.cos(t); ym = (1 - MU) * np.sin(t)
    return np.array([xe, ye]), np.array([xm, ym])

def equations_inertial(t, state):
    x, y, vx, vy = state
    pos_e, pos_m = pos_bodies_inertial(t) 
    
    r_e = np.array([x, y]) - pos_e
    r_m = np.array([x, y]) - pos_m
    
    norm_e3 = np.linalg.norm(r_e)**3 + 1e-15
    norm_m3 = np.linalg.norm(r_m)**3 + 1e-15
    
    ax = -(1 - MU) * r_e[0] / norm_e3 - MU * r_m[0] / norm_m3
    ay = -(1 - MU) * r_e[1] / norm_e3 - MU * r_m[1] / norm_m3
    
    return [vx, vy, ax, ay]
#------------------#


#Events trigger
#------------------#
def crash_earth(t, state):
    x, y, _, _ = state
    pos_e, _ = pos_bodies_inertial(t)
    return np.linalg.norm(np.array([x, y]) - pos_e) - R_EARTH_N
crash_earth.terminal = True 

def crash_moon(t, state):
    x, y, _, _ = state
    _, pos_m = pos_bodies_inertial(t)
    return np.linalg.norm(np.array([x, y]) - pos_m) - R_MOON_N
crash_moon.terminal = True 

def get_lagrange_local():
    Lx, Ly = [0.5 - MU, 0.5 - MU], [np.sqrt(3)/2, -np.sqrt(3)/2]
    def f_grad(x):
        r1 = abs(x + MU); r2 = abs(x - (1 - MU))
        return x - (1 - MU)*np.sign(x + MU)/r1**2 - MU*np.sign(x - (1 - MU))/r2**2
    try:
        Lx.extend([newton(f_grad, 0.83), newton(f_grad, 1.15), newton(f_grad, -1.0)])
        Ly.extend([0, 0, 0])
    except: pass
    return np.array(Lx), np.array(Ly)

L_loc_x, L_loc_y = get_lagrange_local()

def rotate_points(x_arr, y_arr, ang_t):
    c, s = np.cos(ang_t), np.sin(ang_t)
    return x_arr * c - y_arr * s, x_arr * s + y_arr * c
#------------------#


#GUI
#------------------#
class SerenitasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("C3Sim 2.0")
        self.root.geometry("1300x900")
        self.root.configure(bg="#2b2b2b") 

        self.ani = None 
        self.cached_trajectory = None 
        self.cached_events = None
        self.cached_params = None
        
        self.initial_energy_msg = ""
        self.initial_energy_col = ""
        self.final_status_msg = ""
        self.final_status_col = ""

        #LAYOUT
        self.frame_controls = tk.Frame(root, bg="#1e1e1e", width=340, padx=20, pady=20)
        self.frame_controls.pack(side=tk.LEFT, fill=tk.Y)
        
        self.frame_right = tk.Frame(root, bg="black")
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        #Control panel
        self.create_label("Mission Parameters", font=("Arial", 14, "bold"), color="#00ffcc")
        
        self.entry_x = self.create_input("Position X [km]:", "-11500")
        self.entry_y = self.create_input("Position Y [km]:", "0")
        self.entry_vx = self.create_input("Velocity Vx [km/s]:", "0")
        self.entry_vy = self.create_input("Velocity Vy [km/s]:", "-10.0")
        self.entry_days = self.create_input("Duration [days]:", "8")
        
        self.create_separator()

        #Calculator
        self.create_label("Stable Orbital Calculator", font=("Arial", 11, "bold"), color="#ffcc00")
        self.create_label("Target Altitude [km]:", font=("Arial", 9), color="#cccccc")
        
        calc_frame = tk.Frame(self.frame_controls, bg="#1e1e1e")
        calc_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.entry_alt = tk.Entry(calc_frame, bg="#333", fg="white", width=10)
        self.entry_alt.insert(0, "400") 
        self.entry_alt.pack(side=tk.LEFT, padx=(0, 10))
        
        btn_calc = tk.Button(calc_frame, text="Calculate V", bg="#444", fg="white", font=("Arial", 8),
                            command=self.calculate_circular_velocity)
        btn_calc.pack(side=tk.LEFT)

        self.lbl_calc_res = tk.Label(self.frame_controls, text="Result: -", bg="#1e1e1e", fg="orange", font=("Consolas", 10))
        self.lbl_calc_res.pack(anchor="w", pady=(2, 5))
        
        self.btn_autofill = tk.Button(self.frame_controls, text="⬇ Apply to Mission", bg="#333", fg="#00ffcc", 
                                        font=("Arial", 8), state=tk.DISABLED, command=self.apply_calculation)
        self.btn_autofill.pack(fill=tk.X, pady=(0, 15))

        self.create_separator()

        #Buttons
        self.btn_run = tk.Button(self.frame_controls, text="🚀 LAUNCH MISSION", 
                                bg="#00aa00", fg="white", font=("Arial", 11, "bold"),
                                command=self.run_simulation, cursor="hand2", pady=5)
        self.btn_run.pack(fill=tk.X, pady=(10, 5))

        self.btn_replay = tk.Button(self.frame_controls, text="🔄 REPLAY LAST", 
                                bg="#0077cc", fg="white", font=("Arial", 10, "bold"),
                                command=self.replay_simulation, cursor="hand2", pady=5, state=tk.DISABLED)
        self.btn_replay.pack(fill=tk.X, pady=5)

        self.btn_reset = tk.Button(self.frame_controls, text="🗑️ RESET DEFAULTS", 
                                bg="#555555", fg="white", font=("Arial", 9),
                                command=self.reset_defaults, cursor="hand2", pady=5)
        self.btn_reset.pack(fill=tk.X, pady=5)

        #Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("C3Sim 2.0 Systems Online.")
        self.lbl_status = tk.Label(self.frame_controls, textvariable=self.status_var, 
                                    bg="#1e1e1e", fg="gray", font=("Consolas", 10), wraplength=280)
        self.lbl_status.pack(side=tk.BOTTOM, pady=10)

        #Plot
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6,6), dpi=100)
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_right)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.init_plot_screen()
        
        self.calc_vx = 0; self.calc_vy = 0; self.calc_x = 0

    #HELPERS
    def create_label(self, text, font=("Arial", 10), color="white"):
        lbl = tk.Label(self.frame_controls, text=text, bg="#1e1e1e", fg=color, font=font)
        lbl.pack(anchor="w", pady=(5,0))
        return lbl

    def create_input(self, label_text, default_val):
        self.create_label(label_text)
        entry = tk.Entry(self.frame_controls, bg="#333", fg="white", insertbackground="white")
        entry.insert(0, default_val)
        entry.pack(fill=tk.X, pady=(0, 10))
        return entry

    def create_separator(self):
        ttk.Separator(self.frame_controls, orient='horizontal').pack(fill='x', pady=10)

    def init_plot_screen(self):
        self.ax.clear()
        self.ax.set_title("Waiting for telemetry...", color="gray")
        self.ax.text(0.5, 0.5, "C3SIM 2.0\nTHREE BODY PROBLEM", ha='center', color='gray', fontsize=14, fontweight='bold')
        self.ax.axis('off')
        self.canvas.draw()

    def safe_stop_animation(self):
        if self.ani is not None:
            try:
                self.ani.pause()
                if hasattr(self.ani, 'event_source') and self.ani.event_source:
                    self.ani.event_source.stop()
            except Exception: pass
            self.ani = None

    def reset_defaults(self):
        self.safe_stop_animation()
        entries = [self.entry_x, self.entry_y, self.entry_vx, self.entry_vy, self.entry_days]
        defaults = ["-11500", "0", "0", "-10.0", "8"]
        for entry, val in zip(entries, defaults):
            entry.delete(0, tk.END); entry.insert(0, val)
        self.btn_replay.config(state=tk.DISABLED)
        self.cached_trajectory = None
        self.init_plot_screen()
        self.status_var.set("Systems reset.")

    #Calculator
    def calculate_circular_velocity(self):
        try:
            h = float(self.entry_alt.get())
            if h <= 0: raise ValueError
        except:
            self.lbl_calc_res.config(text="Error: Invalid Altitude", fg="red")
            return

        r_orbit = R_EARTH_REAL + h
        v_circ = np.sqrt(MU_EARTH_REAL / r_orbit)

        pos_earth_x = -MU * L_STAR 
        vel_earth_y = -MU * V_STAR 

        self.calc_x = pos_earth_x - r_orbit
        self.calc_vx = 0.0
        self.calc_vy = vel_earth_y - v_circ

        self.lbl_calc_res.config(text=f"Vorb: {v_circ:.3f} km/s", fg="#00ffcc")
        self.btn_autofill.config(state=tk.NORMAL)

    def apply_calculation(self):
        self.entry_x.delete(0, tk.END); self.entry_x.insert(0, f"{self.calc_x:.1f}")
        self.entry_y.delete(0, tk.END); self.entry_y.insert(0, "0")
        self.entry_vx.delete(0, tk.END); self.entry_vx.insert(0, f"{self.calc_vx:.1f}")
        self.entry_vy.delete(0, tk.END); self.entry_vy.insert(0, f"{self.calc_vy:.4f}")
        
        r = abs(float(self.entry_alt.get()) + R_EARTH_REAL)
        period_s = 2 * np.pi * np.sqrt(r**3 / MU_EARTH_REAL)
        days_rec = (period_s * 2.1) / 86400 
        
        self.entry_days.delete(0, tk.END)
        self.entry_days.insert(0, f"{days_rec:.3f}")
        self.status_var.set(f"Loaded params for {float(self.entry_alt.get()):.0f}km altitude.")

    #Simulation
    def run_simulation(self):
        self.safe_stop_animation()
        self.ax.clear()
        
        try:
            x0_km = float(self.entry_x.get())
            y0_km = float(self.entry_y.get())
            vx0_kms = float(self.entry_vx.get())
            vy0_kms = float(self.entry_vy.get())
            days = float(self.entry_days.get())
            if days > 45: raise ValueError("Limit: 45 days")
        except ValueError as e:
            messagebox.showerror("Error", f"{e}")
            return

        #Normalization 
        x0, y0 = x0_km / L_STAR, y0_km / L_STAR
        vx0, vy0 = vx0_kms / V_STAR, vy0_kms / V_STAR
        t_span = days * 24 * 3600 / T_STAR

        # >>> ALTITUDE CHECK (100km SAFETY BUFFER) <<<
        # Posición de la Tierra en t=0 (en unidades normalizadas)
        pe0, _ = pos_bodies_inertial(0)
        
        # Distancia nave - Tierra
        dist_n = np.linalg.norm(np.array([x0, y0]) - pe0)
        
        # Distancia mínima segura (Radio Tierra + 100km) normalizada
        min_safe_dist_n = (R_EARTH_REAL + 100.0) / L_STAR
        
        if dist_n < min_safe_dist_n:
            messagebox.showerror("LAUNCH ABORTED", 
                "⚠️ ALTITUDE TOO LOW!\n\n"
                "The spacecraft is inside the Earth or below the 100km safety limit.\n"
                "Please increase altitude.")
            # Dibujar un aviso en la pantalla también
            self.ax.text(0.5, 0.5, "LAUNCH ABORTED\nALTITUDE < 100km", 
                        ha='center', color='red', fontsize=14, fontweight='bold')
            self.canvas.draw()
            return
        # >>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<
        
        #Hyperbolic Orbit Trigger
        v_mag = np.sqrt(vx0**2 + vy0**2)
        r_mag = np.sqrt(x0**2 + y0**2)
        
        #If E > 0 => Hyperbolic
        specific_energy = 0.5 * v_mag**2 - (1-MU)/r_mag
        
        if specific_energy > 0:
            self.initial_energy_msg = "⚠️ HIGH ENERGY"
            self.initial_energy_col = "orange"
        else:
            self.initial_energy_msg = "✅ ORBIT INJECTION"
            self.initial_energy_col = "#00ff00"

        self.status_var.set("Calculating trajectory...")
        self.lbl_status.config(fg="yellow")
        self.root.update()

        #SOLVER
        t_eval = np.linspace(0, t_span, 500) 
        try:
            sol = solve_ivp(equations_inertial, (0, t_span), [x0, y0, vx0, vy0],
                            method='RK45', rtol=1e-9, atol=1e-12, max_step=0.005,
                            t_eval=t_eval, events=[crash_earth, crash_moon])
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        X_traj = sol.y[0]; Y_traj = sol.y[1]; T_traj = sol.t
        
        #END
        final_msg = "🏁 SIMULATION FINISHED"
        final_col = "white"
        
        if len(sol.t_events[0]) > 0:
            final_msg = "💥 IMPACT: EARTH"
            final_col = "#ff5555"
        elif len(sol.t_events[1]) > 0:
            final_msg = "💥 IMPACT: MOON"
            final_col = "#ff5555"
        else:
            pe, pm = pos_bodies_inertial(T_traj[-1])
            last_pos = np.array([X_traj[-1], Y_traj[-1]])
            if np.linalg.norm(last_pos - pe) > R_SOI_E_N:
                final_msg = "🚀 ESCAPE TRAJECTORY"
                final_col = "orange"
            else:
                final_msg = "✅ ORBIT STABLE"
                final_col = "#55ff55"

        self.final_status_msg = final_msg
        self.final_status_col = final_col

        #Events Handler
        event_markers = [] 
        prev_state = "EARTH"
        for i in range(len(T_traj)):
            pos = np.array([X_traj[i], Y_traj[i]])
            pe, pm = pos_bodies_inertial(T_traj[i])
            curr = "EARTH"
            if np.linalg.norm(pos - pm) < R_SOI_M_N: curr = "MOON"
            elif np.linalg.norm(pos - pe) > R_SOI_E_N: curr = "ESCAPE"
            
            if curr != prev_state:
                mk = "v" if curr == "MOON" else "^" if prev_state == "MOON" else "*" if curr == "ESCAPE" else ""
                if mk: event_markers.append((i, X_traj[i], Y_traj[i], mk))
            prev_state = curr

        self.cached_trajectory = (X_traj, Y_traj, T_traj)
        self.cached_events = event_markers
        self.cached_params = (x0, y0)
        self.btn_replay.config(state=tk.NORMAL)

        self.start_animation(X_traj, Y_traj, T_traj, event_markers, x0, y0)

    def replay_simulation(self):
        if self.cached_trajectory is None: return
        self.safe_stop_animation()
        self.ax.clear()
        X, Y, T = self.cached_trajectory
        evs = self.cached_events
        x0, y0 = self.cached_params
        self.status_var.set("Replaying mission data...")
        self.lbl_status.config(fg="cyan")
        self.start_animation(X, Y, T, evs, x0, y0)

    def start_animation(self, X_traj, Y_traj, T_traj, event_markers, x0, y0):
        self.ax.add_patch(plt.Circle((0,0), R_SOI_E_N, color='#1f77b4', ls=':', fill=False, alpha=0.4))
        self.ax.plot(x0, y0, 'x', color='gold', ms=7, zorder=20, label='Start')

        earth_c = plt.Circle((0,0), R_EARTH_N, color='#1f77b4', label='Earth', zorder=5) 
        moon_c = plt.Circle((0,0), R_MOON_N, color='white', label='Moon', zorder=5)
        soi_m_c = plt.Circle((0,0), R_SOI_M_N, color='white', ls='--', fill=False, alpha=0.3)
        self.ax.add_patch(earth_c); self.ax.add_patch(moon_c); self.ax.add_patch(soi_m_c)

        ship_dot, = self.ax.plot([], [], 'o', color='cyan', ms=4, zorder=10)
        trail_lc = LineCollection([], linewidths=1.2, colors='cyan', alpha=0.6, zorder=9)
        self.ax.add_collection(trail_lc)
        lagrange_dots, = self.ax.plot([], [], 'x', color='#00ff00', ms=8, alpha=0.6)
        
        scats = [self.ax.plot([], [], m, color=c, ms=8, zorder=15, linestyle="None")[0] 
                for m, c in [('v', 'yellow'), ('^', 'orange'), ('*', 'red')]]

        time_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, color='white', fontweight='bold')
        
        warn_text = self.ax.text(0.5, 0.95, self.initial_energy_msg, transform=self.ax.transAxes, 
                                ha='center', color=self.initial_energy_col, fontweight='bold', 
                                bbox=dict(facecolor='black', alpha=0.7))
        LIMIT = 1.5 
        self.ax.set_xlim(-LIMIT, LIMIT); self.ax.set_ylim(-LIMIT, LIMIT)
        self.ax.set_aspect('equal'); self.ax.grid(True, ls=':', alpha=0.2)
        self.ax.legend(loc='lower right', facecolor='black', fontsize=8)

        def update(frame):
            t = T_traj[frame]
            pe, pm = pos_bodies_inertial(t)
            
            earth_c.center = pe; moon_c.center = pm; soi_m_c.center = pm
            Lx, Ly = rotate_points(L_loc_x, L_loc_y, t)
            lagrange_dots.set_data(Lx, Ly)
            ship_dot.set_data([X_traj[frame]], [Y_traj[frame]])
            
            if np.linalg.norm([X_traj[frame]-pm[0], Y_traj[frame]-pm[1]]) < R_SOI_M_N:
                ship_dot.set_color('white')
            else:
                ship_dot.set_color('cyan')

            if frame > 1:
                pts = np.array([X_traj[:frame], Y_traj[:frame]]).T.reshape(-1, 1, 2)
                trail_lc.set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))

            for j, (marker, color) in enumerate([('v','yellow'), ('^','orange'), ('*','red')]):
                xs, ys = [], []
                for ev in event_markers:
                    if ev[0] <= frame and ev[3] == marker:
                        xs.append(ev[1]); ys.append(ev[2])
                scats[j].set_data(xs, ys)

            time_text.set_text(f'T: {t * T_STAR / 86400:.1f} d')
            
            if frame == len(T_traj) - 1:
                warn_text.set_text(self.final_status_msg)
                warn_text.set_color(self.final_status_col)
                self.status_var.set(self.final_status_msg)
                self.lbl_status.config(fg=self.final_status_col)

            return earth_c, moon_c, ship_dot, trail_lc, lagrange_dots, time_text, warn_text

        self.ani = FuncAnimation(self.fig, update, frames=len(T_traj), interval=30, blit=False, repeat=False)
        self.canvas.draw()
#------------------#


if __name__ == "__main__":
    root = tk.Tk()
    app = SerenitasApp(root)
    root.mainloop()
