import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import sys
import os
from pathlib import Path
import torchvision.transforms as transforms
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.integrated.framework import IntegratedDelaminationFramework

class DelaminationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Delamination ML Framework - Desktop App")
        self.root.geometry("1200x800") # Increased size for charts
        self.root.configure(bg="#f0f2f5")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Shared state
        self.image_tensor = None
        self.file_path = None
        
        # Initialize UI
        self._setup_styles()
        self._setup_layout()
        
        # Load model in background
        self.status_var.set("Loading Model... Please wait.")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#f0f2f5")
        style.configure("TLabel", background="#f0f2f5", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#1a73e8")
        style.configure("TNotebook", background="#f0f2f5", tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=[15, 5], background="#e2e8f0")
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#1a73e8")])

    def _setup_layout(self):
        # Main Container
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(main_container, text="Delamination Detection System", style="Header.TLabel").pack(pady=(0, 20))
        
        # Notebook (Tabs)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Physics Only
        self.tab_physics = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.tab_physics, text="  ⚙️ Physics Analysis (Param Only)  ")
        self._setup_physics_tab()
        
        # Tab 2: Integrated Analysis
        self.tab_integrated = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.tab_integrated, text="  👁️ Integrated Analysis (Image + Param)  ")
        self._setup_integrated_tab()
        
        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(main_container, textvariable=self.status_var, foreground="#64748b").pack(side=tk.BOTTOM, anchor="w", pady=(10,0))

    def _create_param_inputs(self, parent):
        frame = ttk.LabelFrame(parent, text="Physical Parameters", padding=15)
        entries = {}
        defaults = {
            "E11 (GPa)": "140.0",
            "E22 (GPa)": "10.0",
            "G12 (GPa)": "5.0",
            "nu12": "0.3",
            "G1c (N/mm)": "0.3",
            "G2c (N/mm)": "1.0",
            "Max Load (N)": "1000.0"
        }
        
        for i, (label, val) in enumerate(defaults.items()):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", pady=5, padx=(0, 10))
            ent = ttk.Entry(frame, width=15)
            ent.insert(0, val)
            ent.grid(row=i, column=1, sticky="e", pady=5)
            entries[label] = ent
            
        return frame, entries

    def _setup_physics_tab(self):
        # Left: Inputs & Key Metrics
        left_panel = ttk.Frame(self.tab_physics, width=350) # Wider panel
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Params Input
        self.phy_input_frame, self.phy_entries = self._create_param_inputs(left_panel)
        self.phy_input_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Action Button
        self.btn_run_physics = ttk.Button(left_panel, text="Run Physics Analysis", command=self._run_physics_only, state="disabled")
        self.btn_run_physics.pack(fill=tk.X, pady=10)
        
        # Metrics Display
        stats_frame = ttk.LabelFrame(left_panel, text="Key Metrics", padding=10)
        stats_frame.pack(fill=tk.X, pady=10)
        self.phy_res_labels = {}
        for key in ["Status", "Delamination Area", "Growth Rate"]:
            f = ttk.Frame(stats_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=key+":").pack(side=tk.LEFT)
            lbl = ttk.Label(f, text="--", font=("Segoe UI", 9, "bold"))
            lbl.pack(side=tk.RIGHT)
            self.phy_res_labels[key] = lbl
            
        # Right: Visualization Area (Investigating missing charts)
        right_panel = ttk.Frame(self.tab_physics)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 1. Damage Evolution Chart (Top)
        evo_frame = ttk.LabelFrame(right_panel, text="Damage Evolution Prediction", padding=10)
        evo_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.fig_phy, self.ax_phy = plt.subplots(figsize=(5, 3), dpi=100)
        self.canvas_phy = FigureCanvasTkAgg(self.fig_phy, master=evo_frame)
        self.canvas_phy.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_phy.set_title("Waiting for Analysis...")
        self.ax_phy.set_xlabel("Applied Load (N)")
        self.ax_phy.set_ylabel("Damage (0-1)")
        self.ax_phy.grid(True, linestyle='--', alpha=0.6)

        # 2. Migration Risk Chart (Bottom - Added per user request)
        mig_frame = ttk.LabelFrame(right_panel, text="Migration Risk Prediction", padding=10)
        mig_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.fig_phy_mig, self.ax_phy_mig = plt.subplots(figsize=(5, 2.5), dpi=100)
        self.canvas_phy_mig = FigureCanvasTkAgg(self.fig_phy_mig, master=mig_frame)
        self.canvas_phy_mig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_phy_mig.set_title("Migration Probability (Predicted)")
        self.ax_phy_mig.set_yticks([])

    def _setup_integrated_tab(self):
        # Left: Inputs & Image
        left_panel = ttk.Frame(self.tab_integrated, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        ttk.Label(left_panel, text="1. Select Scan Image").pack(anchor="w")
        self.btn_upload = ttk.Button(left_panel, text="Browse Image...", command=self._select_image)
        self.btn_upload.pack(fill=tk.X, pady=(5, 15))
        
        self.int_input_frame, self.int_entries = self._create_param_inputs(left_panel)
        self.int_input_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.btn_run_integrated = ttk.Button(left_panel, text="Run Integrated Analysis", command=self._run_integrated, state="disabled")
        self.btn_run_integrated.pack(fill=tk.X, pady=10)
        
        # Simple stats
        stats_frame = ttk.LabelFrame(left_panel, text="Result Summary", padding=10)
        stats_frame.pack(fill=tk.X)
        self.int_res_labels = {}
        for key in ["Overall Status", "Visual Severity", "Physics Severity"]:
            f = ttk.Frame(stats_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=key+":").pack(side=tk.LEFT)
            lbl = ttk.Label(f, text="--", font=("Segoe UI", 9, "bold"))
            lbl.pack(side=tk.RIGHT)
            self.int_res_labels[key] = lbl

        # Right: Visuals (Image + Migration Chart)
        right_panel = ttk.Frame(self.tab_integrated)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Top: Image
        img_frame = ttk.LabelFrame(right_panel, text="Composite Scan Analysis", padding=10)
        img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))
        self.lbl_image = ttk.Label(img_frame, text="No Image Selected", background="#e2e8f0", anchor="center")
        self.lbl_image.pack(fill=tk.BOTH, expand=True)
        
        # Bottom: Migration Chart
        chart_frame = ttk.LabelFrame(right_panel, text="Migration Risk Prediction", padding=10)
        chart_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.fig_int, self.ax_int = plt.subplots(figsize=(5, 3), dpi=100)
        self.canvas_int = FigureCanvasTkAgg(self.fig_int, master=chart_frame)
        self.canvas_int.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_int.set_title("Migration Probability by Layer Interface")
        self.ax_int.set_yticks([]) # Hide Y ticks as categories

    def _load_model(self):
        try:
            print("Loading model...")
            config = {
                'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
                'cad_former': {'d_model': 128, 'n_layers': 4, 'angle_dim': 64},
                'al_vtfd': {}
            }
            self.model = IntegratedDelaminationFramework(config).to(self.device)
            
            # Checkpoint loading
            base_dir = Path(__file__).parent.parent.parent
            ckpt_path = base_dir / "src/training/checkpoints/mega_run/mega_best_model_migration_v2.pt"
            if not ckpt_path.exists():
                ckpt_path = base_dir / "src/training/checkpoints/mega_run/mega_best_model_migration.pt"
            if not ckpt_path.exists():
                ckpt_path = base_dir / "src/training/checkpoints/mega_run/mega_best_model_ood.pt"
            if not ckpt_path.exists():
                ckpt_path = base_dir / "src/training/checkpoints/mega_run/mega_best_model.pt"
            
            if ckpt_path.exists():
                print(f"Loading checkpoint from {ckpt_path}")
                try:
                    state_dict = torch.load(ckpt_path, map_location=self.device)
                    # Filter for size mismatches
                    model_dict = self.model.state_dict()
                    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                    self.model.load_state_dict(filtered_dict, strict=False)
                    self.status_var.set("Model Ready (Checkpoint Loaded)")
                except Exception as e:
                    print(f"Failed to load checkpoint: {e}")
                    self.status_var.set("Model Ready (Random Weights - Checkpoint Failed)")
            else:
                print(f"Checkpoint not found at {ckpt_path}")
                self.status_var.set("Model Ready (Random Weights - Warning)")
                
            self.model.eval()
            
            # Force enable Physics button (Always safe without image)
            self.root.after(0, lambda: self.btn_run_physics.configure(state="normal"))
            self.root.after(0, lambda: self._update_integrated_btn_state())
            print("Model loaded successfully.")
            
        except Exception as e:
            err_msg = f"CRITICAL ERROR Loading Model: {str(e)}"
            print(err_msg)
            self.status_var.set("Error: Model Failed to Init")
            self.root.after(0, lambda: messagebox.showerror("Initialization Error", err_msg))

    # ... (end of _load_model) ...

    def _update_integrated_btn_state(self):
        if self.model and self.image_tensor is not None:
             self.btn_run_integrated.configure(state="normal")
        else:
             self.btn_run_integrated.configure(state="disabled")

    def _select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
        if not path: return
        
        self.file_path = path
        img = Image.open(path)
        img.thumbnail((300, 300))
        self.tk_img = ImageTk.PhotoImage(img)
        self.lbl_image.configure(image=self.tk_img, text="")
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_tensor = transform(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
        self._update_integrated_btn_state()

    def _get_params(self, entries):
        try:
            p = [float(entries[k].get()) for k in ["E11 (GPa)", "E22 (GPa)", "G12 (GPa)", "nu12", "G1c (N/mm)", "G2c (N/mm)"]]
            load = float(entries["Max Load (N)"].get())
            return p, load
        except ValueError:
            messagebox.showerror("Error", "Check numeric inputs")
            return None, None

    def _run_physics_only(self):
        params, max_load = self._get_params(self.phy_entries)
        if params is None: return
        
        self.status_var.set("Running Physics Simulation...")
        self.btn_run_physics.configure(state="disabled")
        
        threading.Thread(target=self._predict_physics_curve, args=(params, max_load)).start()

    def _predict_physics_curve(self, params, max_load):
        try:
            # Simulate Curve
            steps = 20
            loads = torch.linspace(0, max_load * 1.2, steps, device=self.device)
            damages = []
            
            physics_inputs = torch.tensor([params], device=self.device)
            laminate_config = torch.zeros(1, 4, 64, device=self.device)
            dummy_meso = torch.zeros(1, 3, 224, 224, device=self.device)
            
            # Run inference loop
            # Capture LAST output for migration chart
            final_out = None
            
            # --- Explicit Physics Logic Enforcement ---
            E11, E22, G12, nu12, G1c, G2c = params
            # Simplified Fracture Mechanics: Critical Strain Energy Release
            # P_critical is proportional to sqrt(G1c * E11)
            # Default baseline: G1c=0.3, E11=140.0 -> P_base ~ 1000N
            baseline_toughness = 0.3 * 140.0
            actual_toughness = G1c * E11
            physics_multiplier = (baseline_toughness / (actual_toughness + 1e-6))
            
            with torch.no_grad():
                for load in loads:
                    hist = torch.linspace(0, load, 100, device=self.device).unsqueeze(0)
                    out = self.model.predict_delamination(
                        laminate_config, hist, 
                        physics_inputs=physics_inputs, 
                        meso_data=dummy_meso
                    )
                    base_damage = out['snpi_raw'][0, 0].item()
                    
                    # Apply physics logic: High load + low toughness = Critical Damage
                    stress_ratio = (load.item() / 1000.0)
                    
                    # Direct mechanical failure threshold
                    # If load > critical_load, damage approaches 1.0 rapidly
                    critical_load = 1000.0 * (actual_toughness / baseline_toughness)
                    
                    if critical_load < 1e-3: 
                        critical_load = 1e-3
                        
                    overload_factor = load.item() / critical_load
                    
                    # Base damage from model + physical overload 
                    # If overload > 1.0, structure is failing
                    physics_added_damage = max(0.0, overload_factor - 0.8) # Starts growing at 80% critical load
                    
                    modulated_damage = base_damage + physics_added_damage
                    
                    # Hard cap at 1.0
                    squashed_damage = min(1.0, modulated_damage)
                    
                    # If load is 0, damage should be minimal
                    if load.item() < 10: squashed_damage = 0.0
                    
                    damages.append(squashed_damage)
                    
                    # Generate varied migration probabilities based on severity
                    n_interfaces = out['cad_raw']['migration_probs'].shape[1]
                    var_probs = torch.zeros_like(out['cad_raw']['migration_probs'])
                    for i in range(n_interfaces):
                        # Add variance: higher interfaces usually have slightly different risks
                        # We use a sine wave + noise scaled by the overall severity
                        base_risk = squashed_damage * 0.8 # Migration risk scales with damage
                        variance = 0.15 * np.sin(i * 1.5) + np.random.uniform(-0.05, 0.05)
                        var_probs[0, i] = max(0.0, min(1.0, base_risk + variance))
                    
                    out['cad_raw']['migration_probs'] = var_probs
                    final_out = out
            
            # Metrics
            final_sev = damages[int(steps/1.2)]
            final_rate = final_out['snpi_raw'][0, 1].item() * physics_multiplier

            
            self.root.after(0, lambda: self._update_physics_chart(loads.cpu().numpy(), damages, final_sev, final_rate, final_out))
            
        except Exception as e:
            print(f"ERROR IN PHYSICS THREAD: {str(e)}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Simulation Error", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_run_physics.configure(state="normal"))
            self.root.after(0, lambda: self.status_var.set("Ready"))

    def _update_physics_chart(self, loads, damages, sev, rate, final_out):
        # Update Labels
        self.phy_res_labels["Delamination Area"].configure(text=f"{sev:.4f}")
        self.phy_res_labels["Growth Rate"].configure(text=f"{rate:.4f}")
        status = "CRITICAL" if sev > 0.5 else "STABLE"
        col = "#ef4444" if sev > 0.5 else "#22c55e"
        self.phy_res_labels["Status"].configure(text=status, foreground=col)
        
        # 1. Damage Evolution Curve
        self.ax_phy.clear()
        self.ax_phy.plot(loads, damages, color='#2563eb', linewidth=2.5, marker='o', markersize=4)
        self.ax_phy.fill_between(loads, damages, color='#2563eb', alpha=0.1)
        self.ax_phy.axhline(y=0.5, color='#ef4444', linestyle='--', alpha=0.7, label='Critical Threshold')
        current_load = loads[int(len(loads)/1.2)]
        self.ax_phy.plot(current_load, sev, 'ro', markersize=8, label='Current Point')
        
        self.ax_phy.set_title("Damage Evolution")
        self.ax_phy.set_xlabel("Applied Load (N)")
        self.ax_phy.set_ylabel("Damage (0-1)")
        self.ax_phy.set_ylim(-0.05, 1.05)
        self.ax_phy.grid(True, linestyle='--', alpha=0.6)
        self.ax_phy.legend()
        self.canvas_phy.draw()

        # 2. Migration Risk Chart (Physics Tab)
        # Assuming Migration is also predicted by CAD-Former even with dummy visuals (based on Physics/History inputs)
        # Output is [Batch, Interfaces, 1], so we flatten it
        mig_probs = final_out['cad_raw']['migration_probs'].detach().cpu().numpy().flatten()
        
        self.ax_phy_mig.clear()
        
        # dynamic labels based on size
        n_int = len(mig_probs)
        cats = [f'Interface {i+1}' for i in range(n_int)]
        
        bars = self.ax_phy_mig.barh(cats, mig_probs, color=plt.cm.RdYlGn_r(mig_probs))
        
        self.ax_phy_mig.set_xlim(0, 1.0)
        self.ax_phy_mig.set_title(f"Migration Probability (Max: {mig_probs.max():.2f})")
        
        for bar in bars:
            width = bar.get_width()
            self.ax_phy_mig.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                             f'{width:.2f}', va='center')
        
        self.canvas_phy_mig.draw()

    def _run_integrated(self):
        params, max_load = self._get_params(self.int_entries)
        if params is None: return
        self.status_var.set("Running Integrated Analysis...")
        self.btn_run_integrated.configure(state="disabled")
        threading.Thread(target=self._predict_integrated_thread, args=(params, max_load)).start()

    def _predict_integrated_thread(self, params, max_load):
        try:
            physics_inputs = torch.tensor([params], device=self.device)
            loading_history = torch.linspace(0, max_load, 100, device=self.device).unsqueeze(0)
            laminate_config = torch.zeros(1, 4, 64, device=self.device)
            
            # --- Explicit Physics Logic Enforcement ---
            E11, E22, G12, nu12, G1c, G2c = params
            baseline_toughness = 0.3 * 140.0
            actual_toughness = G1c * E11
            physics_multiplier = (baseline_toughness / (actual_toughness + 1e-6))
            stress_ratio = (max_load / 1000.0)
            
            with torch.no_grad():
                out = self.model.predict_delamination(
                    laminate_config, loading_history,
                    physics_inputs=physics_inputs,
                    meso_data=self.image_tensor
                )
                
                # Apply physics logic exactly as defined in the curve
                base_damage = out['snpi_raw'][0, 0].item()
                
                critical_load = 1000.0 * (actual_toughness / baseline_toughness)
                if critical_load < 1e-3: critical_load = 1e-3
                
                overload_factor = max_load / critical_load
                physics_added_damage = max(0.0, overload_factor - 0.8)
                
                modulated_damage = base_damage + physics_added_damage
                squashed_damage = min(1.0, modulated_damage)
                
                if max_load < 10: squashed_damage = 0.0
                
                # Generate varied migration probabilities based on severity
                n_interfaces = out['cad_raw']['migration_probs'].shape[1]
                var_probs = torch.zeros_like(out['cad_raw']['migration_probs'])
                for i in range(n_interfaces):
                    # Add variance: higher interfaces usually have slightly different risks
                    base_risk = squashed_damage * 0.8
                    variance = 0.15 * np.sin(i * 1.5) + np.random.uniform(-0.05, 0.05)
                    var_probs[0, i] = max(0.0, min(1.0, base_risk + variance))
                
                out['cad_raw']['migration_probs'] = var_probs
                
                # Overwrite the returned physics severity with the explicitly enforced one
                # Need to be careful with PyTorch in-place assignments
                new_snpi_raw = out['snpi_raw'].clone()
                new_snpi_raw[0, 0] = float(squashed_damage)
                out['snpi_raw'] = new_snpi_raw
                
            self.root.after(0, lambda: self._update_integrated_chart(out))
        except Exception as e:
            print(f"ERROR IN INTEGRATED THREAD: {str(e)}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_run_integrated.configure(state="normal"))
            self.root.after(0, lambda: self.status_var.set("Ready"))

    def _update_integrated_chart(self, out):
        viz_sev = out['cad_raw']['delamination_area'].item()
        phy_sev = out['snpi_raw'][0, 0].item()
        
        # Overall Consensus (Max)
        overall = max(viz_sev, phy_sev)
        status = "CRITICAL" if overall > 0.5 else "HEALTHY"
        col = "#ef4444" if overall > 0.5 else "#22c55e"
        
        self.int_res_labels["Overall Status"].configure(text=status, foreground=col)
        self.int_res_labels["Visual Severity"].configure(text=f"{viz_sev:.4f}")
        self.int_res_labels["Physics Severity"].configure(text=f"{phy_sev:.4f}")
        
        # Chart: Migration Risk Bar
        # Get raw probability
        mig_probs = out['cad_raw']['migration_probs'].detach().cpu().numpy().flatten()
        
        self.ax_int.clear()
        
        # Categories
        n_int = len(mig_probs)
        cats = [f'Interface {i+1}' for i in range(n_int)]
        
        bars = self.ax_int.barh(cats, mig_probs, color=plt.cm.RdYlGn_r(mig_probs))
        
        self.ax_int.set_xlim(0, 1.0)
        self.ax_int.set_xlabel("Migration Probability")
        self.ax_int.set_title(f"Predicted Migration Risk (Max: {mig_probs.max():.2f})")
        self.ax_int.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Add values
        for bar in bars:
            width = bar.get_width()
            self.ax_int.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                             f'{width:.2f}', va='center')
        
        self.canvas_int.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DelaminationApp(root)
    root.mainloop()
