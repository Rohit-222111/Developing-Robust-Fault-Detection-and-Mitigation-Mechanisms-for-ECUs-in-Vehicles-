import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Oxy_eff import predict as predict_oxy
from Rad_eff import predict_r as predict_rad
from Sp_eff import predict_s as predict_sp
from Tc_eff import predict_t as predict_tc

# --- Component Definitions ---
components = {
    "Oxygen Sensor": {
        "image": "oxygen.png",
        "columns": ['OxygenSensorVoltage[Volts]', 'EngineCoolantTemperature[°C]', 'AirFuelRatio[AFR]',
                    'ExhaustGasTemperature[°C]', 'EngineRPM[Speed]', 'FuelTrim[Percentage]',
                    'ThrottlePosition[Percentage]', 'IntakeAirTemperature[°C]', 'ExhaustBackPressure[PSI]'],
        "thresholds": {
            'OxygenSensorVoltage[Volts]': (0.2, 0.8),
            'EngineCoolantTemperature[°C]': (70, 90),
            'AirFuelRatio[AFR]': (14.5, 15.0),
            'ExhaustGasTemperature[°C]': (200, 400),
            'EngineRPM[Speed]': (900, 3000),
            'FuelTrim[Percentage]': (-10, 10),
            'ThrottlePosition[Percentage]': (5, 95),
            'IntakeAirTemperature[°C]': (20, 50),
            'ExhaustBackPressure[PSI]': (5, 15)
        },
        "predict_func": predict_oxy
    },
    "Radiator": {
        "image": "radiator.jpg",
        "columns": ['AmbientTemperature[°C]', 'CoolantTemperature[°C]', 'CoolantLevel[%]',
                    'CoolantFlowRate[L/min]', 'ThermostatOpening[%]', 'FanSpeed[RPM]',
                    'VehicleSpeed[km/h]', 'RadiatorPressure[PSI]'],
        "thresholds": {
            'AmbientTemperature[°C]': (16.8, 34.07),
            'CoolantTemperature[°C]': (78.79, 103.45),
            'CoolantLevel[%]': (43.25, 88.77),
            'CoolantFlowRate[L/min]': (18.31, 51.92),
            'ThermostatOpening[%]': (26.04, 83.45),
            'FanSpeed[RPM]': (1082.52, 2542.29),
            'VehicleSpeed[km/h]': (26.87, 121.44),
            'RadiatorPressure[PSI]': (8.21, 17.06)
        },
        "predict_func": predict_rad
    },
    "Sparkplug": {
        "image": "sparkplug.jpg",
        "columns": ['CylinderPressure[PSI]', 'AirFuelRatio[AFR]', 'SparkGap[mm]',
                    'SparkDuration[ms]', 'CombustionTemp[°C]', 'IgnitionTiming[°BTDC]',
                    'VoltageSupply[Volts]', 'EngineRPM[Speed]'],
        "thresholds": {
            'CylinderPressure[PSI]': (80.0, 239.99),
            'AirFuelRatio[AFR]': (11.25, 16.25),
            'SparkGap[mm]': (0.3, 1.5),
            'SparkDuration[ms]': (-0.75, 4.25),
            'CombustionTemp[°C]': (100.02, 1299.95),
            'IgnitionTiming[°BTDC]': (-10.0, 49.99),
            'VoltageSupply[Volts]': (9.25, 16.25),
            'EngineRPM[Speed]': (-2449.75, 10149.49)
        },
        "predict_func": predict_sp
    },
    "Turbocharger": {
        "image": "turbocharger.jpg",
        "columns": ["BoostPressure[PSI]", "CompressorSpeed[RPM]", "ExhaustTemperature[°C]",
                    "OilPressure[PSI]", "AirMassFlow[g/s]", "EngineLoad[%]",
                    "AmbientTemperature[°C]", "WastegatePosition[%]", "RPM[Engine]"],
        "thresholds": {
            "BoostPressure[PSI]": (8.4, 17.67),
            "CompressorSpeed[RPM]": (40618.56, 123440.07),
            "ExhaustTemperature[°C]": (504.11, 861.78),
            "OilPressure[PSI]": (28.34, 52.12),
            "AirMassFlow[g/s]": (20.69, 58.96),
            "EngineLoad[%]": (38.64, 85.35),
            "AmbientTemperature[°C]": (-0.15, 34.53),
            "WastegatePosition[%]": (17.05, 80.0),
            "RPM[Engine]": (2116.57, 5772.39)
        },
        "predict_func": predict_tc
    }
}

# Extend thresholds to faulty ranges
for comp in components.values():
    comp["faulty_ranges"] = {
        col: (
            val[0] - (val[1] - val[0]) * 0.5,
            val[1] + (val[1] - val[0]) * 0.5
        ) for col, val in comp["thresholds"].items()
    }

class ComponentFrame(tk.Frame):
    def __init__(self, parent, name, config):
        super().__init__(parent)
        self.name = name
        self.config_data = config
        self.sliders = {}
        self.status_labels = {}
        self.x_data = []
        self.y_data = []
        self.latest_slider_values = []
        self.start_time = time.time()

        self.build_ui()

    def build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Scrollable Slider Panel Setup
        left_panel = tk.Frame(self)
        left_panel.grid(row=0, column=0, sticky="ns", padx=(10, 0))

        slider_outer = tk.Frame(left_panel)
        slider_outer.pack(side="left", fill="y")

        canvas = tk.Canvas(slider_outer, width=300, height=600)
        scrollbar = ttk.Scrollbar(slider_outer, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for col in self.config_data["columns"]:
            min_val, max_val = self.config_data["faulty_ranges"][col]
            actual_min, actual_max = self.config_data["thresholds"][col]

            ttk.Label(scrollable_frame, text=col).pack()
            slider = ttk.Scale(scrollable_frame, from_=min_val, to=max_val, orient="horizontal", length=150,
                               command=self.on_slider_change)
            slider.pack()
            slider.set(min_val)
            self.sliders[col] = slider

            canvas_threshold = tk.Canvas(scrollable_frame, width=150, height=40, bg='white', highlightthickness=0)
            canvas_threshold.pack()

            range_width = max_val - min_val
            pos_min = (actual_min - min_val) / range_width * 150
            pos_max = (actual_max - min_val) / range_width * 150

            canvas_threshold.create_text(pos_min, 10, text="▼", fill="blue", font=("Arial", 10))
            canvas_threshold.create_text(pos_max, 10, text="▼", fill="blue", font=("Arial", 10))
            canvas_threshold.create_text(pos_min, 25, text=f"{actual_min:.2f}", fill="blue", font=("Arial", 8))
            canvas_threshold.create_text(pos_max, 25, text=f"{actual_max:.2f}", fill="blue", font=("Arial", 8))

            self.status_labels[col] = tk.Label(scrollable_frame, text="Status: OK", fg="green")
            self.status_labels[col].pack()

        self.latest_slider_values = [slider.get() for slider in self.sliders.values()]

        original_image = Image.open(self.config_data["image"])
        image_copy = original_image.copy()
        image_copy.thumbnail((800, 600), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image_copy)

        image_label = tk.Label(self, image=self.photo)
        image_label.grid(row=0, column=1, padx=30, sticky="nsew")

        self.graph_frame = tk.Frame(self)
        self.graph_frame.grid(row=0, column=2, padx=30)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack()

        self.update_graph()

    def on_slider_change(self, event=None):
        self.latest_slider_values = [slider.get() for slider in self.sliders.values()]
        for col in self.config_data["columns"]:
            self.update_status(col)

    def update_status(self, col):
        val = self.sliders[col].get()
        min_val, max_val = self.config_data["thresholds"][col]
        if val < min_val or val > max_val:
            self.status_labels[col].config(text="Status: Faulty", fg="red")
        else:
            self.status_labels[col].config(text="Status: Optimal", fg="green")

    def update_graph(self):
        t = (time.time() - self.start_time)
        inputs = self.latest_slider_values.copy()
        inputs.append(t / (12 * 10))
        predict_func = self.config_data.get("predict_func")
        y = predict_func(np.array(inputs).reshape(1, -1)) if predict_func else [0]

        self.x_data.append(t)
        self.y_data.append(max(0, y[0]))

        if len(self.x_data) > 100:
            self.x_data.pop(0)
            self.y_data.pop(0)

        self.ax.clear()
        self.ax.plot(self.x_data, self.y_data, label="Prediction")
        self.ax.set_title("Live Prediction")
        self.ax.set_ylabel("Performance")
        self.ax.set_xlabel("Time (months)")
        self.ax.legend()
        self.canvas.draw()

        self.after(1000, self.update_graph)

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor Dashboard")
        self.current_index = 0
        self.frames = []

        self.header = tk.Label(root, text="", font=("Arial", 20), pady=10)
        self.header.pack()

        self.container = tk.Frame(root)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.left_button_canvas = tk.Canvas(root, width=150, height=60, bg='white', highlightthickness=0)
        self.left_button_canvas.pack(side=tk.LEFT, padx=350, pady=10)
        self.draw_nav_button(self.left_button_canvas, "◀ Prev", lambda: self.switch_component(-1))

        self.right_button_canvas = tk.Canvas(root, width=150, height=60, bg='white', highlightthickness=0)
        self.right_button_canvas.pack(side=tk.RIGHT, padx=20, pady=10)
        self.draw_nav_button(self.right_button_canvas, "Next ▶", lambda: self.switch_component(1))

        for name, config in components.items():
            frame = ComponentFrame(self.container, name, config)
            frame.pack_forget()
            self.frames.append(frame)

        self.switch_component(0)

    def draw_nav_button(self, canvas, text, command):
        oval = canvas.create_oval(10, 10, 140, 50, fill='blue', outline='black')
        label = canvas.create_text(75, 30, text=text, fill='red', font=('Arial', 14, 'bold'))

        def on_click(event):
            command()

        canvas.tag_bind(oval, "<Button-1>", on_click)
        canvas.tag_bind(label, "<Button-1>", on_click)

    def switch_component(self, direction):
        self.frames[self.current_index].pack_forget()
        self.current_index = (self.current_index + direction) % len(self.frames)
        new_frame = self.frames[self.current_index]
        new_frame.pack(fill=tk.BOTH, expand=True)
        self.header.config(text=f"{self.current_index+1}.{new_frame.name}")

if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    app = MainApp(root)
    root.mainloop()