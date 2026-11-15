import json
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# Cargar el JSON generado
with open("salida_fase2_vectorizada.json", "r", encoding="utf-8") as f:
    datos = json.load(f)

# Tomar el primer Epoch y la primera banda como ejemplo
epoch = datos["epochs"][0]
banda = epoch["bandas"][0]
fuentes = banda["fuentes"]

fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
ax.set_facecolor("black")

# Extraer coordenadas y flujos para escalar colores/tamaños
ras = [obj["ra"] for obj in fuentes]
decs = [obj["dec"] for obj in fuentes]
flujos = np.array([max(obj["flujo"], 1e-3) for obj in fuentes])
log_flujos = np.log10(flujos)

# Dibujar cada fuente como una elipse según su morfología
for obj, lf in zip(fuentes, log_flujos):
    color = "cyan" if obj["tipo_morfologia"] == "extendido" else "gold"
    # Escala visual para que los semiejes en píxeles sean visibles en grados
    escala = 0.0005 
    elipse = Ellipse(
        xy=(obj["ra"], obj["dec"]),
        width=obj["semieje_a_px"] * escala * 2,
        height=obj["semieje_b_px"] * escala * 2,
        angle=np.degrees(obj["angulo_rad"]),
        edgecolor=color,
        facecolor=color,
        alpha=0.7
    )
    ax.add_patch(elipse)

ax.scatter(ras, decs, s=5, c="white", alpha=0.8)
ax.set_xlabel("Ascensión Recta (RA) [deg]", color="white")
ax.set_ylabel("Declinación (Dec) [deg]", color="white")
ax.set_title(f"Epoch {epoch['epoch_id']} | Banda {banda['banda']} ({banda['lambda_um']:.2f} µm)", color="white")
ax.tick_params(colors="white")
ax.invert_xaxis() # En astronomía el Este (RA creciente) va hacia la izquierda

plt.tight_layout()
plt.savefig("mapa_fuentes.png", dpi=200)
plt.show()