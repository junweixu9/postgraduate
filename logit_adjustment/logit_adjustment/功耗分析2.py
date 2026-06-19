import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

t = np.linspace(0, 500, 1200)

def make_base_waveform(t):
    y = (
        72 * np.sin(2 * np.pi * (t + 8) / 82)
        + 36 * np.sin(2 * np.pi * (t + 27) / 46)
        + 22 * np.sin(2 * np.pi * (t + 5) / 150)
    )
    envelope = (
        0.85
        + 0.22 * np.exp(-((t - 100) / 70) ** 2)
        + 0.18 * np.exp(-((t - 365) / 55) ** 2)
    )
    return y * envelope + 50

base = make_base_waveform(t)

poi = 365
target_window = (340, 385)

num_traces = 4
traces = []

for _ in range(num_traces):
    jitter = np.random.normal(0, 0.9)
    y = np.interp(t + jitter, t, base, left=base[0], right=base[-1])

    local = np.exp(-((t - poi) / 20) ** 2) * np.sin(2 * np.pi * (t - poi) / 24)
    y += np.random.normal(0, 9) * local
    y += np.random.normal(0, 2.2, size=t.size)

    traces.append(y)

stack = np.concatenate(traces)
traces = [(y - stack.min()) * 300 / (stack.max() - stack.min()) - 100 for y in traces]

fig, ax = plt.subplots(figsize=(9.2, 4.2))

for i, y in enumerate(traces):
    ax.plot(t, y, color="black", linewidth=1.7 if i == 0 else 1.3, alpha=0.95 if i == 0 else 0.72)

ax.axvline(poi, color="red", linewidth=2.2)
ax.axvspan(target_window[0], target_window[1], color="gray", alpha=0.10)

ax.set_xlim(0, 500)
ax.set_ylim(-100, 205)
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Voltage [mV]")
ax.tick_params(direction="in", top=True, right=True)

plt.tight_layout()
plt.show()