import matplotlib.pyplot as plt
import numpy as np

# Data (Avg ACE %)
years = ['2011', '2013', '2015', '2017']
our_ace = [1.65, 0.47, 2.99, 0.59]  # Proposed FAL-Net
sota_ace = [1.67, 0.16, 1.76, 1.91] # SOTA Benchmarks (Chugh 2018, ExpressNet, FLDNet, arfnet)

# Colors (professional palette)
color_ours = '#d62728'  # IEEE Red
color_sota = '#1f77b4'  # IEEE Blue

plt.figure(figsize=(8, 5))

# Plot lines
plt.plot(years, sota_ace, 'o--', label='LivDet SOTA Benchmarks', color=color_sota, linewidth=2, markersize=8)
plt.plot(years, our_ace, 'o-', label='FAL-Net (Proposed)', color=color_ours, linewidth=3, markersize=10)

# Fill between to show gap
# plt.fill_between(years, our_ace, sota_ace, color='lightgray', alpha=0.3)

# Annotations (relative ACE reduction)
for i, (ours, sota) in enumerate(zip(our_ace, sota_ace)):
    if ours < sota:
        reduction = (sota - ours) / sota * 100
        plt.annotate(f'-{reduction:.1f}%', (years[i], ours), 
                     textcoords="offset points", xytext=(0,-20), ha='center', 
                     color=color_ours, fontweight='bold', fontsize=10)

# Formatting
plt.title('Performance Evolution Across LivDet Dataset Progression', fontsize=14, fontweight='bold')
plt.xlabel('LivDet Dataset Year', fontsize=12)
plt.ylabel('Average Classification Error (ACE %)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='upper right')
plt.ylim(0, 5)

# Tight layout
plt.tight_layout()

# Save figure
save_path = r'd:\project_saoudia_work\realfake2017_Lora\IEEEtran\figures\performance_evolution.png'
plt.savefig(save_path, dpi=300)
print(f"Figure saved to {save_path}")
plt.show()
