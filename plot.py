## Plots box plot of episodic returns

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'


filename = 'results/sim_episodic_returns.txt'

data_sim = np.loadtxt(filename)
filename = 'results/real_episodic_returns.txt'
data_real = np.loadtxt(filename)
plt.figure(figsize=(10, 6))
plt.boxplot([data_sim, data_real] )  # Outliers


# Add labels and title
plt.title('Sim vs Real Episodic Returns', fontsize=24)
plt.xticks([1, 2], ['Sim', 'Real'], fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Returns', fontsize=24)
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# Save the plot to a file or display it
plt.savefig('returns_box_plot.png')  # Save the plot as an image
