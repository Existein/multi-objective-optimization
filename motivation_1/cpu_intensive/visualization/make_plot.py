import pandas as pd
import matplotlib.pyplot as plt

# Load the provided CSV file
file_path = '../averaged_cpu_intensive_log_with_cost.csv'
df = pd.read_csv(file_path)

# Plot 1: Memory Size (MB) vs Execution Time (ms)
plt.figure(figsize=(10, 6))
plt.plot(df['MemorySize (MB)'], df['Duration (ms)'], marker='o', linestyle='-', color='b')
plt.title('Memory Size vs Execution Time')
plt.xlabel('Memory Size (MB)')
plt.ylabel('Execution Time (ms)')
plt.grid(True)
plt.savefig('memory_vs_execution_time.png')

# Plot 2: Memory Size (MB) vs Cost
plt.figure(figsize=(10, 6))
plt.plot(df['MemorySize (MB)'], df['avg. cost'], marker='o', linestyle='-', color='g')
plt.title('Memory Size vs Cost')
plt.xlabel('Memory Size (MB)')
plt.ylabel('Cost ($)')
plt.grid(True)
plt.savefig('memory_vs_cost.png')

plt.show()
