import pandas as pd

# Load the provided CSV file
df = pd.read_csv('io_intensive_log.csv')  # Replace with your file path

# Ensure the columns are strings before applying string operations
df['Duration'] = df['Duration'].astype(str).str.replace(' m', '').astype(float).round(2)
df['BilledDuration'] = df['BilledDuration'].astype(str).str.replace(' ms', '').astype(int)
df['MemorySize'] = df['MemorySize'].astype(str).str.replace(' MB', '').astype(int)
df['MaxMemoryUsed'] = df['MaxMemoryUsed'].astype(str).str.replace(' MB', '').astype(int)

# Group by LogStreamName and calculate the average for each column
df_avg = df.groupby('LogStreamName').mean()

# Round Duration column to 2 decimal places
df_avg['Duration'] = df_avg['Duration'].round(2)

# Remove decimal points
df_avg['BilledDuration'] = df_avg['BilledDuration'].astype(int)
df_avg['MemorySize'] = df_avg['MemorySize'].astype(int)
df_avg['MaxMemoryUsed'] = df_avg['MaxMemoryUsed'].astype(int)

# Calculate avg. cost
df_avg['avg. cost'] = 0.0000002 + 0.00001667 * (df_avg['MemorySize'] / 1024) * df_avg['BilledDuration']

# Update the header to reflect the units
df_avg.columns = ['Duration (ms)', 'BilledDuration (ms)', 'MemorySize (MB)', 'MaxMemoryUsed (MB)', 'avg. cost']

# Sort by MemorySize in ascending order
df_avg = df_avg.sort_values(by='MemorySize (MB)')

# Save the result to a new CSV file
df_avg.to_csv('averaged_io_intensive_log_with_cost.csv', index=True)

print("Averaged log has been saved.")
