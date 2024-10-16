import pandas as pd

# Load the provided CSV file
df = pd.read_csv('../averaged_mem_intensive_log_with_cost.csv')

# Find the configuration with the lowest cost
min_cost_row = df.loc[df['avg. cost'].idxmin()]

# Calculate the 10% margin cost
margin_cost = min_cost_row['avg. cost'] * 1.10

# Find the configuration with the lowest duration within the margin cost
within_margin_df = df[df['avg. cost'] <= margin_cost]
min_duration_row = within_margin_df.loc[within_margin_df['Duration (ms)'].idxmin()]

# Calculate the percentage improvement in execution time
execution_time_improvement = ((min_cost_row['Duration (ms)'] - min_duration_row['Duration (ms)']) / min_cost_row['Duration (ms)']) * 100

# Print the results
print("Configuration with the lowest cost:")
print(min_cost_row)
print("\nConfiguration with the lowest duration within 10% margin cost:")
print(min_duration_row)
print(f"\nPercentage improvement in execution time: {execution_time_improvement:.2f}%")

# Write the results to a text file
with open('results.txt', 'w') as file:
    file.write("Configuration with the lowest cost:\n")
    file.write(min_cost_row.to_string())
    file.write("\n\nConfiguration with the lowest duration within 10% margin cost:\n")
    file.write(min_duration_row.to_string())
    file.write(f"\n\nPercentage improvement in execution time: {execution_time_improvement:.2f}%\n")