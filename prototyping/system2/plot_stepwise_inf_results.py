import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data
with open("eval_data.json", 'r') as file:
    data = json.load(file)

df = pd.DataFrame({
    'correct_dist': data['correct_dist'],
    'percentage_correct_dist': data['percentage_correct_dist'],
    'example_height': data['example_height']
})

operation_failed_dist = data['operation_failed_dist']

df_operation_failed = pd.DataFrame(list(data['operation_failed_dist'].items()), columns=['Operation', 'Failure_Count'])

with pd.ExcelWriter('evaluation_data.xlsx') as writer:
    df.to_excel(writer, sheet_name='Main Data', index=False)
    df_operation_failed.to_excel(writer, sheet_name='Operation Failures', index=False)

print("Excel file 'evaluation_data.xlsx' created successfully.")


# 1. Distribution of correct answers with the height of the example
plt.figure(figsize=(10, 6))
plt.scatter(df['example_height'], df['correct_dist'], alpha=0.6)
plt.xlabel('Height of Example')
plt.ylabel('Correct Answers')
plt.title('Distribution of Correct Answers by Example Height')
plt.show()

# 2. Distribution of the percentage of examples predicted accurately
df['percentage_correct_dist_binned'] = pd.cut(df['percentage_correct_dist'], bins=10)
accuracy_distribution = df['percentage_correct_dist_binned'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
accuracy_distribution.plot(kind='bar')
plt.xlabel('Percentage Accuracy Bins')
plt.ylabel('Number of Examples')
plt.title('Distribution of Prediction Accuracy')
plt.show()

# 3. Bar chart showing which operations failed the most
plt.figure(figsize=(10, 6))
plt.bar(operation_failed_dist.keys(), operation_failed_dist.values())
plt.xlabel('Operations')
plt.ylabel('Number of Failures')
plt.title('Operation Failure Frequency')
plt.show()

# 4. Chart showing the most common operator failure at different depths
df['failed_operation_at_depth'] = df['example_height'].map(lambda x: data['operation_failed_dist'].get(x, 'Unknown'))

depth_failure_distribution = df['failed_operation_at_depth'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
depth_failure_distribution.plot(kind='bar')
plt.xlabel('Depth')
plt.ylabel('Most Common Failed Operation')
plt.title('Common Failed Operation at Different Depths')
plt.show()


import time
time.sleep(100)