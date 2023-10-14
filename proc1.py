import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#############################################################

df=pd.read_csv('physician.csv')
# print(df)

#############################################
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
fludara_prescriptions = []
mercapto_prescriptions = []
for i in range(3,9) :
    fludara_prescriptions.append(df.iloc[:, i].sum())
for i in range(9,15) :
    mercapto_prescriptions.append(df.iloc[:, i].sum())

    


# Create the line graph
plt.figure(figsize=(10, 6))  # Optional: Set the figure size

plt.plot(months, fludara_prescriptions, label='Fludara')
plt.plot(months, mercapto_prescriptions, label='Mercapto')

# Add labels and title
plt.xlabel("Months")
plt.ylabel("Total Prescriptions")
plt.title("Prescriptions of Fludara vs. Mercapto Over Months")
plt.legend()  # Add legend

# Show the plot
plt.grid(True)  # Optional: Add grid lines
plt.show()


#############################################################
###Top 200 Physicians based on prescriptions
physician_prescriptions = df.iloc[:, 3:15].groupby(df['Physician Name']).sum()


# Sum across columns to get total prescription counts per physician
physician_prescriptions['TotalPrescriptions'] = physician_prescriptions.sum(axis=1)

# Sort physicians in descending order of total prescription counts
sorted_physicians = physician_prescriptions.sort_values(by='TotalPrescriptions', ascending=False)

# Select the top 200 physicians
top_200_physicians = sorted_physicians.head(200)

# Display the list of top 200 physicians
print(top_200_physicians)

###############################################################

df2=pd.read_csv('affiliation.csv')
df2.info()

# Get the list of hospitals with affiliated top physicians
physician_names=df['Physician Name']
hospitals_with_affiliated_top_physicians = []
for i in physician_names:
    if i not in hospitals_with_affiliated_top_physicians:
        hospitals_with_affiliated_top_physicians.append(i)

# Get the list of all hospitals
all_hospitals = df2['Hospital Name'].unique()

# Find hospitals without affiliated top physicians
hospitals_without_affiliated_top_physicians = list(set(all_hospitals) - set(hospitals_with_affiliated_top_physicians))

# Calculate the number of hospitals without affiliated top physicians
num_hospitals_without_affiliation = len(hospitals_without_affiliated_top_physicians)

# Display the result
print(f"Number of hospitals without affiliated top physicians: {num_hospitals_without_affiliation}")

########################################################





# Merge the two dataframes based on 'ID'
merged_df = df2.merge(df[['Physician ID', 'Specialty']], on='Physician ID', how='left')

print(merged_df)
merged_df['Specialty'].unique()

target_specialties = ["HEMATOLOGY", "HEMATOLOGY/ONCOLOGY", "ONCOLOGY MEDICAL", "PEDIATRIC HEMATOLOGY ONCOLOGY"]
# specialty_filtered_data = merged_df[merged_df['Specialty'].isin(target_specialties)]
specialty_filtered_data=[]
for i in merged_df['Specialty']:
    if i in target_specialties:
        specialty_filtered_data.append(i)

specialty_filtered_data = pd.DataFrame(merged_df)

# List of target specialties
target_specialties = ["HEMATOLOGY", "HEMATOLOGY/ONCOLOGY", "ONCOLOGY MEDICAL", "PEDIATRIC HEMATOLOGY ONCOLOGY"]
# specialty_filtered_data = merged_df[merged_df['Specialty'].isin(target_specialties)]
filtered_specialties=[]
for i in merged_df['Specialty']:
    if i in target_specialties:
        filtered_specialties.append(i)
# Group and count physicians by hospital
physicians_by_hospital = specialty_filtered_data.groupby('Hospital Name')['Physician Name'].nunique().reset_index()

# Rename columns for clarity
physicians_by_hospital.columns = ['Hospital Name', 'Number of Physicians']

# Sort hospitals by physician count
sorted_hospitals = physicians_by_hospital.sort_values(by='Number of Physicians', ascending=False)

# Get the top 5 hospitals
top_5_hospitals = sorted_hospitals.head(5)

print("Top 5 hospitals based on physician count from specified specialties:")
print(top_5_hospitals)



############################################################
df['Fludara_Sales'] = df[["Jan'23","Feb'23","Mar'23","Apr'23", "May'23","Jun'23"]].sum(axis=1)

df['Mercapto_Sales'] = df[["Jan'232","Feb'232","Mar'232","Apr'232", "May'232","Jun'232"]].sum(axis=1)
df['Total_Sales'] = df[['Fludara_Sales', 'Mercapto_Sales']].sum(axis=1)

merged_df = merged_df.merge(df[['Physician ID', 'Total_Sales','Mercapto_Sales','Fludara_Sales']], on='Physician ID', how='left')

print(merged_df)

##################################
merged_df.info()

# Step 2: Rescale total sales to a value of 54,000 for workload index
desired_total_workload = 54000
total_sales = merged_df['Total_Sales'].sum()
scaling_factor = desired_total_workload / total_sales
merged_df['Workload_Index'] = merged_df['Total_Sales'] * scaling_factor

# Desired total workload
desired_total_workload = 54000

# Calculate scaling factor
scaling_factor = desired_total_workload / total_sales

# Calculate the workload index for each territory
merged_df['Workload_Index'] = merged_df['Total_Sales'] * scaling_factor

# Sample workload data with territories as primary key
workload_data = {
    'Territory_Name': merged_df['Territory_Name'],
    'Workload_Index': merged_df['Workload_Index']  # Sample workload data, replace with actual values
}

# Create the workload DataFrame
territory_workload = pd.DataFrame(workload_data)

# Group by territories to calculate accurate workload index
grouped_workload = territory_workload.groupby('Territory_Name')['Workload_Index'].sum().reset_index()

# Merge the workload index back to the territory_data DataFrame
territories = merged_df.merge(grouped_workload, on='Territory_Name', how='left')

# Display the merged data with calculated workload index

print((territories).info())


################################################
# Calculate the number of territories above and below the balanced workload index range
balanced_min = 700
balanced_max = 1300

# Group territories based on the balanced range
territories['Balanced_Status'] = (territories['Workload_Index_y'] >= balanced_min) & (territories['Workload_Index_y'] <= balanced_max)

# Count the number of territories above and below the balanced range
territories_below = territories[territories['Balanced_Status'] == False]['Territory_Name'].nunique()
territories_above = territories[territories['Balanced_Status'] == True]['Territory_Name'].nunique()

print(f"Number of territories below balanced range: {territories_below}")
print(f"Number of territories above balanced range: {territories_above}")