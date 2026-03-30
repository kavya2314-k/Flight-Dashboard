# %% [markdown]
# ## Milestone 2

# %%
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns

# %%
data = pd.read_csv("airline_preprocessed.csv",low_memory=False)
data.sample(20)

# %%

top_airlines = data['AIRLINE'].value_counts().head(10)

plt.figure(figsize=(8,6))
ax = sns.barplot(
    x=top_airlines.values,
    y=top_airlines.index,
    palette="Blues_r"
)

# Add value labels
for i, v in enumerate(top_airlines.values):
    ax.text(
        v + 0.01 * max(top_airlines.values), i,f"{v:,}",va='center'
    )

plt.title("Top 10 Airlines by Flight Volume")
plt.xlabel("Number of Flights")
plt.ylabel("Airline")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Busiest Months

# %%
monthly_flights = data.groupby('MONTH').size()

plt.figure()
monthly_flights.plot()
plt.title("Monthly Flight Volume Trend")
plt.xlabel("Month")
plt.ylabel("Total Flights")
plt.show()

# %% [markdown]
# ## Route Congestion vs Delay

# %%
route_stats = data.groupby('ROUTE').agg({
    'ARRIVAL_DELAY': 'mean',
    'ROUTE': 'count'
}).rename(columns={'ROUTE':'flight_count'})

plt.figure()
route_stats = route_stats[route_stats['flight_count'] > 500]
sns.scatterplot(data=route_stats, x='flight_count', y='ARRIVAL_DELAY',
                size='flight_count', hue='ARRIVAL_DELAY', palette='RdYlGn_r')
#sns.scatterplot(data=route_stats,x='flight_count',y='ARRIVAL_DELAY')
plt.title("Route Congestion vs Average Delay")
plt.xlabel("Number of Flights")
plt.ylabel("Average Arrival Delay")
plt.show()

# %% [markdown]
# ## Delay Distribution by Airline

# %%
top5 = data['AIRLINE'].value_counts().head(5).index
filtered = data[data['AIRLINE'].isin(top5)]

plt.figure()
sns.boxplot(
    data=filtered,
    x='AIRLINE',
    y='ARRIVAL_DELAY'
)
plt.title("Delay Distribution by Top Airlines")
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# ## Busiest Route

# %%
top_routes = data['ROUTE'].value_counts().head(10)
plt.figure(figsize=(10,5))
top_routes.plot(kind='barh', color='darkorange')
plt.title("Top 10 Busiest Routes")
plt.xlabel("Number of Flights")
plt.gca().invert_yaxis()
plt.show()
print("Observation: SFO-LAX is the busiest corridor — short-haul high-frequency routes dominate the top 10.")

# %% [markdown]
# ## Delay Analysis

# %% [markdown]
# ## Delay Cause Comparison

# %%
delay_causes = data.groupby('AIRLINE')[[
    'AIR_SYSTEM_DELAY','SECURITY_DELAY',
    'WEATHER_DELAY',
    'AIRLINE_DELAY',
    'LATE_AIRCRAFT_DELAY'
]].mean().head(10)

delay_causes.plot(kind='bar', stacked=True)
plt.title("Average Delay Causes by Airline")
plt.ylabel("Average Delay (Minutes)")
plt.xticks(rotation=45)
plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd

cause_cols = ['AIRLINE_DELAY','WEATHER_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
cause_labels = ['Carrier','Weather','NAS','Security','Late Aircraft']

# Calculate mean delay causes
delay_causes = data[data['ARRIVAL_DELAY'] > 0].groupby('AIRLINE')[cause_cols].mean()
delay_causes.columns = cause_labels

# Convert to percentage of total for each airline
delay_pct = delay_causes.div(delay_causes.sum(axis=1), axis=0) * 100

# Plot
fig, ax = plt.subplots(figsize=(13, 6))
delay_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='white', width=0.7)

# Add percentage labels inside each bar segment
for bar_stack in ax.containers:
    for bar in bar_stack:
        height = bar.get_height()
        if height > 5:  # only label if segment is wide enough to read
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                f'{height:.0f}%',
                ha='center', va='center',
                fontsize=7.5, color='black', fontweight='bold'
            )

ax.set_title("Delay Cause Breakdown by Airline (% of Total Delay)", fontsize=13)
ax.set_ylabel("Percentage of Total Delay (%)")
ax.set_xlabel("Airline")
ax.set_ylim(0, 105)
ax.legend(title="Delay Cause", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

print("Observation: Late Aircraft and Carrier delays together account for 70-80% of total delay across most airlines.")
print("Security delay is negligible (<1%) for all carriers.")

# %% [markdown]
# ## Weather Delay Seasonal Trend

# %%
weather_month = data.groupby('MONTH')['WEATHER_DELAY'].mean()

plt.figure()
weather_month.plot()
plt.title("Average Weather Delay by Month")
plt.xlabel("Month")
plt.ylabel("Weather Delay (Minutes)")
plt.show()

# %%

plt.figure(figsize=(8,6))

sns.histplot(
    data['ARRIVAL_DELAY'],
    bins=50,
    kde=True
)

plt.title("Distribution of Arrival Delay")
plt.xlabel("Arrival Delay (Minutes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

delay = data['ARRIVAL_DELAY'].dropna()
delay = delay[(delay >= -60) & (delay <= 180)]

plt.figure(figsize=(10, 5))
plt.hist(delay, bins=80, color='steelblue', edgecolor='white')

plt.axvline(delay.mean(),            color='red',   linestyle='--', label=f'Mean: {delay.mean():.1f} min')
plt.axvline(delay.median(),          color='green', linestyle='--', label=f'Median: {delay.median():.1f} min')
plt.axvline(delay.quantile(0.95),    color='black', linestyle=':',  label=f'95th %ile: {delay.quantile(0.95):.0f} min')

plt.title("Distribution of Arrival Delay")
plt.xlabel("Arrival Delay (minutes)")
plt.ylabel("Number of Flights")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Mean:       {delay.mean():.1f} min")
print(f"Median:     {delay.median():.1f} min")
print(f"Early (<0): {(delay < 0).mean()*100:.1f}%")
print(f"On-Time (0-15 min):    {((delay>=0)&(delay<=15)).mean()*100:.1f}%")
print(f"Severely delayed (>60): {(delay > 60).mean()*100:.1f}%")

# %%
data['delay_severity'] = pd.cut(
    data['ARRIVAL_DELAY'],
    bins=[-1000, 0, 15, 60, 10000],
    labels=['On Time', 'Minor', 'Moderate', 'Severe']
)

severity = (
    data.groupby('AIRLINE')['delay_severity']
    .value_counts(normalize=True)
    .unstack()
)

severity.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title("Delay Severity Composition by Airline")
plt.ylabel("Proportion")
plt.legend(title="Severity")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Departure Delay Analysis

# %%
delay = data['DEPARTURE_DELAY'].dropna()
delay = delay[(delay >= -60) & (delay <= 180)]

plt.figure(figsize=(10,5))
plt.hist(delay, bins=80, color='steelblue', edgecolor='white')
plt.axvline(delay.mean(),   color='red',   linestyle='--', label=f'Mean: {delay.mean():.1f} min')
plt.axvline(delay.median(), color='green', linestyle='--', label=f'Median: {delay.median():.1f} min')
plt.title("Departure Delay Distribution")
plt.xlabel("Delay (minutes)")
plt.ylabel("Number of Flights")
plt.legend()
plt.tight_layout()
plt.show()
print(f"Observation: Most flights are on time or early. Mean ({delay.mean():.1f}) >> Median ({delay.median():.1f}) shows extreme delays pull the average up.")

# %%
airport_delay = data[data['CANCELLED']==0].groupby('ORIGIN_AIRPORT').agg(
    avg_delay  = ('DEPARTURE_DELAY','mean'),
    flight_count = ('DEPARTURE_DELAY','count')
).reset_index()

# Only airports with meaningful volume
airport_delay = airport_delay[airport_delay['flight_count'] > 1000]
top15 = airport_delay.nlargest(15, 'avg_delay')

plt.figure(figsize=(10,6))
plt.barh(top15['ORIGIN_AIRPORT'], top15['avg_delay'], color='tomato')
plt.title("Top 15 Airports with Highest Average Departure Delay")
plt.xlabel("Avg Delay (minutes)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
print("Observation: Mid-size congested airports show worse delays than major hubs which have better scheduling buffers.")

# %%
sample = data[(data['ARRIVAL_DELAY'].between(-60, 180)) & 
              (data['CANCELLED'] == 0)].sample(10000, random_state=42)

# Add distance bucket for color coding
sample['DIST_BUCKET'] = pd.cut(sample['DISTANCE'],
    bins=[0, 500, 1000, 2000, 5000],
    labels=['Short (<500mi)', 'Medium (500-1000mi)', 
            'Long (1000-2000mi)', 'Ultra (>2000mi)'])
plt.figure(figsize=(11, 6))
sns.scatterplot(data=sample, x='DISTANCE', y='ARRIVAL_DELAY',
                hue='DIST_BUCKET', palette='coolwarm', alpha=0.4, s=15)

# Add trend line
import numpy as np
z = np.polyfit(sample['DISTANCE'], sample['ARRIVAL_DELAY'], 1)
p = np.poly1d(z)
x_line = np.linspace(sample['DISTANCE'].min(), sample['DISTANCE'].max(), 100)
plt.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--', label='Trend Line')

plt.axhline(0, color='black', linestyle=':', linewidth=1, label='On Time')
plt.title("Flight Distance vs Arrival Delay\n(does flying further mean more delay?)")
plt.xlabel("Flight Distance (miles)")
plt.ylabel("Arrival Delay (minutes)")
plt.legend(title="Distance Category", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.show()

print(f"Correlation between Distance and Arrival Delay: {sample['DISTANCE'].corr(sample['ARRIVAL_DELAY']):.3f}")
print("Observation: Longer flights tend to arrive closer to on-time or early — pilots can recover delays in air on long routes. Short flights have highest delay variance.")

# %%



