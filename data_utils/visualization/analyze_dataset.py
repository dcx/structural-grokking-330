import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load your data

input_file = '../ds_addmult_mod10_data/data-addmult-231102-1.5m.csv'
data = pd.read_csv(input_file)

# Function to plot interactive histograms using Plotly
def interactive_histogram(data, column, title, log_scale=False, num_bins=30):
    # Calculate the count per bin
    counts, bins = np.histogram(data[column], bins=num_bins)
    bins = 0.5 * (bins[:-1] + bins[1:])  # get the mid-point of bins
    # Calculate the percentage of total for each bin
    bin_percentages = (counts / sum(counts)) * 100
    
    # Create a new DataFrame for the histogram
    hist_data = pd.DataFrame({column: bins, 'count': counts, 'percentage': bin_percentages})
    
    # Create the figure
    fig = px.bar(hist_data, x=column, y='count',
                 hover_data={column: False, 'count': True, 'percentage': True},
                 labels={'percentage': 'Percentage of Total'},
                 title=title)
    fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))
    fig.update_layout(bargap=0.1)
    
    # Apply logarithmic scale if specified
    if log_scale:
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)

st.title(f'Dataset Distribution Viewer: {input_file}')


st.header('Distribution of Height')
interactive_histogram(data, 'height', 'Distribution of Height')


st.header('Distribution of Width')
interactive_histogram(data, 'width', 'Distribution of Width')


st.header('Distribution of Answer')
interactive_histogram(data, 'answer', 'Distribution of Answer', log_scale=True)


st.header('Distribution of Answer Mod 10')
interactive_histogram(data, 'ans_mod10', 'Distribution of Answer Mod 10')


