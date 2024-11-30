import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="COVID-19 Global Dashboard", layout="wide")

# Load data function
@st.cache_data
def load_data():
    # Upload files from user input
    world_data=pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\world_data.csv')
    country_data=pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\countries_data.csv')
    vietnam_data=country_data[country_data['location']=='Vietnam']
    df_territory = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\country-codes.csv')
    continent_data=pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\continents_data.csv')
    sp_data=pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\sp.csv')
    gdp_data=pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\gdp1.csv')
    travel_data=pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\travel.csv')

    
    if world_data and country_data and df_territory and gdp_data:
        world_df = pd.read_csv(world_data)
        country_df = pd.read_csv(country_data)
        territory_df = pd.read_csv(df_territory)
        gdp_df = pd.read_csv(gdp_data)
        
        # Convert date columns to datetime
        world_df['date'] = pd.to_datetime(world_df['date'])
        country_df['date'] = pd.to_datetime(country_df['date'])
        
        return {
            'world_data': world_df,
            'country_data': country_df,
            'df_territory': territory_df,
            'gdp_data': gdp_df
        }
    else:
        return None

# Main dashboard
def main():
    st.title("🦠 Global COVID-19 Dashboard")

    # Load data
    data_dict = load_data()
    
    if data_dict:
        # Sidebar navigation
        page = st.sidebar.selectbox("Select Analysis", [
            "Global Overview",
            "Economic Impact",
            "Future Prediction"
        ])
        
        # Global Overview Page
        if page == "Global Overview":
            st.header("Global COVID-19 Overview")
            
            # Heatmaps for cases and deaths
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Total Cases Heatmap")
                df_cases = data_dict['country_data'].groupby('iso_code')['total_cases'].max().reset_index()
                df_merged = df_cases.merge(data_dict['df_territory'], left_on='iso_code', right_on='ISO3166-1-Alpha-3', how='left')
                
                fig_cases = px.choropleth(
                    df_merged,
                    locations='iso_code',
                    color='total_cases',
                    hover_name='official_name_en',
                    color_continuous_scale='Reds',
                    title='COVID-19 Total Cases Heatmap'
                )
                st.plotly_chart(fig_cases, use_container_width=True)
            
            with col2:
                st.subheader("Total Deaths Heatmap")
                df_deaths = data_dict['country_data'].groupby('iso_code')['total_deaths'].max().reset_index()
                df_merged = df_deaths.merge(data_dict['df_territory'], left_on='iso_code', right_on='ISO3166-1-Alpha-3', how='left')
                
                fig_deaths = px.choropleth(
                    df_merged,
                    locations='iso_code',
                    color='total_deaths',
                    hover_name='official_name_en',
                    color_continuous_scale='Reds',
                    title='COVID-19 Total Deaths Heatmap'
                )
                st.plotly_chart(fig_deaths, use_container_width=True)

        # Economic Impact Page
        elif page == "Economic Impact":
            st.header("Economic Impact of COVID-19")
            
            year = st.selectbox("Select Year", [2021, 2022])
            gdp_data = data_dict['gdp_data']
            
            if gdp_data is not None:
                # Example mapping logic for GDP data
                gdp_data['Year'] = gdp_data['Year'].fillna(2021)  # Ensure year column exists
                gdp_filtered = gdp_data[gdp_data['Year'] == year]
                st.dataframe(gdp_filtered.head())

        # Future Prediction Page
        elif page == "Future Prediction":
            st.header("Future Predictions")
            
            World_data = data_dict['world_data']
            World_data['days_since_start'] = (World_data['date'] - World_data['date'].min()).dt.days
            
            # Linear Regression for Prediction
            X = World_data[['days_since_start']]
            y_cases = World_data['new_cases'].fillna(0)
            regressor_cases = LinearRegression().fit(X, y_cases)
            
            # Future predictions
            future_days = np.arange(X['days_since_start'].max() + 1, X['days_since_start'].max() + 366).reshape(-1, 1)
            future_cases = np.maximum(regressor_cases.predict(future_days), 0)
            
            fig, ax = plt.subplots()
            ax.plot(World_data['date'], y_cases, label='Historical Cases', color='blue')
            ax.plot(pd.to_datetime(World_data['date'].min()) + pd.to_timedelta(future_days.flatten(), unit='D'), 
                     future_cases, label='Predicted Cases', color='red', linestyle='--')
            ax.set_title('Cases Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('New Cases')
            ax.legend()
            st.pyplot(fig)
    else:
        st.error("Please upload all required datasets.")

if __name__ == "__main__":
    main()
