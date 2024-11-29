import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
World_data = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\world_data.csv')
country_data = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\countries_data.csv')
df_territory = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\country-codes.csv')
df_support = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\sp.csv')

# Xử lý dữ liệu
World_data['date'] = pd.to_datetime(World_data['date'])
cases_over_time = World_data.groupby('date')[['new_cases', 'new_deaths']].sum().reset_index()
cases_over_time['CumulativeCases'] = cases_over_time['new_cases'].cumsum()
cases_over_time['CumulativeDeaths'] = cases_over_time['new_deaths'].cumsum()

# Layout Streamlit
st.title("Xu hướng toàn cầu COVID-19")

# 1. Biểu đồ tích lũy ca nhiễm, tử vong và ảnh hưởng của tiêm vaccine
st.subheader("Tích lũy ca nhiễm và tử vong theo thời gian, ảnh hưởng của tiêm vaccine")
time_vaccine_cases = World_data.groupby('date').sum()

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Ngày')
ax1.set_ylabel('Số ca tử vong tích lũy', color='red')
ax1.plot(cases_over_time['date'], cases_over_time['CumulativeDeaths'], label='Ca tử vong tích lũy', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Tổng số liều tiêm (triệu)', color='blue')
ax2.plot(cases_over_time['date'], time_vaccine_cases['total_vaccinations'] / 1e6, label='Tổng số liều tiêm (triệu)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Tích lũy ca tử vong và số liều tiêm vaccine theo thời gian')
fig.tight_layout()
st.pyplot(fig)

# 2. Biểu đồ hỗ trợ chính phủ toàn cầu
st.subheader("Hỗ trợ chính phủ toàn cầu")
gov_support = df_support.groupby('Code')['e1_income_support'].max().reset_index()
df_merged = gov_support.merge(df_territory, left_on='Code', right_on='ISO3166-1-Alpha-3', how='left')

fig = px.choropleth(
    df_merged,
    locations='Code',
    color='e1_income_support',
    hover_name='official_name_en',
    color_continuous_scale='Blues',
    title='Hỗ trợ thu nhập toàn cầu',
    projection='natural earth'
)

fig.update_layout(
    coloraxis_colorbar=dict(
        title="Income Support",
        tickvals=[0, 1, 2],
        ticktext=["0: Không hỗ trợ", "1: Lên đến 50% thu nhập", "2: Hơn 50% thu nhập"]
    )
)

st.plotly_chart(fig)

# 3. Tương quan giữa ca nhiễm, tử vong và vaccine
st.subheader("Tương quan giữa ca nhiễm, tử vong và số liều vaccine")
correlation_data = World_data[['new_cases', 'new_deaths', 'total_vaccinations']].dropna()

fig = px.scatter(
    correlation_data,
    x='new_cases',
    y='new_deaths',
    size='total_vaccinations',
    color='total_vaccinations',
    labels={'new_cases': 'Số ca nhiễm mới', 'new_deaths': 'Số ca tử vong mới', 'total_vaccinations': 'Số liều tiêm'},
    title='Tương quan giữa ca nhiễm, tử vong và số liều vaccine',
    hover_data=['new_cases', 'new_deaths', 'total_vaccinations']
)
st.plotly_chart(fig)

# 4. Biểu đồ tích lũy ca nhiễm và tử vong toàn cầu
st.subheader("Tích lũy ca nhiễm và tử vong toàn cầu theo thời gian")

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x='date', y='CumulativeCases', data=cases_over_time, label='Ca nhiễm tích lũy', color='blue', linewidth=2)
sns.lineplot(x='date', y='CumulativeDeaths', data=cases_over_time, label='Ca tử vong tích lũy', color='red', linewidth=2)
plt.title("Tích lũy ca nhiễm và tử vong theo thời gian")
plt.xlabel("Ngày")
plt.ylabel("Số lượng")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)
