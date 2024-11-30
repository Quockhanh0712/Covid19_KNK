import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from matplotlib.ticker import FuncFormatter 
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.cm import rainbow
import matplotlib.dates as mdates  
import geopandas as gpd
import unidecode  # Xử lý dấu tiếng Việt


# Tải dữ liệu từ file CSV hoặc Excel
@st.cache_data
def load_data():
    World_data=pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\world_data.csv')
    country_data=pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\countries_data.csv')
    vietnam_data=country_data[country_data['location']=='Vietnam']
    df_territory = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\country-codes.csv')
    continent_data=pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\continents_data.csv')
    df_support = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\sp.csv')

    return World_data, country_data, vietnam_data, df_territory, continent_data,df_support


# Tải dữ liệu khi người dùng mở ứng dụng
World_data, country_data, vietnam_data, df_territory, continent_data,df_support = load_data()
# lọc dữ liệu
World_data['date'] = pd.to_datetime(World_data['date'])
cases_over_time = World_data.groupby('date')['new_cases'].sum().reset_index() 
deaths_over_time = World_data.groupby('date')['new_deaths'].sum().reset_index()  
cases_over_time['CumulativeCases'] = cases_over_time['new_cases'].cumsum()
deaths_over_time['CumulativeDeaths'] = deaths_over_time['new_deaths'].cumsum()
# Tựa đề của ứng dụng
st.title("Phân Tích Tình Hình Dịch COVID-19")

# Sidebar: Tùy chọn phân cấp
st.sidebar.header("Tùy chọn chính")
main_option = st.sidebar.selectbox(
    "Chọn phân tích:",
    [
        "Tổng quan về tình hình dịch COVID-19",
        "Các châu lục và quốc gia",
        "Các ảnh hưởng của COVID-19",
        "Việt Nam",
    ],
)

# Xử lý logic cho từng tùy chọn chính
if main_option == "Tổng quan về tình hình dịch COVID-19":
    st.header("Tổng quan về tình hình dịch COVID-19")

    # Các tùy chọn cho phần tổng quan
    sub_option = st.selectbox(
        "Chọn phân tích chi tiết:",
        [
            "Số ca nhiễm và tử vong theo thời gian",
            "Tích lũy số ca nhiễm và tử vong",
            "Bản đồ thế giới (số ca nhiễm, tử vong, hạn chế phòng dịch)",
            "Biểu đồ tương quan số ca nhiễm và tử vong",
            "So sánh biến thể Omicron và Delta",
            "Tiêm chủng",
            "Dự đoán số ca nhiễm và tử vong trong tương lai",
        ],
    )

    # Hiển thị nội dung dựa trên tùy chọn chi tiết
    if sub_option == "Số ca nhiễm và tử vong theo thời gian":
        st.subheader("Số ca nhiễm và tử vong theo thời gian")

# Sử dụng Plotly để vẽ biểu đồ tương tác
        fig = px.line(cases_over_time, x='date', y='new_cases', title='Số Ca Nhiễm Mới Theo Thời Gian', 
                      labels={'new_cases': 'Số Ca Nhiễm Mới', 'date': 'Ngày'})

        # Hiển thị biểu đồ tương tác trong Streamlit
        st.plotly_chart(fig)

        

# Vẽ biểu đồ với Plotly cho 'new_deaths' với màu đỏ
        fig = px.line(deaths_over_time, 
              x='date', 
              y='new_deaths', 
              title='Số Ca Tử Vong Mới Theo Thời Gian', 
              labels={'new_deaths': 'Số Ca Tử Vong Mới', 'date': 'Ngày'},
              line_shape='linear')  # Thêm dòng biểu đồ mượt mà

# Chỉnh màu đỏ cho đường biểu đồ
        fig.update_traces(line=dict(color='red'))

# Hiển thị biểu đồ tương tác trong Streamlit
        st.plotly_chart(fig)



    elif sub_option == "Tích lũy số ca nhiễm và tử vong":
        st.subheader("Tích lũy số ca nhiễm và tử vong")
        # TODO: Vẽ biểu đồ tích lũy số ca nhiễm và tử vong
        # Sử dụng Plotly để vẽ biểu đồ tương tác

# Biểu đồ Tích Lũy Ca Nhiễm
        fig_cases = px.line(cases_over_time, 
                    x='date', 
                    y='CumulativeCases', 
                    title='Biểu Đồ Tích Lũy Ca Nhiễm',
                    labels={'CumulativeCases': 'Số Ca Nhiễm Tích Lũy', 'date': 'Ngày'},
                    line_shape='linear')
        fig_cases.update_traces(line=dict(color='green', width=2))

# Hiển thị biểu đồ Tích Lũy Ca Nhiễm trong Streamlit
        st.plotly_chart(fig_cases)

# Biểu đồ Tích Lũy Ca Tử Vong
        fig_deaths = px.line(deaths_over_time, 
                     x='date', 
                     y='CumulativeDeaths', 
                     title='Biểu Đồ Tích Lũy Ca Tử Vong',
                     labels={'CumulativeDeaths': 'Số Ca Tử Vong Tích Lũy', 'date': 'Ngày'},
                     line_shape='linear')
        fig_deaths.update_traces(line=dict(color='purple', width=2))

# Hiển thị biểu đồ Tích Lũy Ca Tử Vong trong Streamlit
        st.plotly_chart(fig_deaths)


    elif sub_option == "Bản đồ thế giới (số ca nhiễm, tử vong, hạn chế phòng dịch)":
        st.subheader("Bản đồ thế giới về COVID-19")
        # Bản đồ số ca nhiễm trên toàn cầu
        df_cases = country_data.groupby('iso_code')['total_cases'].max().reset_index()
        df_merged = df_cases.merge(df_territory, left_on='iso_code', right_on='ISO3166-1-Alpha-3', how='left')

        fig_cases = px.choropleth(df_merged,
                          locations='iso_code',
                          color='total_cases', 
                          hover_name='official_name_en',
                          color_continuous_scale='Reds',
                          title='World COVID-19 Heatmap (Max Total Cases per Country)',
                          projection='natural earth')

# Hiển thị bản đồ số ca nhiễm trong Streamlit
        st.plotly_chart(fig_cases)

# Bản đồ số ca tử vong trên toàn cầu
        df_deaths = country_data.groupby('iso_code')['total_deaths'].max().reset_index()
        df_merged = df_deaths.merge(df_territory, left_on='iso_code', right_on='ISO3166-1-Alpha-3', how='left')

        fig_deaths = px.choropleth(df_merged,
                           locations='iso_code',
                           color='total_deaths',
                           hover_name='official_name_en',
                           color_continuous_scale='Reds',
                           title='World COVID-19 Heatmap (Max Total Deaths per Country)',
                           projection='natural earth')

# Hiển thị bản đồ số ca tử vong trong Streamlit
        st.plotly_chart(fig_deaths)

# Bản đồ hỗ trợ của chính phủ trên toàn cầu

        gov_support = df_support.groupby('Code')['e1_income_support'].max().reset_index()
        df_merged = gov_support.merge(df_territory, left_on='Code', right_on='ISO3166-1-Alpha-3', how='left')

        fig_support = px.choropleth(df_merged,
                            locations='Code',
                            color='e1_income_support',  
                            hover_name='official_name_en',
                            color_continuous_scale='Blues', 
                            title='World Income Support Heatmap',
                            projection='natural earth')

        fig_support.update_layout(coloraxis_colorbar=dict(
                title="Income Support",
                tickvals=[0, 1, 2],  
                ticktext=["0: No support", "1: Up to 50% lost income", "2: More than 50% lost income"]
        ))

# Hiển thị bản đồ hỗ trợ của chính phủ trong Streamlit
        st.plotly_chart(fig_support)
        # TODO: Vẽ bản đồ số ca nhiễm, tử vong, và các hạn chế

    elif sub_option == "Biểu đồ tương quan số ca nhiễm và tử vong":
        st.subheader("Biểu đồ tương quan giữa số ca nhiễm và số ca tử vong")
        st.title("Phân Tích Tương Quan COVID-19")

# Bước 2: Lọc các cột liên quan đến phân tích tương quan
        correlation_data = World_data[['new_cases', 'new_deaths', 'total_cases', 'total_deaths']]

# Bước 3: Tính toán ma trận tương quan
        correlation_matrix = correlation_data.corr()

# Hiển thị ma trận tương quan
        st.subheader("Ma Trận Tương Quan")
        st.write(correlation_matrix)

# --- Biểu đồ Tương quan giữa Số Ca Nhiễm Mới và Số Ca Tử Vong Mới ---
        st.subheader("Biểu Đồ Tương Quan Giữa Số Ca Nhiễm Mới và Số Ca Tử Vong Mới")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=World_data, x='new_cases', y='new_deaths', color='green', alpha=0.7, ax=ax)
        ax.set_title('Correlation Between New Cases and New Deaths', fontsize=16)
        ax.set_xlabel('New Cases', fontsize=12)
        ax.set_ylabel('New Deaths', fontsize=12)
        ax.grid(alpha=0.3)

# Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)



    elif sub_option == "So sánh biến thể Omicron và Delta":
        st.subheader("So sánh biến thể Omicron và Delta")
        # TODO: Vẽ biểu đồ so sánh hai biến thể

# Tiêu đề
        st.title("So Sánh Biến Chủng Omicron và Delta")

# Lọc dữ liệu cho biến chủng Delta (từ tháng 5/2021 đến tháng 12/2021)
        delta_wave = World_data[(World_data['date'] >= '2021-05-01') & (World_data['date'] <= '2021-12-31')]

# Lọc dữ liệu cho biến chủng Omicron (từ tháng 12/2021 đến giữa năm 2022)
        omicron_wave = World_data[(World_data['date'] >= '2021-12-01') & (World_data['date'] <= '2022-06-30')]

# --- So sánh số ca nhiễm COVID-19 giữa Delta và Omicron ---
        st.subheader("So Sánh Số Ca Nhiễm COVID-19 Giữa Biến Chủng Delta và Omicron")

# Vẽ biểu đồ số ca nhiễm
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(delta_wave['date'], delta_wave['new_cases'], label='Delta Wave', color='blue')
        ax.plot(omicron_wave['date'], omicron_wave['new_cases'], label='Omicron Wave', color='orange')
        ax.set_title('So sánh số ca nhiễm COVID-19 giữa biến chủng Delta và Omicron', fontsize=16)
        ax.set_xlabel('Ngày', fontsize=12)
        ax.set_ylabel('Số ca nhiễm mới', fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.set_xticklabels(delta_wave['date'], rotation=45)

# Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)

# --- So sánh số ca tử vong COVID-19 giữa Delta và Omicron ---
        st.subheader("So Sánh Số Ca Tử Vong COVID-19 Giữa Biến Chủng Delta và Omicron")

# Vẽ biểu đồ số ca tử vong
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(delta_wave['date'], delta_wave['new_deaths'], label='Delta Wave', color='red')
        ax.plot(omicron_wave['date'], omicron_wave['new_deaths'], label='Omicron Wave', color='green')
        ax.set_title('So sánh số ca tử vong COVID-19 giữa biến chủng Delta và Omicron', fontsize=16)
        ax.set_xlabel('Ngày', fontsize=12)
        ax.set_ylabel('Số ca tử vong mới', fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.set_xticklabels(delta_wave['date'], rotation=45)

# Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)

    elif sub_option == "Tiêm chủng":
        st.subheader("Phân tích tiêm chủng toàn cầu")
        # TODO: Vẽ biểu đồ tiêm chủng
        # Tiêu đề
        st.title("Xu Hướng Số Ca Nhiễm và Số Liều Tiêm Theo Thời Gian")

# Bước 1: Tính tổng số ca nhiễm và tổng số liều tiêm theo ngày
        time_vaccine_cases = World_data.groupby('date').sum()

# --- Xu hướng số ca tử vong mới và số liều tiêm ---
        st.subheader("Số Ca Tử Vong Mới và Số Liều Tiêm Theo Thời Gian")

# Vẽ biểu đồ với hai trục Y
        fig, ax1 = plt.subplots(figsize=(12, 6))

# Trục Y thứ nhất - Số ca tử vong mới
        ax1.set_xlabel('Ngày')
        ax1.set_ylabel('Số ca tử vong mới', color='red')
        ax1.plot(time_vaccine_cases.index, time_vaccine_cases['new_deaths'], label='Số ca tử vong mới', color='red')
        ax1.tick_params(axis='y', labelcolor='red')

# Trục Y thứ hai - Số liều tiêm
        ax2 = ax1.twinx()
        ax2.set_ylabel('Tổng số liều tiêm (triệu)', color='blue')
        ax2.plot(time_vaccine_cases.index, time_vaccine_cases['total_vaccinations'] / 1e6, label='Tổng số liều tiêm (triệu)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

# Cài đặt tiêu đề và hiển thị biểu đồ
        plt.title('Số Ca Tử Vong Mới và Số Liều Tiêm Theo Thời Gian')
        fig.tight_layout()

# Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)

# --- Xu hướng số ca nhiễm mới và số liều tiêm ---
        st.subheader("Số Ca Nhiễm Mới và Số Liều Tiêm Theo Thời Gian")

# Vẽ biểu đồ với hai trục Y
        fig, ax1 = plt.subplots(figsize=(12, 6))

# Trục Y thứ nhất - Số ca nhiễm mới
        ax1.set_xlabel('Ngày')
        ax1.set_ylabel('Số ca nhiễm mới (trung bình)', color='red')
        ax1.plot(time_vaccine_cases.index, time_vaccine_cases['new_cases'], label='Số ca nhiễm mới (trung bình)', color='red')
        ax1.tick_params(axis='y', labelcolor='red')

# Trục Y thứ hai - Số liều tiêm
        ax2 = ax1.twinx()
        ax2.set_ylabel('Tổng số liều tiêm (triệu)', color='blue')
        ax2.plot(time_vaccine_cases.index, time_vaccine_cases['total_vaccinations'] / 1e6, label='Tổng số liều tiêm (triệu)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

# Cài đặt tiêu đề và hiển thị biểu đồ
        plt.title('Số Ca Nhiễm Mới và Số Liều Tiêm Theo Thời Gian')
        fig.tight_layout()

# Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)

    elif sub_option == "Dự đoán số ca nhiễm và tử vong trong tương lai":
        st.subheader("Dự đoán số ca nhiễm và tử vong trong tương lai")
        # TODO: Vẽ biểu đồ dự đoán
        # Chuyển đổi cột 'date' thành kiểu dữ liệu datetime
        World_data['date'] = pd.to_datetime(World_data['date'])

# Tính số ngày kể từ ngày đầu tiên trong dữ liệu
        World_data['days_since_start'] = (World_data['date'] - World_data['date'].min()).dt.days

# Tạo mô hình hồi quy tuyến tính cho số ca nhiễm
        X = World_data[['days_since_start']]  # Biến độc lập (số ngày)
        y_cases = World_data['new_cases']  # Biến phụ (số ca nhiễm)

        regressor_cases = LinearRegression()
        regressor_cases.fit(X, y_cases)

# Dự đoán số ca nhiễm trong tương lai (365 ngày tới)
        future_days = np.arange(X['days_since_start'].max() + 1, X['days_since_start'].max() + 366).reshape(-1, 1)
        future_cases = regressor_cases.predict(future_days)

# Đảm bảo rằng số ca nhiễm không thể âm
        future_cases = np.maximum(future_cases, 0)

# Tạo mô hình hồi quy tuyến tính cho số ca tử vong
        y_deaths = World_data['new_deaths']  # Biến phụ (số ca tử vong)

        regressor_deaths = LinearRegression()
        regressor_deaths.fit(X, y_deaths)

# Dự đoán số ca tử vong trong tương lai (365 ngày tới)
        future_deaths = regressor_deaths.predict(future_days)

# Đảm bảo rằng số ca tử vong không thể âm
        future_deaths = np.maximum(future_deaths, 0)

# Tạo layout cho Streamlit
        st.title("Dự đoán Số Ca Nhiễm và Ca Tử Vong COVID-19")

# Biểu đồ dự đoán số ca nhiễm
        st.subheader('Dự đoán số ca nhiễm COVID-19 trong 1 năm tới')
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(World_data['date'], World_data['new_cases'], label='Số ca nhiễm thực tế', color='green')
        ax1.plot(pd.to_datetime(World_data['date'].min()) + pd.to_timedelta(future_days.flatten(), unit='D'), future_cases, label='Dự đoán số ca nhiễm', color='blue', linestyle='--')
        ax1.set_xlabel('Ngày')
        ax1.set_ylabel('Số ca nhiễm mới')
        ax1.legend()
        st.pyplot(fig)

# Biểu đồ dự đoán số ca tử vong
        st.subheader('Dự đoán số ca tử vong COVID-19 trong 1 năm tới')
        fig, ax2 = plt.subplots(figsize=(14, 7))
        ax2.plot(World_data['date'], World_data['new_deaths'], label='Số ca tử vong thực tế', color='red')
        ax2.plot(pd.to_datetime(World_data['date'].min()) + pd.to_timedelta(future_days.flatten(), unit='D'), future_deaths, label='Dự đoán số ca tử vong', color='orange', linestyle='--')
        ax2.set_xlabel('Ngày')
        ax2.set_ylabel('Số ca tử vong mới')
        ax2.legend()
        st.pyplot(fig)

elif main_option == "Các châu lục và quốc gia":
    st.header("Phân tích theo châu lục và quốc gia")

    # Các tùy chọn cho phần châu lục và quốc gia
    sub_option = st.selectbox(
        "Chọn phân tích chi tiết:",
        [
            "Tỉ lệ nhiễm và tử vong của các châu lục",
            "Các nước dẫn đầu về số ca nhiễm và tử vong",
            "Xu hướng của các châu lục về số ca nhiễm, tử vong, tiêm chủng",
        ],
    )

    if sub_option == "Tỉ lệ nhiễm và tử vong của các châu lục":
        st.subheader("Tỉ lệ nhiễm và tử vong của các châu lục")
        # TODO: Vẽ biểu đồ tỉ lệ nhiễm và tử vong
        # Tính tổng số ca nhiễm và tử vong theo các châu lục
        df_grouped = country_data.groupby('continent').agg({
            'total_cases': 'sum',
            'total_deaths': 'sum'
        }).reset_index()

# Vẽ biểu đồ tròn tỉ lệ nhiễm và tử vong của các châu lục
        plt.figure(figsize=(10, 6))

# Biểu đồ tròn cho số ca nhiễm
        plt.subplot(1, 2, 1)
        plt.pie(df_grouped['total_cases'], labels=df_grouped['continent'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('coolwarm', len(df_grouped)))
        plt.title('Total Cases by Continent')

# Biểu đồ tròn cho số ca tử vong
        plt.subplot(1, 2, 2)
        plt.pie(df_grouped['total_deaths'], labels=df_grouped['continent'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(df_grouped)))
        plt.title('Total Deaths by Continent')

# Cải thiện bố cục
        plt.tight_layout()

# Hiển thị biểu đồ trong ứng dụng Streamlit
        st.title("Tỉ Lệ Nhiễm và Tử Vong của Các Châu Lục")
        st.pyplot(plt)
    elif sub_option == "Các nước dẫn đầu về số ca nhiễm và tử vong":
        st.subheader("Các nước dẫn đầu về số ca nhiễm và tử vong")
        # TODO: Vẽ biểu đồ các nước dẫn đầu
        top_countries_cases = country_data.groupby('location')['total_cases'].max().nlargest(10)
        top_countries_deaths = country_data.groupby('location')['total_deaths'].max().nlargest(10)

# Vẽ biểu đồ cho 2 biểu đồ top 10 quốc gia
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Chọn màu sắc cho biểu đồ
        case_colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries_cases))) 
        death_colors = plt.cm.Paired(np.linspace(0, 1, len(top_countries_deaths)))  

# Biểu đồ số ca nhiễm
        ax[0].barh(top_countries_cases.index, top_countries_cases.values, color=case_colors)
        ax[0].set_title('Top 10 quốc gia có số ca nhiễm cao nhất', fontsize=14, fontweight='bold')
        ax[0].set_xlabel('Số ca nhiễm', fontsize=12)
        ax[0].tick_params(axis='both', labelsize=10)

# Biểu đồ số ca tử vong
        ax[1].barh(top_countries_deaths.index, top_countries_deaths.values, color=death_colors)
        ax[1].set_title('Top 10 quốc gia có số ca tử vong cao nhất', fontsize=14, fontweight='bold')
        ax[1].set_xlabel('Số ca tử vong', fontsize=12)
        ax[1].tick_params(axis='both', labelsize=10)

# Cải thiện bố cục
        plt.tight_layout()

# Hiển thị biểu đồ trong ứng dụng Streamlit
        st.title("Top 10 Quốc Gia về Số Ca Nhiễm và Tử Vong COVID-19")
        st.pyplot(fig)

    elif sub_option == "Xu hướng của các châu lục về số ca nhiễm, tử vong, tiêm chủng":
        st.subheader("Xu hướng của các châu lục")
        # TODO: Vẽ biểu đồ xu hướng
        
# Chuyển đổi cột 'date' thành kiểu datetime và tạo cột 'year_quarter'
        continent_data['date'] = pd.to_datetime(continent_data['date'])
        continent_data['year_quarter'] = continent_data['date'].dt.to_period('Q')  # Tạo cột năm và quý

# Thay thế giá trị NaN bằng 0 cho các cột cần thiết
        for col in ['new_cases', 'new_deaths', 'total_vaccinations']:
            continent_data[col] = continent_data[col].fillna(0)

# Tạo dữ liệu theo quý cho các chỉ số
        metrics = ['new_cases', 'new_deaths', 'total_vaccinations']
        quarterly_data = {metric: (
            continent_data.groupby(['location', 'year_quarter'])[metric]
            .sum()
            .reset_index()
            .assign(year_quarter=lambda df: df['year_quarter'].astype(str))
        ) for metric in metrics}

# Hàm vẽ biểu đồ cho từng chỉ số
        def plot_metric(data, metric, title, ylabel, color):
            plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=data, 
                x='year_quarter', 
                y=metric, 
                hue='location', 
                palette='Set2', 
                marker='o', 
                markersize=6, 
                linewidth=2,
            )
            plt.title(title, fontsize=16, fontweight="bold", color=color, pad=15)
            plt.xlabel("Năm - Quý", fontsize=12, labelpad=10, fontweight="bold")
            plt.ylabel(ylabel, fontsize=12, labelpad=10, fontweight="bold")
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M')) 
            plt.tight_layout()
            plt.legend(title="Châu lục", loc="upper right", fontsize=10, title_fontsize=12, frameon=True)
            st.pyplot(plt)  # Hiển thị biểu đồ trong Streamlit

# Gọi hàm vẽ cho các chỉ số
        st.title("Biểu đồ COVID-19 theo Quý tại các Châu Lục (2020-2024)")

        plot_metric(
            quarterly_data['new_cases'], 
            'new_cases', 
            "Tổng số ca nhiễm COVID-19 theo quý tại các châu lục (2020-2024)", 
            "Tổng số ca nhiễm", 
            "darkblue"
        )

        plot_metric(
            quarterly_data['new_deaths'], 
            'new_deaths', 
            "Tổng số ca tử vong COVID-19 theo quý tại các châu lục (2020-2024)", 
            "Tổng số ca tử vong", 
            "darkred"
        )

        plot_metric(
            quarterly_data['total_vaccinations'], 
            'total_vaccinations', 
            "Tổng số liều tiêm chủng COVID-19 theo quý tại các châu lục (2020-2024)", 
            "Tổng số liều tiêm chủng", 
            "darkgreen"
        )

elif main_option == "Các ảnh hưởng của COVID-19":
    st.header("Ảnh hưởng của COVID-19")

    # Các tùy chọn cho phần ảnh hưởng
    sub_option = st.selectbox(
        "Chọn phân tích chi tiết:",
        [
            "Ảnh hưởng kinh tế với các quốc gia lớn (Q2 2020)",
            "Ảnh hưởng đến du lịch (tỉ lệ chuyến đi)",
            "GDP và tỉ lệ thất nghiệp tương quan với số ca nhiễm",
        ],
    )

    if sub_option == "Ảnh hưởng kinh tế với các quốc gia lớn (Q2 2020)":
        st.subheader("Ảnh hưởng kinh tế với các quốc gia lớn")
        # TODO: Vẽ biểu đồ ảnh hưởng kinh tế
        gdp1 = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\gdp1.csv')
        data_economic = gdp1[["Entity", "GDP growth from previous year, 2020 Q2"]]
        data_economic = data_economic.sort_values(by="GDP growth from previous year, 2020 Q2", ascending=True)

# Tạo màu sắc cho biểu đồ
        colors = rainbow(np.linspace(0, 1, len(data_economic)))

# Vẽ biểu đồ
        plt.figure(figsize=(14, 10), dpi=100, facecolor='white')

        bars = plt.barh(data_economic["Entity"], data_economic["GDP growth from previous year, 2020 Q2"], color=colors)
        plt.xlabel("GDP (%)", fontsize=15, color="#FF00FF")
        plt.ylabel("Khu vực", fontsize=15, color="#9933FF")
        plt.tick_params(axis="y", labelcolor="#00EE00", labelsize=12)
        plt.tick_params(axis="x", labelcolor="#FFD700", labelsize=12)
        plt.title(
            "Kinh tế các nước bị ảnh hưởng do dịch Covid-19",
            fontsize=18,
            color="#FF0033",
        )

# Thêm giá trị vào biểu đồ
        for bar in bars:
            plt.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%",
                va="center",
                ha="left",
                fontsize=10,
            )

# Hiển thị biểu đồ trong Streamlit
        st.title("Kinh tế các quốc gia phát triển trong Quý 2 năm 2020 bị ảnh hưởng bởi dịch Covid-19")
        st.pyplot(plt)

    elif sub_option == "Ảnh hưởng đến du lịch (tỉ lệ chuyến đi)":
        st.subheader("Ảnh hưởng đến du lịch")
        # TODO: Vẽ biểu đồ du lịch
        # Đọc dữ liệu
        travel = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\travel.csv')

# Lọc dữ liệu từ 2018 đến 2022
        df_filtered = travel[(travel['Year'] >= 2018) & (travel['Year'] <= 2022)]

# Tạo biểu đồ
        plt.figure(figsize=(12, 6))
        for entity in df_filtered['Entity'].unique():
            entity_data = df_filtered[df_filtered['Entity'] == entity]
            plt.plot(entity_data['Year'], entity_data['inbound_tourism_by_region'], marker='o', label=entity)

        plt.title('Lượng Khách Du Lịch Inbound Theo Năm (2018-2022)', fontsize=14)
        plt.xlabel('Năm', fontsize=12)
        plt.ylabel('Lượng Khách Du Lịch (Inbound)', fontsize=12)
        plt.xticks(df_filtered['Year'].unique())
        plt.legend(title="Entity", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Hiển thị biểu đồ trong Streamlit
        st.title('Ảnh Hưởng Đến Ngành Giao Thông, Du Lịch Dịch Vụ')
        st.pyplot(plt)

    elif sub_option == "GDP và tỉ lệ thất nghiệp tương quan với số ca nhiễm":
        st.subheader("Tương quan GDP và tỉ lệ thất nghiệp với số ca nhiễm")
        # TODO: Vẽ biểu đồ tương quan

# Dữ liệu Covid và GDP
        covid_data = country_data
        gdp_data = pd.read_csv(r'A:\UET-VNU\Covid19_KNK\Datacovid19\worldbank_data_g20_vietnam_2021_2022.csv')

# Mã hóa các quốc gia
        mapping = {
            'AR': 'ARG', 'AU': 'AUS', 'BR': 'BRA', 'CA': 'CAN', 'CN': 'CHN', 'DE': 'DEU',
            'EU': 'EU', 'FR': 'FRA', 'GB': 'GBR', 'ID': 'IDN', 'IN': 'IND', 'IT': 'ITA',
            'JP': 'JPN', 'KR': 'KOR', 'MX': 'MEX', 'RU': 'RUS', 'SA': 'SAU', 'TR': 'TUR',
            'US': 'USA', 'VN': 'VNM', 'ZA': 'ZAF'
        }
        gdp_data['iso_code'] = gdp_data['Country'].map(mapping)

        # Hàm chuẩn bị dữ liệu
        def prepare_data(year):
            gdp_unemployment = gdp_data[gdp_data['Year'] == year][['iso_code', 'GDP Growth (%)', 'Unemployment Rate (%)']]
            covid_year = covid_data[covid_data['date'] <= f'{year}-12-31']
            total_cases = covid_year.groupby('iso_code')['total_cases'].max().reset_index().rename(columns={'total_cases': 'Total Cases'})
            merged_data = total_cases.merge(gdp_unemployment, on='iso_code', how='inner').sort_values(by='Total Cases', ascending=False)
            return merged_data

        # Chuẩn bị dữ liệu cho 2021 và 2022
        merged_data_2021 = prepare_data(2021)
        merged_data_2022 = prepare_data(2022)

        # Vẽ biểu đồ cho 2021
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(merged_data_2021['iso_code'], merged_data_2021['Total Cases'], color='skyblue', label='Total Cases')
        ax1.set_xlabel('Country', fontsize=12)
        ax1.set_ylabel('Total Cases (log scale)', fontsize=12)
        ax1.set_yscale('log')
        ax1.set_xticks(np.arange(len(merged_data_2021['iso_code'])))
        ax1.set_xticklabels(merged_data_2021['iso_code'], rotation=45, ha='right', fontsize=10)
        ax1.set_title('COVID-19 Total Cases in 2021', fontsize=16)

        # Biểu đồ đường (GDP Growth)
        ax2 = ax1.twinx()
        ax2.plot(merged_data_2021['iso_code'], merged_data_2021['GDP Growth (%)'], color='red', marker='o', label='GDP Growth (%)')
        ax2.set_ylabel('GDP Growth (%)', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')

        # Biểu đồ đường (Unemployment Rate)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(merged_data_2021['iso_code'], merged_data_2021['Unemployment Rate (%)'], color='green', marker='s', label='Unemployment Rate (%)')
        ax3.set_ylabel('Unemployment Rate (%)', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='green')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax3.legend(loc='upper center')

        # Hiển thị biểu đồ 2021 trong Streamlit
        st.title('COVID-19 Total Cases, GDP Growth, and Unemployment Rate in 2021')
        st.pyplot(fig1)

        # Vẽ biểu đồ cho 2022
        fig2, ax4 = plt.subplots(figsize=(10, 6))
        ax4.bar(merged_data_2022['iso_code'], merged_data_2022['Total Cases'], color='skyblue', label='Total Cases')
        ax4.set_xlabel('Country', fontsize=12)
        ax4.set_ylabel('Total Cases (log scale)', fontsize=12)
        ax4.set_yscale('log')
        ax4.set_xticks(np.arange(len(merged_data_2022['iso_code'])))
        ax4.set_xticklabels(merged_data_2022['iso_code'], rotation=45, ha='right', fontsize=10)
        ax4.set_title('COVID-19 Total Cases in 2022', fontsize=16)

        # Biểu đồ đường (GDP Growth)
        ax5 = ax4.twinx()
        ax5.plot(merged_data_2022['iso_code'], merged_data_2022['GDP Growth (%)'], color='red', marker='o', label='GDP Growth (%)')
        ax5.set_ylabel('GDP Growth (%)', fontsize=12)
        ax5.tick_params(axis='y', labelcolor='red')

        # Biểu đồ đường (Unemployment Rate)
        ax6 = ax4.twinx()
        ax6.spines['right'].set_position(('outward', 60))
        ax6.plot(merged_data_2022['iso_code'], merged_data_2022['Unemployment Rate (%)'], color='green', marker='s', label='Unemployment Rate (%)')
        ax6.set_ylabel('Unemployment Rate (%)', fontsize=12)
        ax6.tick_params(axis='y', labelcolor='green')

        ax4.legend(loc='upper left')
        ax5.legend(loc='upper right')
        ax6.legend(loc='upper center')

        # Hiển thị biểu đồ 2022 trong Streamlit
        st.title('COVID-19 Total Cases, GDP Growth, and Unemployment Rate in 2022')
        st.pyplot(fig2)
elif main_option == "Việt Nam":
    st.header("Phân tích tình hình tại Việt Nam")

    # Các tùy chọn cho phần Việt Nam
    sub_option = st.selectbox(
        "Chọn phân tích chi tiết:",
        [
            "Biểu đồ giai đoạn áp dụng với số ca nhiễm/tử vong",
            "Heatmap bản đồ Việt Nam",
        ],
    )

    if sub_option == "Biểu đồ giai đoạn áp dụng với số ca nhiễm/tử vong":
        st.subheader("Biểu đồ giai đoạn áp dụng với số ca nhiễm/tử vong")
        # TODO: Vẽ biểu đồ giai đoạn

        # Chuyển cột 'date' thành kiểu datetime nếu chưa có
        vietnam_data['date'] = pd.to_datetime(vietnam_data['date'])

        # Tạo các giai đoạn
        giai_doan_1 = vietnam_data[(vietnam_data['date'] >= '2020-01-01') & (vietnam_data['date'] <= '2020-12-31')]
        giai_doan_2 = vietnam_data[(vietnam_data['date'] >= '2021-01-01') & (vietnam_data['date'] <= '2021-06-30')]
        giai_doan_3 = vietnam_data[(vietnam_data['date'] >= '2021-07-01') & (vietnam_data['date'] <= '2021-12-31')]
        giai_doan_4 = vietnam_data[(vietnam_data['date'] >= '2022-01-01')]

        # Tạo biểu đồ ca nhiễm
        fig, ax = plt.subplots(figsize=(12, 6))

        # Vẽ các đường cho từng giai đoạn ca nhiễm
        ax.plot(giai_doan_1['date'], giai_doan_1['new_cases'], label='Giai đoạn 1 - Ca nhiễm', color='blue')
        ax.plot(giai_doan_2['date'], giai_doan_2['new_cases'], label='Giai đoạn 2 - Ca nhiễm', color='orange')
        ax.plot(giai_doan_3['date'], giai_doan_3['new_cases'], label='Giai đoạn 3 - Ca nhiễm', color='green')
        ax.plot(giai_doan_4['date'], giai_doan_4['new_cases'], label='Giai đoạn 4 - Ca nhiễm', color='red')

        # Thêm mốc phân chia các giai đoạn bằng các đường dọc
        ax.axvline(x=pd.to_datetime('2021-01-01'), color='black', linestyle='--', label='Giai đoạn 2 bắt đầu')
        ax.axvline(x=pd.to_datetime('2021-07-01'), color='black', linestyle='--', label='Giai đoạn 3 bắt đầu')
        ax.axvline(x=pd.to_datetime('2022-01-01'), color='black', linestyle='--', label='Giai đoạn 4 bắt đầu')

        # Định dạng trục x (ngày tháng)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Đặt khoảng cách giữa các mốc thời gian là 3 tháng
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Định dạng ngày tháng
        fig.autofmt_xdate()  # Xoay chữ cho dễ đọc

        # Thêm các nhãn và tiêu đề
        ax.set_xlabel('Ngày', fontsize=12)
        ax.set_ylabel('Số ca nhiễm mới', fontsize=12)
        ax.set_title('Số ca nhiễm mới COVID-19 tại Việt Nam theo các giai đoạn', fontsize=14)

        # Thêm chú thích và hiển thị
        ax.legend(loc='upper left')
        plt.tight_layout()

        # Hiển thị biểu đồ trong Streamlit
        st.title('Biểu đồ số ca nhiễm COVID-19 tại Việt Nam')
        st.pyplot(fig)

        # Tạo biểu đồ ca tử vong
        fig, ax = plt.subplots(figsize=(12, 6))

        # Vẽ các đường cho từng giai đoạn ca tử vong
        ax.plot(giai_doan_1['date'], giai_doan_1['new_deaths'], label='Giai đoạn 1 - Tử vong', color='blue')
        ax.plot(giai_doan_2['date'], giai_doan_2['new_deaths'], label='Giai đoạn 2 - Tử vong', color='orange')
        ax.plot(giai_doan_3['date'], giai_doan_3['new_deaths'], label='Giai đoạn 3 - Tử vong', color='green')
        ax.plot(giai_doan_4['date'], giai_doan_4['new_deaths'], label='Giai đoạn 4 - Tử vong', color='red')

        # Thêm mốc phân chia các giai đoạn bằng các đường dọc
        ax.axvline(x=pd.to_datetime('2021-01-01'), color='black', linestyle='--', label='Giai đoạn 2 bắt đầu')
        ax.axvline(x=pd.to_datetime('2021-07-01'), color='black', linestyle='--', label='Giai đoạn 3 bắt đầu')
        ax.axvline(x=pd.to_datetime('2022-01-01'), color='black', linestyle='--', label='Giai đoạn 4 bắt đầu')

        # Định dạng trục x (ngày tháng)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Đặt khoảng cách giữa các mốc thời gian là 3 tháng
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Định dạng ngày tháng
        fig.autofmt_xdate()  # Xoay chữ cho dễ đọc

        # Thêm các nhãn và tiêu đề
        ax.set_xlabel('Ngày', fontsize=12)
        ax.set_ylabel('Số ca tử vong mới', fontsize=12)
        ax.set_title('Số ca tử vong COVID-19 tại Việt Nam theo các giai đoạn', fontsize=14)

        # Thêm chú thích và hiển thị
        ax.legend(loc='upper left')
        plt.tight_layout()

        # Hiển thị biểu đồ trong Streamlit
        st.title('Biểu đồ số ca tử vong COVID-19 tại Việt Nam')
        st.pyplot(fig)
    elif sub_option == "Heatmap bản đồ Việt Nam":
        st.subheader("Heatmap bản đồ Việt Nam")
        # TODO: Vẽ heatmap bản đồ


        # Dữ liệu COVID-19 đầy đủ
        covid_data = pd.DataFrame({
            'Province': ['ha noi', 'tp. ho chi minh', 'hai phong', 'nghe an', 'bac giang', 'vinh phuc',
                         'hai duong', 'quang ninh', 'bac ninh', 'thai nguyen', 'phu tho', 'binh duong',
                         'nam dinh', 'thai binh', 'hung yen', 'hoa binh', 'lao cai', 'thanh hoa',
                         'dak lak', 'lang son', 'yen bai', 'son la', 'ca mau', 'tuyen quang', 'tay ninh',
                         'binh dinh', 'quang binh', 'ha giang', 'khanh hoa', 'binh phuoc', 'ba ria - vung tau',
                         'da nang', 'dong nai', 'ninh binh', 'vinh long', 'ben tre', 'cao bang', 'lam dong',
                         'ha nam', 'dien bien', 'quang tri', 'bac kan', 'can tho', 'lai chau', 'tra vinh',
                         'dak nong', 'gia lai', 'ha tinh', 'binh thuan', 'dong thap', 'quang ngai',
                         'long an', 'quang nam', 'thua thien hue', 'bac lieu', 'phu yen', 'kien giang',
                         'an giang', 'tien giang', 'soc trang', 'kon tum', 'hau giang', 'ninh thuan'],
            'Cases': [1646923, 629018, 537527, 502049, 391440, 375686, 372391, 356404, 353869,
                      347519, 331520, 325667, 301101, 296789, 244028, 239941, 188846, 178595,
                      172439, 160752, 158046, 153602, 147734, 147582, 140444, 139890, 129648,
                      122610, 122036, 120003, 110822, 108712, 107518, 104800, 103505, 99799,
                      99051, 98238, 91467, 90757, 86293, 77048, 76925, 75519, 75174, 73427,
                      70961, 55279, 54300, 51614, 50513, 50297, 49556, 48186, 46949, 44481,
                      43659, 43297, 39902, 34457, 26342, 17900, 9001],
            'Deaths': [1232, 19985, 138, 145, 92, 19, 117, 150, 136,
                       112, 97, 3519, 150, 23, 5, 102, 38, 109, 201, 85,
                       13, 0, 357, 14, 944, 281, 74, 80, 364, 224, 497,
                       338, 1903, 93, 829, 490, 59, 141, 65, 23, 38, 19,
                       958, 0, 115, 47, 116, 50, 482, 1007, 127, 1088, 150,
                       173, 470, 137, 1027, 1325, 1290, 634, 1, 231, 60]
        })

        # Đọc file GeoJSON
        vietnam_map = gpd.read_file(r'A:\UET-VNU\Covid19_KNK\Datacovid19\diaphantinhenglish.geojson')

        # Chuẩn hóa tên tỉnh thành
        vietnam_map['Name'] = vietnam_map['Name'].apply(lambda x: unidecode.unidecode(x).lower())

        # Gộp dữ liệu
        merged_data = vietnam_map.merge(covid_data, left_on='Name', right_on='Province', how='left')

        # Điền giá trị NaN bằng 0
        merged_data['Cases'] = merged_data['Cases'].fillna(0)
        merged_data['Deaths'] = merged_data['Deaths'].fillna(0)

        # Vẽ biểu đồ số ca nhiễm
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        merged_data.plot(
            column='Cases',
            cmap='Reds',
            linewidth=0.8,
            ax=ax1,
            edgecolor='0.8',
           legend=True,
           legend_kwds={'label': "Số ca nhiễm"}
        )
        ax1.set_title('Heatmap Số ca nhiễm COVID-19 theo tỉnh tại Việt Nam', fontsize=16)
        ax1.axis('off')

        # Vẽ biểu đồ số ca tử vong
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        merged_data.plot(
            column='Deaths',
            cmap='Blues',
            linewidth=0.8,
            ax=ax2,
            edgecolor='0.8',
            legend=True,
            legend_kwds={'label': "Số ca tử vong"}
        )
        ax2.set_title('Heatmap Số ca tử vong COVID-19 theo tỉnh tại Việt Nam', fontsize=16)
        ax2.axis('off')

        # Hiển thị trong Streamlit
        st.title('Biểu đồ COVID-19 tại Việt Nam')

        # Hiển thị các biểu đồ riêng biệt
        st.subheader('Biểu đồ số ca nhiễm')
        st.pyplot(fig1)

        st.subheader('Biểu đồ số ca tử vong')
        st.pyplot(fig2)

