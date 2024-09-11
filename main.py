import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

GCI = pd.read_excel("C:/Users/minth/PythonProject/GCI_Dataset_2006-2016_20170116.xlsx", engine = "openpyxl")

GCI = GCI[2:]
newColumns = GCI.iloc[0]
newColumns.name = ""
GCI.columns = newColumns

GCI = GCI[1:]
GCI.drop(GCI.columns[-8:], axis=1, inplace=True)

newGCI = pd.DataFrame({}, columns=GCI.columns)
newGCI = GCI[GCI["Attribute"] == "Value"]

newGCI.index = newGCI["Placement"]
newGCI.drop(["Placement", "Series"], axis=1, inplace=True)

Countries = newGCI.columns[6:]
Countries.name = ""
clust_data = pd.DataFrame({}, index=Countries)

for i in range(len(newGCI.index)):
    input_data = []
    if newGCI.iloc[i, 1] != "2016-2017":
        break
    else:
        input_data = newGCI.iloc[i].drop(newGCI.columns[0:6]).values
    clust_data[newGCI.iloc[i, 4]] = input_data
    
columns_to_use = clust_data.columns
clust_data = clust_data.apply(pd.to_numeric, errors='coerce').fillna(0)
clust_data.dropna(inplace=True)

# 데이터 타입을 float로 변환
for column in columns_to_use:
    clust_data[column] = clust_data[column].astype(float)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(clust_data)

# 엘보우 기법을 사용하여 최적의 클러스터 개수를 찾습니다
wcss = []  # within-cluster sum of squares (클러스터 내 제곱합)

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# 결과를 그래프로 시각화합니다
plt.plot(range(1, 11), wcss)
plt.title('엘보우 기법')
plt.xlabel('클러스터 개수')
plt.ylabel('WCSS')
plt.show()


optimal_clusters = 4

kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_data)

clust_data['Cluster'] = clusters
# PCA를 사용하여 3차원으로 변환
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)

# 다양한 각도에서 시각화
angles = [(30, 30), (45, 45), (60, 60), (90, 90)]
colors = ['#1f77b4', '#ff7f0e', '#ffff00', '#e377c2']
fig = plt.figure(figsize=(20, 15))

for i, angle in enumerate(angles):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    for cluster in range(optimal_clusters):
        ax.scatter(pca_data[clusters == cluster, 0], pca_data[clusters == cluster, 1], pca_data[clusters == cluster, 2], label=f'Cluster {cluster}', color=colors[cluster])
    ax.set_title(f'Angle {angle}')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.view_init(angle[0], angle[1])
    ax.legend()

plt.show()

cluster_2_countries = clust_data[clust_data['Cluster'] == 2].index.values
print(cluster_2_countries)

# 독립 변수와 종속 변수 설정
X = clust_data.drop(columns=['Public trust in politicians, 1-7 (best)'])
y = clust_data['Public trust in politicians, 1-7 (best)']

# 상수항 추가
X = sm.add_constant(X)

# 회귀 모델 적합
model = sm.OLS(y, X).fit()

# 결과 요약 출력
print(model.summary())

# p-값이 0.05보다 작은 지수들만 선택
significant_features = model.pvalues[model.pvalues < 0.05].index
significant_features = significant_features.drop('const', errors='ignore')  # 상수항 제거

# 유의미한 지수들 출력
print("P>|t| 값이 0.05보다 작은 지수들:")
print(significant_features.values)

new_df = newGCI.set_index(['Edition', 'Series unindented']).drop(columns=['Dataset', 'GLOBAL ID', 'Code GCR', 'Attribute'])
new_df.fillna(0, inplace=True)
new_df = new_df[cluster_2_countries]
to_get_keys = significant_features.values.copy()
to_get_keys = np.append(to_get_keys, "Public trust in politicians, 1-7 (best)")
new_df = new_df.loc[np.isin(new_df.index.get_level_values(1), to_get_keys)]
new_df

correlation_data = {}
indicators = new_df[(new_df.index.get_level_values('Edition') == "2016-2017") & (new_df.index.get_level_values('Series unindented') != 'Public trust in politicians, 1-7 (best)')].index.get_level_values('Series unindented').unique()
for country in new_df.columns:
    country_data = new_df[country]
    public_trust = country_data.xs('Public trust in politicians, 1-7 (best)', level='Series unindented')
    correlations = {}
    for indicator in indicators:
        indicator_data = country_data.xs(indicator, level='Series unindented')
        try:
            correlation = public_trust.corr(indicator_data)
            correlations[indicator] = correlation
        except:
            correlations[indicator] = 0
    correlation_data[country] = correlations

# 상관계수 데이터프레임 생성
correlation_df = pd.DataFrame(correlation_data).transpose()
#sorted_index = correlation_df.loc['Korea, Rep.'].sort_values(ascending=False).index
sorted_columns = correlation_df.mean().sort_values(ascending=False).index
correlation_df = correlation_df[sorted_columns]

# 히트맵 시각화
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_df.iloc[:, :10], annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Public Trust in Politicians and Other Indicators')
plt.xlabel('Indicators')
plt.ylabel('Countries')
plt.show()
