import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.indicators.distance_indicator import derive_ideal_and_nadir_from_pf

# 文件路径和读取数据
file_path1 = r"C:\Users\26689\Desktop\chaotic.xlsx"
data1 = pd.read_excel(file_path1, sheet_name='Pareto_Pop300_Ngen300')

file_path2 = r"C:\Users\26689\Desktop\mopso.xlsx"
data2 = pd.read_excel(file_path2, sheet_name='Pareto_Pop300_Ngen300')

file_path3 = r"C:\Users\26689\Desktop\nsga2.xlsx"
data3 = pd.read_excel(file_path3, sheet_name='Pareto_Pop300_Ngen300')

file_path4 = r"C:\Users\26689\Desktop\moead.xlsx"
data4 = pd.read_excel(file_path4, sheet_name='Pareto_Pop300_Ngen300')

# 提取目标值
data1 = data1[['delta_V', 'delta_D']].values
data2 = data2[['delta_V', 'delta_D']].values
data3 = data3[['delta_V', 'delta_D']].values
data4 = data4[['delta_V', 'delta_D']].values

# 定义参考点（通常取目标值的最小值和最大值之外的点）
reference_point = np.array([1200, 0.09])  # 假设目标值范围在 [0, 1] 之间

# 初始化指标
hv = HV(ref_point=reference_point)
spacing_indicator = SpacingIndicator()

# 计算 Diversity Metric △
def calculate_diversity_metric(data):
    # 计算每个解之间的欧几里得距离
    distances = np.linalg.norm(data[:-1] - data[1:], axis=1)
    d_mean = np.mean(distances)
    df = np.linalg.norm(data[0] - data[1])
    dl = np.linalg.norm(data[-1] - data[-2])
    numerator = np.sum(np.abs(distances - d_mean)) + df + dl
    denominator = np.sum(distances) + df + dl
    return numerator / denominator

# 评估每个算法的帕累托前沿
algorithms = ['Chaotic', 'MOPSO', 'NSGA-II', 'MOEA/D']
data_list = [data1, data2, data3, data4]

print("Algorithm\tHV\tSpacing\tDiversity Metric\tNumber of Solutions")
for name, data in zip(algorithms, data_list):
    # 计算超体积指标
    hv_value = hv(data)
    
    # 计算间距指标
    spacing_value = spacing_indicator(data)
    
    # 计算 Diversity Metric △
    diversity_metric = calculate_diversity_metric(data)
    
    # 计算解的数量
    num_solutions = len(data)
    
    # 计算理想点和纳什点
    ideal_point, nadir_point = derive_ideal_and_nadir_from_pf(data)

    print(f"{name}\t{hv_value:.4f}\t{spacing_value:.4f}\t{diversity_metric:.4f}\t{num_solutions}")
 #   print(ideal_point, nadir_point)