# Electricity Pricing Data

此目录用于存放电价数据文件（如实时电价CSV、分时电价表等）。

## 当前实现方式

电价模型在 `experiment_params.yaml` 中配置为"分时电价"(time_of_use)模式：

```yaml
price_model_type: "time_of_use"

time_of_use_periods:
  peak: [8, 11, 18, 21]     # 峰时
  off_peak: [6, 7, 12, 17, 22, 23]  # 平时
  valley: [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 24]  # 谷时

time_of_use_prices:
  peak: 1.0       # 元/kWh
  off_peak: 0.6   # 元/kWh
  valley: 0.3      # 元/kWh
```

运行时通过 `src/utils/common.py` 中的 `compute_electricity_price()` 函数计算当前电价。

## 如需使用外部电价数据文件

1. 将CSV文件放入此目录，格式示例：
   ```csv
   hour,price
   0,0.3
   1,0.3
   ...
   8,1.0
   ...
   ```

2. 修改 `experiment_params.yaml` 中 `price_model_type` 为 `"from_file"` 并添加：
   ```yaml
   price_model_type: "from_file"
   electricity_data_file: "data/electricity/your_price_file.csv"
   ```

3. 更新 `src/utils/common.py` 中的 `compute_electricity_price()` 函数以支持文件读取模式。
