# 非寿险准备金评估智能 Agent 初版

本项目是课程大作业的无 API 初版，实现了从 Excel 赔案数据到准备金模型结果与规则型 Agent 解释的完整流程。

## 功能

- 读取 Excel 赔案数据
- 自动生成 Paid / Incurred 累计赔付进展三角
- 计算 Chain Ladder、Expected Loss Ratio、Bornhuetter-Ferguson
- 输出准备金、最终赔款和模型对比结果
- 展示准备金柱状图、最终赔款折线图
- 基于规则生成数据诊断、模型建议和结果解释
- 可选接入 DeepSeek API 生成增强解释

## 运行方式

在项目根目录运行：

```powershell
python -m pip install -r reserve_agent/requirements.txt
streamlit run reserve_agent/app.py
```

如果使用 Codex bundled Python，请把命令中的 `python` 替换为对应 Python 路径。

## 数据说明

默认读取根目录下的 `Chapter 08 - Data sets - Examples.xlsx`，并使用 `Claims data` 工作表构建累计赔付三角。系统也支持在界面中上传新的 `.xlsx` 文件。

## 后续 API 增强

系统支持可选 DeepSeek API 增强解释。不要把真实 API key 写进代码或报告。

方式一：在项目根目录新建 `.env` 文件：

```env
DEEPSEEK_API_KEY=你的DeepSeekKey
DEEPSEEK_MODEL=deepseek-chat
```

方式二：启动系统后，在左侧栏的密码框临时输入 key。

如果没有 key，系统会自动回退到规则型 Agent 解释。

