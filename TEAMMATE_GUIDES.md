# 开发说明

每个人都需要下载完整项目，因为网页运行依赖整个 `reserve_agent` 文件夹；但每个人只改自己负责的几个文件，避免互相冲突。

## 所有人先做的准备

### 1. 下载代码

推荐使用 Git：

```powershell
git clone https://github.com/Noeva-S/reserve-agent.git
cd reserve-agent
```

如果暂时不会 Git，也可以在 GitHub 页面点击：

```text
Code -> Download ZIP -> 解压
```

但是后面要合并代码时，最好还是用 Git 分支提交。

### 2. 安装 Python 依赖

建议 Python 版本为 3.10 或以上。在项目根目录运行：

```powershell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果提示 `python` 找不到，可以试：

```powershell
py -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 本地启动网页

在项目根目录运行：

```powershell
streamlit run reserve_agent/app.py
```

浏览器打开：

```text
http://localhost:8501
```

正常情况下，网页应该在几十秒内打开。不要运行整个文件夹，也不要用 `python reserve_agent/app.py` 启动。

### 4. 本地 DeepSeek API 设置

如果需要测试 AI 对话功能，在本地新建：

```text
.streamlit/secrets.toml
```

内容写：

```toml
DEEPSEEK_API_KEY = "自己的 API key"
```

注意：不要把 `.streamlit/secrets.toml` 上传到 GitHub，也不要把 API key 写进任何 `.py` 文件。

### 5. 建议的 Git 工作方式

每个人先建自己的分支：

```powershell
git checkout -b feature-自己的任务名
```

比如：

```powershell
git checkout -b feature-excel-detect
```

修改完成后：

```powershell
git status
git add reserve_agent/自己负责的文件夹
git commit -m "说明自己完成了什么"
git push origin feature-自己的任务名
```

然后在 GitHub 上创建 Pull Request，由组长统一合并。

## 队友 A：Excel 格式自动识别

### 你的目标

现在系统主要能识别固定格式的 `Claims data` 工作表。你的任务不是只判断“第一行表头”的整齐 Excel，而是要让系统先找到真正的数据区域，再判断上传的 Excel 是哪种格式。

实际 Excel 可能前几行是标题、说明、单位、空行，也可能一个 sheet 里有几块表。所以你的识别流程应该尽量变成：

```text
先扫描有效表格区域 -> 再识别表格格式 -> 再转换成准备金模型能用的三角形
```

你要支持或逐步支持的格式包括：

```text
1. 当前这种赔案明细格式：Claim ID / Loss Year / Type / 2007 / 2008 ...
2. 已经整理好的三角形格式：事故年 × 发展期
3. 长表格式：事故年 / 发展期 / 金额
4. 无法识别的格式
```

### 你主要改这些文件

```text
reserve_agent/data/detector.py
reserve_agent/data/adapters.py
reserve_agent/data/table_scanner.py
reserve_agent/data/loader.py
```

建议优先改：

```text
reserve_agent/data/table_scanner.py
reserve_agent/data/detector.py
```

里面有这个函数：

```python
def detect_excel_format(df):
    ...
```

`table_scanner.py` 里有这些函数：

```python
find_candidate_table_regions(raw_df)
slice_region_with_header(raw_df, region)
```

它们的作用是处理“表头不在第一行”“前面有说明行”“sheet 里有多块表”这类情况。

它应该返回：

```text
"claims_snapshot"
"triangle"
"long_table"
"unknown"
```

### 你可以怎么开发

第一步：准备几个不同格式的 Excel 表。不要只用默认数据，至少准备一个“前 3 行是说明，第 4 行才是表头”的测试表。

第二步：先用 `header=None` 读取原始 sheet：

```python
raw = pd.read_excel(file, sheet_name=sheet_name, header=None)
```

第三步：用 `find_candidate_table_regions(raw)` 找候选表格区域，再用 `slice_region_with_header(raw, region)` 切出真正的数据表。

第四步：在 `detector.py` 里根据列名和表格形状判断格式。比如：

```python
if {"claim id", "loss year", "type"}.issubset(columns):
    return "claims_snapshot"
```

第五步：在 `adapters.py` 里完善转换函数：

```python
triangle_sheet_to_triangle(df)
long_table_to_triangle(df)
```

不管用户上传什么格式，最终都要转换成这样的三角形：

```text
index = 事故年
columns = 发展期
values = 累计赔款金额
```

### 你怎么测试

先启动网页：

```powershell
streamlit run reserve_agent/app.py
```

然后上传不同格式的 Excel，看系统有没有识别出来、有没有正常生成三角形。

也可以在命令行快速测试：

```powershell
python -c "import pandas as pd; from reserve_agent.data.detector import detect_excel_format; df=pd.read_excel('你的测试文件.xlsx'); print(detect_excel_format(df))"
```

### 交付标准

你完成后应该能说明：

```text
1. 是否支持表头不在第一行的 Excel
2. 是否支持前面有说明行、空行的 Excel
3. 支持了哪些 Excel 格式
4. 每种格式需要哪些字段
5. 无法识别时会返回什么
6. 上传测试文件后网页是否能正常运行
```

## 队友 B：字段映射和数据校验

### 你的目标

如果系统无法自动识别 Excel，用户应该可以手动选择字段。例如：

```text
哪一行是表头？
哪一块区域是真正的数据表？
事故年是哪一列？
发展期是哪一列？
金额是哪一列？
数据是累计赔款还是增量赔款？
```

注意：不要假设用户上传的 Excel 一定从第一行开始就是规整表格。队友 A 会做自动扫描，但自动扫描不一定每次都准；你的手动映射功能就是兜底。

同时，你要检查数据有没有明显问题，例如：

```text
空值
负数
重复的事故年 + 发展期
发展期不连续
金额列不是数字
三角形为空
```

### 你主要改这些文件

```text
reserve_agent/data/mapping.py
reserve_agent/data/validator.py
reserve_agent/data/adapters.py
reserve_agent/data/table_scanner.py
```

其中 `mapping.py` 里有：

```python
class FieldMapping:
    ...
```

`validator.py` 里有：

```python
def validate_triangle(triangle):
    ...
```

### 你可以怎么开发

第一步：完善 `FieldMapping`，记录用户选了哪些列：

```text
accident_year_col
development_col
amount_col
measure_col
is_cumulative
```

第二步：完善 `validate_triangle(triangle)`，返回问题列表。

目前返回的是：

```python
ValidationIssue(level="warning", message="...")
```

`level` 可以用：

```text
"info"
"warning"
"error"
```

第三步：后续可以让组长把这些校验结果显示到网页上。

### 你怎么测试

可以用命令行测试：

```powershell
python -c "import pandas as pd; from reserve_agent.data.validator import validate_triangle; tri=pd.DataFrame({0:[100,200],1:[150,None]}); print(validate_triangle(tri))"
```

也可以故意做一个有负数、空值、重复事故年的表，看校验结果是否合理。

### 交付标准

你完成后应该能说明：

```text
1. 如果自动识别失败，用户应该怎么选择表头行或字段
2. 系统能检查哪些数据问题
3. 哪些问题只是提醒，哪些问题会阻止建模
4. 字段不完整时，网页应该给用户什么提示
```

## 队友 C：Agent 实时对话

### 你的目标

现在系统可以生成一段解释。你的任务是把解释区升级成“可以继续问问题”的聊天式 Agent。

用户可以问：

```text
为什么某一年准备金最高？
Chain Ladder 和 BF 方法有什么区别？
这个结果有哪些风险？
如果期望赔付率调高会怎样？
模型结果适合写进报告吗？
```

### 你主要改这些文件

```text
reserve_agent/agent/chat_agent.py
reserve_agent/agent/prompts.py
reserve_agent/agent/context_builder.py
reserve_agent/agent/llm_client.py
```

重点是：

```python
def answer_user_question(question, context, api_key, chat_history=None):
    ...
```

### 你可以怎么开发

第一步：完善 `prompts.py`，让提示词更适合准备金课程项目。

第二步：完善 `context_builder.py`，确保只把必要内容发给 DeepSeek，例如：

```text
数据质量摘要
准备金总额
各事故年模型结果
发展因子
当前选择的方法
```

不要把完整 Excel 原始数据全部发给 API。

第三步：在 Streamlit 页面里可以用这些组件：

```python
st.chat_message(...)
st.chat_input(...)
st.session_state
```

如果需要改页面，先和组长沟通，因为 `app.py` 是主入口。

### 你怎么测试

先确认本地有：

```text
.streamlit/secrets.toml
```

内容是：

```toml
DEEPSEEK_API_KEY = "自己的 API key"
```

然后运行：

```powershell
streamlit run reserve_agent/app.py
```

打开网页后，测试几个问题：

```text
解释一下准备金最高的事故年
用更通俗的话解释 BF 方法
给我一段可以写进报告的模型结果解释
指出这个模型有什么局限
```

### 交付标准

你完成后应该能说明：

```text
1. 是否可以连续追问
2. 是否保留聊天历史
3. API key 不在代码里
4. 回答是否基于当前模型结果，而不是凭空编造
```

## 队友 D：下载导出功能

### 你的目标

给网页增加下载功能，让用户可以把结果保存下来。建议至少支持：

```text
1. 下载模型结果 Excel
2. 下载 Agent 解释 Word
3. 下载打包 ZIP
4. 如果时间够，再支持图表图片下载
```

### 你主要改这些文件

```text
reserve_agent/exports/export_excel.py
reserve_agent/exports/export_word.py
reserve_agent/exports/export_zip.py
```

目前已有这些函数：

```python
build_excel_download(outputs, triangle, report)
build_word_report(explanation_text)
build_zip_package(files)
```

### 你可以怎么开发

第一步：完善 Excel 导出内容。建议 Excel 里至少有这些 sheet：

```text
Triangle
Model Results
Data Quality
Development Factors
```

第二步：完善 Word 报告格式。建议包括：

```text
标题
数据诊断
模型结果摘要
Agent 解释
注意事项
```

第三步：完善 ZIP，把 Excel 和 Word 一起打包。

第四步：让组长把下载按钮接入网页：

```python
st.download_button(...)
```

### 你怎么测试

可以先用命令行测试函数是否能生成文件：

```powershell
python -c "from reserve_agent.exports.export_zip import build_zip_package; data=build_zip_package({'test.txt': b'hello'}); print(len(data))"
```

也可以启动网页：

```powershell
streamlit run reserve_agent/app.py
```

让组长接入按钮后，点击下载，看文件能不能正常打开。

### 交付标准

你完成后应该能说明：

```text
1. 可以下载哪些文件
2. 每个文件里包含哪些内容
3. 下载后的 Excel / Word 是否能正常打开
4. 文件名是否清楚，例如 reserve_results.xlsx、agent_report.docx
```

## 常见问题

### 运行很久怎么办

正常启动网页不应该超过几分钟，模型计算通常是秒级。如果很久没反应，先检查：

```text
1. 是不是正在 pip install，而不是运行网页
2. 是不是误运行了整个文件夹
3. 是不是把 .venv 文件夹一起下载了
4. 是否使用了 streamlit run reserve_agent/app.py
```

### 改了代码网页没变化怎么办

可以按顺序试：

```text
1. 刷新浏览器页面
2. 停止终端里的 Streamlit，重新运行
3. 检查是否改错文件
4. 看终端有没有报错
```

### 出现 ModuleNotFoundError 怎么办

确认自己在项目根目录运行：

```powershell
streamlit run reserve_agent/app.py
```

不要进入 `reserve_agent` 文件夹里面运行。

### 代码合并前要注意什么

每个人提交前至少做三件事：

```powershell
streamlit run reserve_agent/app.py
git status
```

确认网页能打开，确认没有把 `.env`、`.streamlit/secrets.toml`、`.venv`、测试 Excel 大文件误提交。
