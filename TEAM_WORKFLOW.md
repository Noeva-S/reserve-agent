# 团队协作说明

## 本地运行

每个队友都下载完整仓库，然后在项目根目录运行：

```powershell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
streamlit run reserve_agent/app.py
```

浏览器打开：

```text
http://localhost:8501
```

## 分工边界

- 组长：主要维护 `reserve_agent/app.py`，负责合并和部署。
- 队友 A：修改 `reserve_agent/data/detector.py` 和 `reserve_agent/data/adapters.py`，负责 Excel 格式自动识别。
- 队友 B：修改 `reserve_agent/data/mapping.py` 和 `reserve_agent/data/validator.py`，负责字段映射和数据校验。
- 队友 C：修改 `reserve_agent/agent/chat_agent.py`、`reserve_agent/agent/prompts.py` 和 `reserve_agent/agent/context_builder.py`，负责实时问答 Agent。
- 队友 D：修改 `reserve_agent/exports/`，负责 Excel、Word、ZIP 下载。

## 注意事项

- 不要把 `.env`、`.streamlit/secrets.toml`、`.venv` 上传到 GitHub。
- 不要把 DeepSeek API key 写进代码。
- 每个人尽量只改自己负责的文件夹，减少合并冲突。
- 改完后先本地运行 `streamlit run reserve_agent/app.py`，确认网页能打开。
