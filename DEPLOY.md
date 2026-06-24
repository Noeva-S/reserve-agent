# 云端部署说明

本项目可以部署到 Streamlit Community Cloud，从而获得一个不依赖本机开机的公网链接。

## 推荐部署步骤

1. 将本项目上传到 GitHub 仓库。
2. 打开 https://share.streamlit.io/ 并用 GitHub 登录。
3. 点击 `New app`。
4. 选择本项目仓库。
5. Main file path 填：

```text
reserve_agent/app.py
```

6. 在 `Advanced settings` 或 App 的 `Secrets` 中新增一个名为 `DEEPSEEK_API_KEY` 的密钥，值填写你的 DeepSeek API key。

7. 点击 Deploy。

## 安全提醒

- 不要把真实 API key 写进 GitHub 仓库。
- 本项目已支持从 Streamlit Secrets 读取 `DEEPSEEK_API_KEY`。
- `.env` 和 `.streamlit/secrets.toml` 已被 `.gitignore` 忽略。

