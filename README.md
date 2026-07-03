# 更易用的 AI Toolkit-XPU版本（中文 README）

> 原项目作者：Ostris
> 
> AI-Toolkit-Easy2use项目作者：DocWorkBox
> 
> 本项目是基于[（I卡）AI-Toolkit安装教程｜ARC.AI.Next｜腾讯频道](https://pd.qq.com/g/mojr1c1n4v/post/B_6f843f69d1e10a001441152196989399570X60)  （停止更新） 进行二次开发。

## 特别说明
- 项目是利用AI开发服务于intel显卡，其他类型显卡暂不支持。
- 测试设备：Arc A770 16G + i5-11600KF + 32G内存
### 原有支持项
- lodestones/Chroma1-Base
- ostris/Flex.1-alpha
- ostris/Flex.2-preview
- black-forest-labs/FLUX.1-dev
- black-forest-labs/FLUX.2-dev
- HiDream-ai/HiDream-I1-Full
- Alpha-VLLM/Lumina-Image-2.0
- OmniGen2/OmniGen2
- Qwen/Qwen-Image
- stable-diffusion-v1-5/stable-diffusion-v1-5
- stabilityai/stable-diffusion-xl-base-1.0
- Tongyi-MAI/Z-Image-Turbo

- black-forest-labs/FLUX.1-Kontext-dev
- HiDream-ai/HiDream-E1-1
- Qwen/Qwen-Image-Edit
- Qwen/Qwen-Image-Edit-2509

- Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- Wan-AI/Wan2.1-T2V-14B-Diffusers
- Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
- Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
- ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16
- ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16
- Wan-AI/Wan2.2-TI2V-5B-Diffusers
  
### 以下是在原有的基础上新增的支持
- Flux.2-klein-base-4B
- Flux.2-klein-base-9B(未测试)

## 项目简介

- 面向扩散模型（Diffusion Models）的训练与推理一体化工具。
- 支持常见的图像与视频模型，适配消费级硬件。
- 提供命令行（CLI）与 Web 用户界面（UI），上手门槛低同时功能完备。

## 环境要求

- Python ≥ 3.10
- Git（用于拉取仓库）
- NVIDIA GPU（显存需满足你的训练或推理需求）
- Python 虚拟环境（建议）
- Node.js ≥ 18（用于运行 Web UI）

## 安装（Linux / Windows）

> 🐳 **推荐**：如需使用 Docker 快速部署，请查看 [Docker 部署指南](DOCKER_README.md)。

### 1）克隆仓库

```bash
git clone https://github.com/xhq2025/ai-toolkit-xpu.git
cd ai-toolkit-xpu
```

### 2）创建并激活虚拟环境

Linux / macOS：

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows（PowerShell）：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3）安装 PyTorch（示例，CUDA 12.6 对应版本）

根据你的 CUDA / 显卡环境调整版本。以下为参考示例：

```bash
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

### 4）安装项目依赖

```bash
pip install -r requirements.txt
```

## 运行 UI（中文界面）

### 环境要求

- Node.js ≥ 18

UI 为基于 Next.js 的 Web 应用。UI 无需持续运行即可执行训练任务，仅在启动/停止/监控任务时需要使用。

### 开发模式

开发模式运行在 `http://localhost:3000`：

```bash
cd ui
npm install
npm run dev
```

打开浏览器访问：

- `http://localhost:3000/`（首页）
- `http://localhost:3000/dashboard`（仪表盘）
- `http://localhost:3000/jobs/new`（新建任务）

### 生产环境

生产环境运行在端口 `8675`。以下命令将安装/更新 UI 及其依赖并启动 UI：

```bash
cd ui
npm run build_and_start
```

启动后可通过以下地址访问：

- `http://localhost:8675`（本地访问）
- `http://<your-ip>:8675`（服务器部署时的远程访问）

> **注意**：UI 无需持续运行即可执行训练任务。UI 仅用于启动、停止和监控任务。

## 中文版 UI 截图

![仪表盘（中文）](ui/public/screenshots/dashboard_zh.png)
![新建任务（中文）](ui/public/screenshots/jobs_new_zh.png)


## 常见问题（FAQ）

- 显存不足如何处理？
  - 训练大型模型时，如遇显存限制，可在配置中开启低显存选项（例如 `low_vram: true`），或在 CPU 上量化部分模块以降低显存占用。
- Windows 环境安装遇到困难？
  - 建议优先确认 Python、CUDA 与驱动版本匹配；也可以考虑使用 WSL（Windows Subsystem for Linux）以获得更稳定的依赖环境。
- UI 无法访问或接口报错？
  - 请检查 Node.js 版本（≥18）、依赖是否安装完成（`npm install`）、以及开发服务器是否正常运行（`npm run dev`）。

## 许可证

本仓库遵循原项目的许可证政策（如有变更将于此处更新）。请在商用或分发前确认模型及数据集的独立许可证要求。
