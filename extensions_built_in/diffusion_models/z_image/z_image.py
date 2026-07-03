import os
# 导入类型注解，用于代码类型提示：List列表、Optional可选类型
from typing import List, Optional

# huggingface模型仓库下载工具
import huggingface_hub
# PyTorch深度学习核心库
import torch
# yaml配置文件读写库
import yaml
# 从训练框架导入各类配置数据类
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
# 框架自定义专用LoRA网络实现类
from toolkit.lora_special import LoRASpecialNetwork
# 所有模型统一继承的基础父类
from toolkit.models.base_model import BaseModel
# 工具函数：刷新控制台输出缓存
from toolkit.basic import flush
# 提示词嵌入封装对象，统一管理正负向文本特征
from toolkit.prompt_utils import PromptEmbeds
# 自定义Flow Matching流匹配欧拉采样调度器
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
# 加速器工具：去除DDP/分布式包装，拿到原始模型
from toolkit.accelerator import unwrap_model
# quanto量化库，freeze冻结量化后的权重防止更新
from optimum.quanto import freeze
# 量化相关工具函数
from toolkit.util.quantize import quantize, get_qtype, quantize_model
# 分层显存卸载管理器，实现低显存模型加载
from toolkit.memory_management import MemoryManager
# safetensors安全权重加载，避免pickle安全漏洞
from safetensors.torch import load_file

# transformers：分词器 + Qwen3大模型作为文本编码器
from transformers import AutoTokenizer, Qwen3ForCausalLM
# diffusers内置VAE变分自编码器
from diffusers import AutoencoderKL

# 尝试导入ZImage专属流水线与Transformer主干模型
try:
    from diffusers import ZImagePipeline
    from diffusers.models.transformers import ZImageTransformer2DModel
# 导入失败则抛出版本错误提示
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )

# 训练用流匹配调度器全局固定配置
scheduler_config = {
    "num_train_timesteps": 1000,       # 训练总时间步数量
    "use_dynamic_shifting": False,     # 关闭动态时间偏移缩放
    "shift": 3.0,                      # 固定时间偏移系数
}

# ZImage模型封装类，继承训练框架基础模型类
class ZImageModel(BaseModel):
    arch = "zimage"  # 模型架构标识，框架内部区分不同模型类型

    def __init__(
        self,
        device,                          # 模型运行设备(cuda/cpu)
        model_config: ModelConfig,       # 模型全局配置对象，包含路径、量化、显存、LoRA等参数
        dtype="bf16",                    # 模型默认计算精度
        custom_pipeline=None,            # 自定义流水线实例，预留扩展
        noise_scheduler=None,            # 外部传入的噪声调度器，训练时会覆盖
        **kwargs,                        # 其余扩展参数透传给父类构造函数
    ):
        # 调用父类BaseModel构造，初始化基础属性、设备、配置、精度等
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True        # 标记模型为Flow Matching流匹配架构
        self.is_transformer = True           # 标记主干网络为Transformer，非UNet
        self.target_lora_modules = ["ZImageTransformer2DModel"]  # LoRA生效的目标模型模块

    # 静态工具方法：获取训练阶段专用噪声调度器
    @staticmethod
    def get_train_scheduler():
        # 使用全局预设配置实例化流匹配欧拉离散调度器
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    # 获取图像尺寸对齐除数，图片宽高必须是该数值整数倍
    def get_bucket_divisibility(self):
        # 16是VAE下采样倍数，2是Transformer Patch分块尺寸，相乘得到最小对齐单位
        return 16 * 2

    # 加载训练辅助LoRA适配器，训练时抵消、推理时生效
    def load_training_adapter(self, transformer: ZImageTransformer2DModel):
        self.print_and_status_update("Loading assistant LoRA")
        lora_path = self.model_config.assistant_lora_path  # 读取配置中辅助LoRA路径
        # 判断路径不存在，则判定为HF Hub线上仓库地址
        if not os.path.exists(lora_path):
            lora_splits = lora_path.split("/")
            # Hub标准格式 repo_id/文件名 必须两段，拆分后长度不等于3则格式错误
            if len(lora_splits) != 3:
                raise ValueError(
                    f"Assistant LoRA path {lora_path} is not a valid local path or hub path."
                )
            repo_id = "/".join(lora_splits[:2])  # 提取仓库ID：用户名/模型名
            filename = lora_splits[2]            # 提取仓库内权重文件名
            try:
                # 从HF Hub下载权重到本地缓存，返回本地真实路径
                lora_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                )
                # 更新配置内LoRA路径为本地缓存路径
                self.model_config.assistant_lora_path = lora_path
            except Exception as e:
                raise ValueError(
                    f"Failed to download assistant LoRA from {lora_path}: {e}"
                )
        # 使用safetensors加载LoRA权重字典
        lora_state_dict = load_file(lora_path)
        # 读取LoRA秩dim：取第一层注意力k矩阵A权重的第一维
        dim = int(
            lora_state_dict[
                "diffusion_model.layers.0.attention.to_k.lora_A.weight"
            ].shape[0]
        )

        new_sd = {}
        # 遍历原始LoRA权重key，把前缀diffusion_model.替换为transformer.适配当前模型
        for key, value in lora_state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        lora_state_dict = new_sd  # 替换转换后的权重字典

        # 组装LoRA网络基础配置
        network_config = {
            "type": "lora",
            "linear": dim,
            "linear_alpha": dim,
            "transformer_only": True,
        }
        # 转换为框架标准NetworkConfig配置对象
        network_config = NetworkConfig(**network_config)
        # 修改LoRA前缀，适配ZImage Transformer命名规则
        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        # 实例化专用LoRA网络，仅作用于Transformer主干
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            is_assistant_adapter=True,  # 标记该LoRA为训练辅助适配器
        )
        # 将LoRA网络挂载到Transformer主干上，不作用于文本编码器
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        self.print_and_status_update("Merging in assistant LoRA")
        # 将LoRA网络移动到指定设备与精度
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        # 更新LoRA权重缩放系数
        network._update_torch_multiplier()
        # 加载转换完成的LoRA权重
        network.load_weights(lora_state_dict)
        # 将LoRA权重融合进Transformer主干
        network.merge_in(merge_weight=1.0)
        # 标记未完成融合，推理阶段会反向抵消效果
        network.is_merged_in = False
        # 保存辅助LoRA实例到模型类变量，采样器会读取该适配器
        self.assistant_lora: LoRASpecialNetwork = network
        # 训练时乘数设为-1，抵消LoRA原有效果
        self.assistant_lora.multiplier = -1.0
        self.assistant_lora.is_active = False  # 训练阶段禁用LoRA
        # 标记推理时需要反转辅助LoRA乘数，还原正常效果
        self.invert_assistant_lora = True

    # 完整加载ZImage全套模型：Transformer、文本编码器、VAE、流水线
    def load_model(self):
        dtype = self.torch_dtype  # 获取全局模型精度
        self.print_and_status_update("Loading ZImage model")
        model_path = self.model_config.name_or_path          # Transformer主干模型路径
        base_model_path = self.model_config.extras_name_or_path  # 整套模型基础路径(TE/VAE)

        self.print_and_status_update("Loading transformer")

        transformer_path = model_path
        transformer_subfolder = "transformer"  # 默认子文件夹名
        # 如果传入路径是本地文件夹
        if os.path.exists(transformer_path):
            transformer_subfolder = None
            transformer_path = os.path.join(transformer_path, "transformer")
            te_folder_path = os.path.join(model_path, "text_encoder")
            # 文件夹包含text_encoder则是完整模型包，更新基础路径
            if os.path.exists(te_folder_path):
                base_model_path = model_path
        # 从本地/HF Hub加载ZImage Transformer主干
        transformer = ZImageTransformer2DModel.from_pretrained(
            transformer_path, subfolder=transformer_subfolder, torch_dtype=dtype
        )
        # 配置存在辅助LoRA路径时，加载训练适配器
        if self.model_config.assistant_lora_path is not None:
            self.load_training_adapter(transformer)
            # qfloat8统一改为float8适配量化逻辑
            if self.model_config.qtype == "qfloat8":
                self.model_config.qtype = "float8"
        # 开启量化则执行Transformer量化
        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()
        # 开启分层显存卸载，且Transformer卸载比例大于0
        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            # 绑定显存管理器，分层卸载部分Transformer层到CPU
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )
        # 低显存模式下先把Transformer移到CPU节省显存
        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")
        flush()

        self.print_and_status_update("Text Encoder")
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype
        )
        # 加载Qwen3文本编码器大模型
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
        )
        # 文本编码器分层显存卸载
        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )
        # 文本编码器移动到目标计算设备
        text_encoder.to(self.device_torch, dtype=dtype)
        flush()
        # 开启文本编码器量化
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)  # 冻结量化权重，训练不更新TE
            flush()

        self.print_and_status_update("Loading VAE")
        # 加载VAE解码器
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype
        )
        # 初始化训练用流匹配调度器
        self.noise_scheduler = ZImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")
        kwargs = {}
        # 初始化ZImage文生图流水线，先空传TE/Transformer后续赋值
        pipe: ZImagePipeline = ZImagePipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        # 量化场景下分开赋值效果更好
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")
        text_encoder = [pipe.text_encoder]  # 封装为列表兼容多TE逻辑
        tokenizer = [pipe.tokenizer]        # 分词器同步封装列表
        # 非低显存模式将Transformer移至计算设备
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()
        # 确认文本编码器设备正确，冻结权重不参与训练
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()
        # 保存各组件到类成员变量，全局调用
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    # 获取推理专用生成流水线
    def get_generation_pipeline(self):
        scheduler = ZImageModel.get_train_scheduler()
        # 实例化推理流水线，unwrap去除分布式包装拿到原始模型
        pipeline: ZImagePipeline = ZImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )
        # 流水线整体移动到计算设备
        pipeline = pipeline.to(self.device_torch)
        return pipeline

    # 单张图像生成推理函数
    def generate_single_image(
        self,
        pipeline: ZImagePipeline,               # 推理流水线实例
        gen_config: GenerateImageConfig,        # 生成参数配置(尺寸、步数、CFG等)
        conditional_embeds: PromptEmbeds,        # 正向提示词嵌入
        unconditional_embeds: PromptEmbeds,      # 负面提示词嵌入
        generator: torch.Generator,             # 随机种子生成器
        extra: dict,                            # 额外透传参数
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        self.model.to(self.device_torch)
        # 保证模型顶层buffer/常量张量在正确设备，解决分层卸载带来的设备不匹配
        self._ensure_buffers_on_device(self.model, self.device_torch)
        sc = self.get_bucket_divisibility()  # 获取尺寸对齐除数
        # 将宽高向下取整为对齐单位整数倍，避免VAE/Transformer尺寸报错
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)
        # 调用流水线生成图片，取第一张输出图像
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return img

    # 工具函数：同步模型顶层buffer与参数到目标设备，修复分层卸载设备错位
    def _ensure_buffers_on_device(self, model: torch.nn.Module, device: torch.device):
        """
        Ensure all direct buffers and parameters on the model are on the correct device.
        This is needed because MemoryManager.to() only moves tracked submodules,
        not direct buffers like x_pad_token.
        """
        # 遍历模型直属buffer(不递归子模块)，移至目标设备
        for name, buf in model.named_buffers(recurse=False):
            if buf.device != device:
                setattr(model, name, buf.to(device))
        # 遍历模型直属参数(不递归子模块)，移至目标设备
        for name, param in model.named_parameters(recurse=False):
            if param.device != device:
                param.data = param.data.to(device)

    # 训练前向：获取Transformer噪声预测输出
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,   # 输入潜变量
        timestep: torch.Tensor,              # 原始0~1000时间步
        text_embeddings: PromptEmbeds,       # 文本提示词嵌入特征
        **kwargs,
    ):
        self.model.to(self.device_torch)
        # 同步顶层buffer设备，解决分层卸载设备不匹配报错
        self._ensure_buffers_on_device(self.model, self.device_torch)
        # 扩展维度适配Transformer输入格式
        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))
        # 将0~1000时间步转换为0~1流匹配输入时间值
        timestep_model_input = (1000 - timestep) / 1000
        # Transformer前向传播，获取模型输出
        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_model_input,
            text_embeddings.text_embeds,
        )[0]
        # 转换输出精度为float并堆叠恢复batch维度
        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
        # 去除多余维度
        noise_pred = noise_pred.squeeze(2)
        # Flow Matching架构取负得到最终噪声预测
        noise_pred = -noise_pred
        return noise_pred

    # 编码提示词，生成文本嵌入封装对象
    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        # 文本编码器不在计算设备时先移过去
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)
        # 流水线编码提示词，不做CFG分离
        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=self.device_torch,
        )
        # 封装为统一PromptEmbeds对象返回
        pe = PromptEmbeds([prompt_embeds, None])
        return pe

    # 判断主干模型是否开启梯度更新，ZImage训练默认冻结主干返回False
    def get_model_has_grad(self):
        return False

    # 判断文本编码器是否开启梯度更新，默认冻结TE返回False
    def get_te_has_grad(self):
        return False

    # 保存训练后的Transformer模型，同时输出训练元数据yaml
    def save_model(self, output_path, meta, save_dtype):
        transformer: ZImageTransformer2DModel = unwrap_model(self.model)
        # 保存Transformer权重到输出目录transformer子文件夹，使用safetensors格式
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        # 写入训练元数据yaml文件
        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    # 获取损失计算目标值(Flow Matching损失目标)
    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        # 损失目标 = 原始噪声 - 输入潜变量，分离计算图不参与梯度
        return (noise - batch.latents).detach()

    # 返回模型版本标识
    def get_base_model_version(self):
        return "zimage"

    # 返回Transformer块层级前缀名称，用于LoRA/梯度分层控制
    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    # 保存LoRA前转换权重key：transformer. -> diffusion_model. 兼容原始模型命名
    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    # 加载LoRA前转换权重key：diffusion_model. -> transformer. 适配代码内命名
    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd