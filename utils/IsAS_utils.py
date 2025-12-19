import torch
from typing import Tuple, List
from scene.gaussian_model import GaussianModel
import torch.nn.functional as F
from scene.cameras import Camera


def collect_theta_params(
    gaussians: GaussianModel, pbr_kwargs: dict | None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect all parameters from the gaussian model and the pbr model.

    Args:
        gaussians (GaussianModel): Gaussian model.
        pbr_kwargs (dict): PBR model parameters.
    Returns:
        torch.Tensor: All parameters.
    """
    print("Collecting parameters from gaussian and env light map ...")
    theta_params = []
    theta_lrs = []

    def collect_params(
        opti: torch.optim.Optimizer,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        params = []
        lrs = []
        for group in opti.param_groups:
            if "optimized_in_IsAS" in group and not group["optimized_in_IsAS"]:
                continue
            for param in group["params"]:
                if param.requires_grad:
                    params.append(param)
                    lrs.append(torch.ones_like(param, device=param.device) * group["lr"])
        return params, lrs

    # collect gaussian parameters
    opti = gaussians.optimizer
    params, lrs = collect_params(opti)
    theta_params.extend(params)
    theta_lrs.extend(lrs)

    # collect env light map parameters
    for comp in pbr_kwargs.values():
        opti = comp.optimizer
        params, lrs = collect_params(opti)
        theta_params.extend(params)
        theta_lrs.extend(lrs)
    
    # flatten tensors
    assert len(theta_params) == len(theta_lrs), "number of parameters and learning rates do not match."
    for idx, (param, lr) in enumerate(zip(theta_params, theta_lrs)):
        assert param.shape == lr.shape, f"shape of parameter {idx} and learning rate {idx} do not match."
        theta_params[idx] = param.reshape(-1)
        theta_lrs[idx] = lr.reshape(-1)

    theta_params = torch.cat(theta_params, dim=0)
    theta_lr = torch.cat(theta_lrs, dim=0)
    print("Total number of parameters: ", theta_params.shape)
    print("Learning rates shape: ", theta_lr.shape)
    return theta_params, theta_lr


def compute_dLoss_dI(Loss: torch.Tensor, I_Theta: torch.Tensor) -> torch.Tensor:
    """
    用backward()计算完整梯度（显存优化版），解决分Chunk无梯度+内存溢出问题
    Args:
        Loss: 标量损失Tensor（cuda）
        I_Theta: 图像Tensor，shape [C, H, W]（cuda）
    Returns:
        dLoss_dI: 完整梯度，shape [C, H, W]
    """
    
    I_Theta.requires_grad_(True)
    if I_Theta.grad is not None:
        I_Theta.grad.zero_()  
    if Loss.grad is not None:
        Loss.grad.zero_()

    torch.backends.cudnn.benchmark = False  # 关闭基准测试，减少显存碎片
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32，加速+减显存
    torch.cuda.empty_cache()

    Loss.backward(
        retain_graph=False,  # 计算完梯度后释放计算图
        inputs=[I_Theta]     # 仅计算I_Theta的梯度
    )

    # 4. 获取梯度并校验
    dLoss_dI = I_Theta.grad
    if dLoss_dI is None:
        raise RuntimeError("I_Theta.grad 为None！请检查：1.I_Theta是否参与Loss计算；2.是否被detach()")
    assert dLoss_dI.shape == I_Theta.shape, f"梯度形状不匹配！预期{I_Theta.shape}，实际{dLoss_dI.shape}"

    I_Theta.grad = None  # 清空梯度，释放显存
    torch.cuda.empty_cache()
    return dLoss_dI


def compute_dI_dtheta(
    I_Theta: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the gradient of I with respect to theta.

    Args:
        I_Theta (torch.Tensor): I(θ).
        theta (torch.Tensor): All parameters.
    Returns:
        torch.Tensor: dI/dθ.
    """
    assert theta.requires_grad, "参数θ必须开启梯度追踪 (requires_grad=True)"

    if I_Theta.grad is not None:
        I_Theta.grad.zero_()  # 清空梯度
    if theta.grad is not None:
        theta.grad.zero_()

    dI_dtheta = torch.autograd.grad(
        outputs=[I_Theta],
        inputs=[theta],
        retain_graph=False,
        create_graph=False,  # 不需要高阶导数
        allow_unused=True,  # 允许无贡献的参数（避免报错）
    )[0]    # 3, H, W, m_theta

    if dI_dtheta is not None:
        torch.cuda.empty_cache()    # clear cache
    else:
        raise RuntimeError("梯度计算失败，请检查参数θ的形状和梯度追踪设置")

    return dI_dtheta


def upsampling_half_pj(
    half_pj: torch.Tensor, scale_factor=2, mode="bilinear"
) -> torch.Tensor:
    """
    Upsample half_pj to full resolution.

    Args:
        half_pj: tensor, the half resolution image to be upsampled
        scale_factor: int, the scale factor for upsampling
        mode: str, the interpolation mode for upsampling
    Returns:
        full_pj: tensor, the full resolution image
    """

    half_pj_4d = half_pj.unsqueeze(0)  # [1, 3, H_half, W_half]
    pj_4d = F.interpolate(
        half_pj_4d,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=(
            True if mode == "bilinear" else None
        ),  # bilinear需要指定align_corners
    )
    pj = pj_4d.squeeze(0)  # [3, H, W]
    pj = torch.clamp(pj / pj.sum().clamp_min(1e-8), min=1e-8)  # 最小概率限制为1e-8
    return pj
