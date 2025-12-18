import torch
from typing import Tuple, List
from scene.gaussian_model import GaussianModel
import torch.nn.functional as F
from scene.cameras import Camera


def collect_theta_params(gaussians: GaussianModel, pbr_kwargs: dict | None) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def collect_params(opti: torch.optim.Optimizer) -> Tuple[List[torch.Tensor], List[float]]:
        params = []
        lrs = []
        for group in opti.param_groups:
            # 检查参数组中是否包含optimized_in_IsAS字段
            if "optimized_in_IsAS" in group:
                if group["optimized_in_IsAS"]:
                    params.extend(group["params"])
                    lrs.extend([group["lr"]] * len(group["params"]))
            else:
                # 如果没有optimized_in_IsAS字段,默认保留
                params.extend(group["params"])
                lrs.extend([group["lr"]] * len(group["params"]))
        return params, lrs

    # collect gaussian parameters
    opti = gaussians.optimizer
    params, lrs = collect_params(opti)
    theta_params.extend(params)
    theta_lrs.extend(lrs)

    # collect env light map parameters
    for comp in pbr_kwargs.values():
        try:
            opti = comp.optimizer
            params, lrs = collect_params(opti)
            theta_params.extend(params)
            theta_lrs.extend(lrs)
        except:
            pass

    theta_params = torch.cat(theta_params, dim=0)
    theta_lr = torch.tensor(theta_lrs, device=theta_params.device)
    print("Total number of parameters: ", theta_params.shape)
    print("Learning rates shape: ", theta_lr.shape)
    return theta_params, theta_lr


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

    W, H = I_Theta.shape[-2:]
    I_flat = I_Theta.reshape(-1)
    num_pixels = I_flat.numel()  # 3*W*H
    num_theta = theta.numel()

    dI_dtheta_flat = torch.zeros(
        num_pixels, num_theta, device=I_Theta.device, dtype=I_Theta.dtype
    )

    grad = torch.autograd.grad(
        outputs=[I_flat],
        inputs=[theta],
        retain_graph=False,
        create_graph=False,  # 不需要高阶导数
        allow_unused=True,  # 允许无贡献的参数（避免报错）
    )[0]

    dI_dtheta = None
    if grad is not None:
        # reshape grad
        dI_dtheta = grad.reshape(3, W, H, -1)  # 3, H, W, m_theta
        # clear cache
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("梯度计算失败，请检查参数θ的形状和梯度追踪设置")

    return dI_dtheta


def compute_dLoss_dI(Loss: torch.Tensor, I_Theta: torch.Tensor) -> torch.Tensor:
    """
    calculate the gradient of Loss with respect to I_Theta

    Args:
        Loss: scalar tensor, the loss to be minimized
        I_Theta: tensor, the image to be optimized
    Returns:
        dLoss_dI: tensor, the gradient of Loss with respect to I_Theta
    """
    I_flat = I_Theta.reshape(-1)

    grad = torch.autograd.grad(
        outputs=[Loss],
        inputs=[I_flat],
        retain_graph=False,
        create_graph=False,  # 不需要高阶导数
        allow_unused=True,  # 允许无贡献的参数（避免报错）
    )[0]

    dLoss_dI = None
    if grad is not None:
        # reshape
        dLoss_dI = grad.reshape(I_Theta.shape)  # 3, H, W
        # clear cache
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("梯度计算失败，请检查参数θ的形状和梯度追踪设置")

    return dLoss_dI


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
    pj = pj_4d.squeeze(0)   # [3, H, W]
    pj = torch.clamp(pj / pj.sum().clamp_min(1e-8), min=1e-8)  # 最小概率限制为1e-8
    return pj


def sample_pixels_from_pj(
    pj, num_samples=1024, replacement=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample pixels from a probability map.

    Args:
        pj: tensor, the probability map
        num_samples: int, the number of pixels to sample
        replacement: bool, whether to sample with replacement
    Returns:
        sampled_pixels: tensor, the sampled pixels
    """
    W, H = pj.shape
    pj_flat = pj.reshape(-1)  # [H_full*W_full,]
    # 2. 采样展平索引（multinomial按概率采样，输入需为非负且和为1）
    # 注：multinomial要求输入sum=1，我们已归一化，直接采样
    sampled_indices = torch.multinomial(
        pj_flat,
        num_samples=num_samples,
        replacement=replacement,
        generator=None,  # 可选：指定随机种子保证可复现
    )
    y = sampled_indices // W
    x = sampled_indices % W
    sampled_prob = pj_flat[sampled_indices]  # [N,]
    sampled_points = torch.stack([y, x], dim=1)  # [N, 2]
    return sampled_points, sampled_prob


def pixel2ray(
    cam: Camera, pixel_coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert pixel coordinates to ray directions in camera space.

    Args:
        cam: Camera, the camera object
        pixel_coords: tensor, the pixel coordinates
    Returns:
        ray_dirs: tensor, the ray directions in camera space
        ray_origin: tensor, the ray origins in camera space
    """
    W, H = cam.image_width, cam.image_height
    fx, fy = cam.fx, cam.fy  # 焦距
    cx, cy = cam.cx, cam.cy  # 主点

    # 像素坐标归一化到相机平面（NDC）
    x = pixel_coords[:, 1]  # x列
    y = pixel_coords[:, 0]  # y行
    x_normalized = (x - cx) / fx
    y_normalized = (y - cy) / fy

    # 相机坐标系下的射线方向（假设相机朝向-z轴）
    ray_dirs_cam = torch.stack(
        [x_normalized, -y_normalized, -torch.ones_like(x)], dim=1
    )

    # 转换为世界坐标系（相机外参：旋转+平移）
    ray_dirs_world = torch.matmul(cam.R, ray_dirs_cam.unsqueeze(-1)).squeeze(-1)
    ray_origin = cam.camera_center

    return ray_dirs_world, ray_origin
