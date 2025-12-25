import torch
from typing import Tuple, List
from scene.gaussian_model import GaussianModel
import torch.nn.functional as F
from scene.cameras import Camera


def collect_theta_params(
    gaussians: GaussianModel, pbr_kwargs
) -> Tuple:
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
            print(f"Collecting parameters from group: name={group['name']}, lr={group['lr']}")
            for param in group["params"]:
                if param.requires_grad:
                    print("Collecting parameter size: ", param.size())
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
        try:
            opti = comp.optimizer
            params, lrs = collect_params(opti)
            theta_params.extend(params)
            theta_lrs.extend(lrs)
        except:
            pass
    
    # flatten tensors
    # assert len(theta_params) == len(theta_lrs), "number of parameters and learning rates do not match."
    # for idx, (param, lr) in enumerate(zip(theta_params, theta_lrs)):
    #     assert param.shape == lr.shape, f"shape of parameter {idx} and learning rate {idx} do not match."
    #     theta_params[idx] = param.reshape(-1)
    #     theta_lrs[idx] = lr.reshape(-1)

    # theta_params = torch.cat(theta_params, dim=0)
    # theta_lr = torch.cat(theta_lrs, dim=0)
    # print("Total number of parameters: ", theta_params.shape)
    # print("Learning rates shape: ", theta_lr.shape)
    # return theta_params, theta_lr

    return theta_params, theta_lrs


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
        retain_graph=True,
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


def compute_dI_dtheta_square_sum(
    I_Theta: torch.Tensor,
    theta: List[torch.Tensor],
    theta_lr: List[torch.Tensor],
) -> torch.Tensor:
    """
    calculate sum of square dI_dtheta
    
    :param I_Theta: shape [3, W, H]
    :param theta: shape [m_theta]
    :param theta_lr: shape [m_theta]
    :return: shape [3, W, H]
    """
    assert all([t.requires_grad for t in theta]), "参数θ必须开启梯度追踪 (requires_grad=True)"
    assert len(I_Theta.shape) == 3, f"I_Theta需为[C, H, W]形状，当前形状：{I_Theta.shape}"
    C, H, W = I_Theta.shape
    num_pixels = C * H * W

    I_flat = I_Theta.flatten()  # [num_pixels]
    theta_lr2 = [lr ** 2 for lr in theta_lr]     # [m_theta]
    dI_dtheta_flat = torch.zeros(num_pixels, device=I_Theta.device, dtype=I_Theta.dtype)

    for pixel_idx in range(num_pixels):
        print(f"pixel_idx: {pixel_idx}")
        I_flat[pixel_idx].backward(retain_graph=(pixel_idx != num_pixels - 1))

        for param, lr in zip(theta, theta_lr2):
            dI_dtheta_flat[pixel_idx] += torch.sum(torch.pow(param.grad.detach(), 2) * lr)
            param.grad.zero_()
    
    dI_dtheta = dI_dtheta_flat.reshape(C, H, W)
    del I_flat, dI_dtheta_flat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dI_dtheta



def compute_dI_dtheta(
    I_Theta: torch.Tensor,
    theta: torch.Tensor,
    block_size: int = 1024,  # 每块像素数，可根据显存调整
) -> torch.Tensor:
    """
    分块计算每个像素在I中对theta的梯度（解决显存爆炸问题）
    输出形状：[3, H, W, num_theta]（或展平为 [num_pixels, num_theta]）

    Args:
        I_Theta (torch.Tensor): I(θ)，形状 [C, H, W]（C, H, W）
        theta (torch.Tensor): 待求导参数，形状 [num_theta]
        block_size (int): 分块大小（每块的像素数量），默认600，可按需调整
    Returns:
        torch.Tensor: 每个像素对每个theta参数的梯度，形状 [3, H, W, num_theta]
    """
    assert theta.requires_grad, "参数θ必须开启梯度追踪 (requires_grad=True)"
    assert len(I_Theta.shape) == 3, f"I_Theta需为[C, H, W]形状，当前形状：{I_Theta.shape}"
    C, H, W = I_Theta.shape
    num_pixels = C * H * W
    num_theta = theta.numel()
    device = I_Theta.device
    dtype = I_Theta.dtype
    print(f"num_pixels: {num_pixels}, {I_Theta.shape}, num_theta: {num_theta}, 分块大小: {block_size}")

    I_flat = I_Theta.flatten()  # 形状 [num_pixels]

    dI_dtheta_flat = torch.zeros((num_pixels, num_theta), device=device, dtype=dtype)

    # 计算总块数
    num_blocks = (num_pixels + block_size - 1) // block_size 

    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min((block_idx + 1) * block_size, num_pixels)
        current_block_pixels = end_idx - start_idx 
        if current_block_pixels <= 0:
            break

        I_flat_block = I_flat[start_idx:end_idx]  # 形状 [current_block_pixels]

        # 2. 构造当前块的grad_outputs：current_block_pixels × current_block_pixels 单位矩阵
        #    仅针对当前块像素，显存占用为 current_block_pixels²，远小于原num_pixels²
        grad_outputs_block = torch.eye(
            current_block_pixels,
            device=device,
            dtype=dtype
        )

        # 3. 计算当前块的梯度：判断是否为最后一个块，决定是否保留计算图
        retain_graph = (block_idx != num_blocks - 1)
        current_grad = torch.autograd.grad(
            outputs=I_flat_block,
            inputs=theta,
            grad_outputs=grad_outputs_block,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )[0]

        # 4. 校验当前块梯度是否有效
        if current_grad is None:
            raise RuntimeError(f"第{block_idx}块梯度计算失败，请检查θ是否在I_Theta的计算图中")
        assert current_grad.shape == (current_block_pixels, num_theta), \
            f"当前块梯度形状异常，预期({current_block_pixels}, {num_theta})，实际{current_grad.shape}"

        dI_dtheta_flat[start_idx:end_idx, :] = current_grad

        del I_flat_block, grad_outputs_block, current_grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 形状还原：[num_pixels, num_theta] -> [C, H, W, num_theta]
    dI_dtheta = dI_dtheta_flat.reshape(C, H, W, num_theta)

    # 最终释放缓存
    del I_flat, dI_dtheta_flat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
