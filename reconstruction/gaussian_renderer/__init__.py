"""Differentiable rendering for deformable 2D Gaussian surfels."""

import math

import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from scene.flexible_deform_model import GaussianModel
from utils.sh_utils import eval_sh


def _deform_primitives(pc: GaussianModel, time: float):
    means = pc.get_xyz
    scales = pc._scaling
    rotations = pc._rotation
    opacities = pc._opacity
    deform_mask = pc._deformation_table

    if deform_mask.any():
        deformed_means, deformed_scales, deformed_rotations = pc.deformation(
            means[deform_mask],
            scales[deform_mask],
            rotations[deform_mask],
            torch.as_tensor(time, device=means.device),
        )

        with torch.no_grad():
            pc._deformation_accum[deform_mask] += torch.abs(
                deformed_means - means[deform_mask]
            )

    final_means = means.clone()
    final_scales = scales.clone()
    final_rotations = rotations.clone()
    if deform_mask.any():
        final_means[deform_mask] = deformed_means
        final_scales[deform_mask] = deformed_scales
        final_rotations[deform_mask] = deformed_rotations

    return (
        final_means,
        pc.scaling_activation(final_scales),
        pc.rotation_activation(final_rotations),
        pc.opacity_activation(opacities),
    )


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """Render RGB and geometric buffers for one camera and timestamp."""
    screenspace_points = torch.zeros_like(
        pc.get_xyz, requires_grad=True, device=pc.get_xyz.device
    )
    screenspace_points.retain_grad()
    device = pc.get_xyz.device
    view_matrix = viewpoint_camera.world_view_transform.to(device)
    projection_matrix = viewpoint_camera.full_proj_transform.to(device)
    camera_center = viewpoint_camera.camera_center.to(device)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=view_matrix,
        projmatrix=projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means, scales, rotations, opacities = _deform_primitives(
        pc, viewpoint_camera.time
    )

    if pipe.compute_cov3D_python:
        raise NotImplementedError(
            "Precomputed covariance is not supported for deformable 2D surfels."
        )

    shs = None
    colors_precomp = None
    if override_color is not None:
        colors_precomp = override_color
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )
        directions = means - camera_center.repeat(
            pc.get_features.shape[0], 1
        )
        directions = directions / directions.norm(dim=1, keepdim=True)
        colors_precomp = torch.clamp_min(
            eval_sh(pc.active_sh_degree, shs_view, directions) + 0.5, 0.0
        )
    else:
        shs = pc.get_features

    rendered_image, radii, allmap = rasterizer(
        means3D=means,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    render_alpha = allmap[1:2]
    render_normal = allmap[2:5]
    render_normal = (
        render_normal.permute(1, 2, 0)
        @ view_matrix[:3, :3].T
    ).permute(2, 0, 1)
    render_depth_expected = torch.nan_to_num(
        allmap[0:1] / render_alpha, nan=0.0, posinf=0.0, neginf=0.0
    )
    render_depth_median = torch.nan_to_num(
        allmap[5:6], nan=0.0, posinf=0.0, neginf=0.0
    )
    render_depth = (
        render_depth_expected * (1.0 - pipe.depth_ratio)
        + render_depth_median * pipe.depth_ratio
    )

    return {
        "render": rendered_image,
        "depth": render_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "render_alpha": render_alpha,
        "render_normal": render_normal,
        "render_depth_median": render_depth_median,
        "render_depth_expected": render_depth_expected,
        "render_dist": allmap[6:7],
    }
