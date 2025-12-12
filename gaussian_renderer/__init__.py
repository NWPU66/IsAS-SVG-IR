from gaussian_renderer.render import render
from gaussian_renderer.svgss import render_svgss
from gaussian_renderer.IsAS_svgss import render_IsAS_svgss


render_fn_dict = {
    "render": render,
    # "render_relight": render_svgss,
    "render_relight": render_IsAS_svgss,
}