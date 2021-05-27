from matplotlib.colors import to_rgb
import ammolite


def init_pymol_parameters():
    ammolite.cmd.bg_color("white")
    ammolite.cmd.set("dash_gap", 0.3)
    ammolite.cmd.set("dash_width", 2.0)
    ammolite.cmd.set("ray_trace_mode", 3)
    ammolite.cmd.set("ray_trace_disco_factor", 1.0)
    ammolite.cmd.set("ray_shadows", 0)
    ammolite.cmd.set("spec_reflect", 0)
    ammolite.cmd.set("spec_power", 0)
    ammolite.cmd.set("depth_cue", 0)


COLORS = {
    "H": to_rgb("#ffffff"),
    "C": to_rgb("#767676"),
    "N": to_rgb("#0a6efd"),
    "O": to_rgb("#e1301d"),
    "P": to_rgb("#098a07"),
    #df059e
}