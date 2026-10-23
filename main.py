import argparse

import torch

from model import GaussianSplatting2D, get_mapped_qp
from utils.misc_utils import load_cfg


def get_gaussian_cfg(args):
    if args.qp > 0.0:  # The number of Gaussians, the feature step, and lambda are derived from the (mapped) QP
        gaussian_cfg = (f"qp-{args.qp:g}-at-{get_mapped_qp(args):.4g}-over-{args.qp_overestimate:g}_prune-x{args.prune_times:d}-"
                        f"max-{args.prune_max_ratio:g}")
        if args.qp_lambda_scale != 1.0:
            gaussian_cfg += f"-lam-x{args.qp_lambda_scale:g}"
        if args.prune_add_ratio > 0.0:
            gaussian_cfg += f"-add-{args.prune_add_ratio:g}"
    else:
        gaussian_cfg = f"num-{args.num_gaussians:d}"
        if args.prune_mode == "lambda":
            gaussian_cfg += f"_prune-lambda-{args.prune_lambda:g}-x{args.prune_times:d}-max-{args.prune_max_ratio:g}"
            if args.prune_add_ratio > 0.0:
                gaussian_cfg += f"-add-{args.prune_add_ratio:g}"
        elif args.prune_ratio > 0.0:
            gaussian_cfg += f"_prune-{args.prune_ratio:.2f}"
    if args.relocate_ratio > 0.0:
        gaussian_cfg += f"_reloc-{args.relocate_ratio:.2f}"
    if args.disable_inverse_scale:
        gaussian_cfg += f"_scale-{args.init_scale:.1f}"
    else:
        gaussian_cfg += f"_inv-{args.init_scale:.1f}"
    if args.init_scale_knn > 0.0:
        gaussian_cfg += f"-knn-{args.init_scale_knn:g}"
    if args.init_aspect > 1.0:
        gaussian_cfg += f"_aspect-{args.init_aspect:g}"
    if args.init_fine_ratio > 0.0:
        gaussian_cfg += f"_fine-{args.init_fine_ratio:g}"
    if args.add_color_init != "residual":
        gaussian_cfg += f"_add-{args.add_color_init}"
    if args.max_scale > 0.0:
        gaussian_cfg += f"_max-scale-{args.max_scale:g}"
    if args.quant_mode == "none":
        args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits, args.y_bits, args.uv_bits = 32, 32, 32, 32, 32, 32
    all_bits = [args.pos_bits, args.scale_bits, args.rot_bits, args.feat_bits, args.y_bits, args.uv_bits]
    if min(all_bits) < 4 or max(all_bits) > 32:
        raise ValueError(f"Bit precision must be between 4 and 32 but got: {', '.join(f'{bits:d}' for bits in all_bits)}")
    if args.quant_mode == "step":
        feat_step = "qp" if args.qp > 0.0 else f"{args.feat_step:g}"
        gaussian_cfg += f"_steps-{args.pos_step:g}-{args.scale_step:g}-{args.rot_step:g}-{feat_step}"
        if args.color_space == "yuv":
            gaussian_cfg += f"_yuv-{args.y_step:g}-{args.uv_step:g}" if args.qp == 0.0 else f"_yuv-{args.uv_step_ratio:g}"
            if args.qp > 0.0 and args.y_step_ratio != 1.0:
                gaussian_cfg += f"-y{args.y_step_ratio:g}"
    else:
        gaussian_cfg += f"_bits-{args.pos_bits:d}-{args.scale_bits:d}-{args.rot_bits:d}-{args.feat_bits:d}"
        if args.color_space == "yuv":
            gaussian_cfg += f"_yuv-{args.y_bits:d}-{args.uv_bits:d}"
    if args.quant_mode == "none" and (args.qp > 0.0 or args.prune_mode == "lambda"):  # The steps estimate the bits for lambda pruning
        feat_step = "qp" if args.qp > 0.0 else f"{args.feat_step:g}"
        gaussian_cfg += f"_rate-steps-{args.pos_step:g}-{args.scale_step:g}-{args.rot_step:g}-{feat_step}"
    if args.quant_mode != "none" and args.wrap_rot:
        gaussian_cfg += "_wrap-rot"
    if args.canonical_rot:
        gaussian_cfg += "_canon-rot"
    if args.quant_mode == "step" and args.feat_step_adapt != "none":
        gaussian_cfg += f"_adapt-{args.feat_step_adapt}-{args.feat_step_gamma:g}"
        if args.color_space == "yuv" and args.uv_step_gamma >= 0.0:
            gaussian_cfg += f"-uv-{args.uv_step_gamma:g}"
    if args.color_space == "yuv" and args.uv_weight != 1.0:
        gaussian_cfg += f"_uvw-{args.uv_weight:g}"
    if not args.disable_topk_norm:
        gaussian_cfg += f"_top-{args.topk:d}"
    gaussian_cfg += f"_{args.init_mode[0]}-{args.init_random_ratio:.1f}"
    return gaussian_cfg


def get_log_dir(args):
    gaussian_cfg = get_gaussian_cfg(args)
    loss_cfg = f"l1-{args.l1_loss_ratio:.1f}_l2-{args.l2_loss_ratio:.1f}_ssim-{args.ssim_loss_ratio:.1f}"
    folder = f"{gaussian_cfg}_{loss_cfg}"
    if args.downsample:
        folder += f"_ds-{args.downsample_ratio:.1f}"
    if not args.disable_lr_schedule:
        folder += f"_decay-{args.max_decay_times:d}-{args.decay_ratio:.1f}"
    if args.max_time > 0.0:
        folder += f"_time-{args.max_time:g}"
    if not args.disable_prog_optim:
        folder += "_prog"
    return f"{args.log_root}/{args.exp_name}/{folder}"


def main(args):
    args.log_dir = get_log_dir(args)
    ImageGS = GaussianSplatting2D(args)
    if args.eval:
        ImageGS.render(render_height=args.render_height)
    else:
        ImageGS.optimize()


if __name__ == "__main__":
    torch.hub.set_dir("models/torch")
    parser = argparse.ArgumentParser()
    parser = load_cfg(cfg_path="cfgs/default.yaml", parser=parser)
    arguments = parser.parse_args()
    main(arguments)
