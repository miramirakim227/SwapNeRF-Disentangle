# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import warnings
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch
from dotmap import DotMap

import pdb 

def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    return parser


args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
device = util.get_cuda(args.gpu_id[0])

dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)
print(
    "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp)
)

# make_model: model์ ๋ํ option. 
net = make_model(conf["model"]).to(device=device)   # PixelNeRFNet


# conf['renderer']
# renderer {
#     n_coarse = 64
#     n_fine = 32
#     # Try using expected depth sample
#     n_fine_depth = 16
#     # Noise to add to depth sample
#     depth_std = 0.01
#     # Decay schedule, not used
#     sched = []
#     # White background color (false : black)
#     white_bkgd = True
# }

# Ours๋ก ๋ณ๊ฒฝ ์์?!     # from_config: ๋ชจ๋ธ ์ธํ ์๋?ค์ค    
renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp,).to(
    device=device       # NeRFRenderer -> renderer setting 
)

# Parallize         # net: pixelNeRF -> pixelNeRF๋ฅผ 
render_par = renderer.bind_parallel(net, args.gpu_id).eval()   # -> _RenderWrapper๋ฅผ ์?์ธํจ -> ์์ forward ํจ์๊ฐ class NeRFRenderer ์คํํ๋๊ฑฐ!
# self๊น์ง๋ ์์ฑ๋ฐ์๋ฒ๋ฆผ!
# renderer.bind_parallel -> _RenderWrapper(net, self, simple_output=simple_output)

nviews = list(map(int, args.nviews.split()))        # 1. 


class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=device)
                )

        self.z_near = dset.z_near       # ์ผ๋จ์ ๊ทธ๋ฅ ๋๊ธฐ 
        self.z_far = dset.z_far

        self.use_bbox = args.no_bbox_step > 0

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        #######################################################################################
        ################### ์ฌ๊ธฐ์๋ถํฐ ์ ์ง์คํด์ ์ฝ์ด๋ณด๊ธฐ! ray ๊ฐ์?ธ์ค๋ ๋ถ๋ถ!!! ########################
        #######################################################################################

        # SB: number of batches 
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape      # SB: number of obj, NV: number of view     -> 4, 50, 3, 128, 128
        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"]  # (SB)      # ๊ฐ batch sample๋ง๋ค์ focal length๊ฐ ์กด์ฌํจ 
        all_c = data.get("c")  # (SB)

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []

        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        if curr_nviews == 1:       # (0,) ์ batch size๋งํผ ๋ง๋ค์ด์ค๋ค!
            image_ord = torch.randint(0, NV, (SB, 1))   # ours -> ๊ณ์ nviews=1์ผ ์์?! 
        else: # Pass
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        
        ##### object๋ง๋ค์ Process 
        ##### ์ฌ๊ธฐ์๋ RGB samplingํ๋ ๊ณผ์?์ ์์ ๋นผ๊ณ?, extrinsic์ ํตํ camera ray๋ฅผ ๊ฐ์?ธ์ฌ ๊ฒ pix_inds๋ ํ์์์ 
        for obj_idx in range(SB):       # batch ์์ index๋ง๋ค pose๊ฐ ๋ค๋ฅด๊ธฐ ๋๋ฌธ!      # SB: 4     # meshgrid๋ง ๊ด์ฐฎ๋ค๋ฉด batch ์ฐ์ฐ์ผ๋ก ํผ์ง๋งํ๊ฒ ํ๋ฒ ๊ฐ๋ ๊ด์ฐฎ์๋ฏ 
            # batch size๋ ์์ ํธ, ๊ฐ sample์ ๋ํด์ ์ฒ๋ฆฌํจ 
            if all_bboxes is not None:              
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)       # (50, 3, 128, 128)
            poses = all_poses[obj_idx]  # (NV, 4, 4)            # (50, 4, 4)        # <- multi-view rotation
            focal = all_focals[obj_idx]
            c = None
            if "c" in data:
                c = data["c"][obj_idx]
            if curr_nviews > 1: # Pass
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )
            images_0to1 = images * 0.5 + 0.5

            # ใใ ๋ค ๋ฃ๊ณ? ๋ด๋ ๋? ๋ฏ. ์ด์ฐจํผ feature field์ ๋ํด์ ๋ณด๋๊ฑฐ๋ผ! 
            cam_rays = util.gen_rays(       # ์ฌ๊ธฐ์์ W, H ์ฌ์ด์ฆ๋ output target feature image์ resolution์ด์ด์ผ ํจ!
                poses, W, H, focal, self.z_near, self.z_far, c=c        # poses์ ํด๋นํ๋ ๋ถ๋ถ์ด extrinsic์ผ๋ก ์ ๋ฐ์๋๊ณ? ์์..!
            )  # (NV, H, W, 8)
            rgb_gt_all = images_0to1        # image๋ encoder์ ๋ค์ด๊ฐ๋ ๊ทธ๋๋ก ๋ฃ์ด์ฃผ๋ฉด ๋จ
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)

            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                pix_inds = torch.randint(0, NV * H * W, (args.ray_batch_size,))     

            # ์ฌ๊ธฐ์? Ray sampling์ ํด์ pix_inds๋ฅผ ์ป์ด๋ด๋?ค๊ณ? ํ๋๋ฐ, ์ฐ๋ฆฌ๋ Feature map์ ๋ณด๊ณ? ํ๊ธฐ ๋๋ฌธ์ 
            # pix_inds๋ก ์ธ๋ฑ์ฑํด์ค ๋์์ด ์์. ๊ทธ๋ฅ ์ด๊ฑฐ ์์ฒด๋ฅผ ์์?๋ ๋จ. 
            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                device=device       # ๊ทธ๋ฅ ์ด๋ค resolution์ ๋ํด ์์ฑํ๊ธฐ ๋๋ฌธ..
            )  # (ray_batch_size, 8)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(device)    #  single-view์ด๊ธฐ ๋๋ฌธ์ ์ด์ฐจํผ 0์ผ๋ก ์?๋ถ indexing ๋์ด์์ 
        src_images = util.batched_index_select_nd(      # NS: number of samples 
            all_images, image_ord
        )  # (SB, NS, 3, H, W) <- NV์์ NS๋ก ๋ฐ๋ -> index_select_nd์ ๋ฐ๋ผ์ ๊ฒฐ์?๋จ! <- ใใ ์ธ์? ์ด์ฐจํผ ํ obj ์์ 50๊ฐ ์์ผ๋๊น 
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)
        # 4๊ฐ์ batch, ๊ฐ batch์ NS๊ฐ ์ค ์ผ๋ถ๋ง ๊ณจ๋ผ์ poses๋ก ์ฒ๋ฆฌ <- ์คํค.. <- ์ด๊ฑฐ๋ ์ง์ง camera poses
        # ์ฅ ์ src poses๋ ํ๋๋ฐ์ ์๋๊ฑฐ์ง..? (4, 1, 4, 4)

        # ๊ฐ batch์์ ํ view๋ง ๊ณจ๋ผ์ ํ์ตํจ 

        all_bboxes = all_poses = all_images = None

        #######################################################################################
        ################### ์ฌ๊ธฐ๊น์ง ์ ์ง์คํด์ ์ฝ์ด๋ณด๊ธฐ! ray ๊ฐ์?ธ์ค๋ ๋ถ๋ถ!!! ########################
        #######################################################################################

        # remove 
        ############### NeRF encodingํ๋ ๋ถ๋ถ!!!!!!!!
        net.encode(
            src_images,      # batch, 1, 3, 128, 128
            src_poses,       # batch, 1, 4, 4       # input poses ๊ทธ๋๋ก ์ฌ์ฉ!
            all_focals.to(device=device),   # batch
            c=all_c.to(device=device) if all_c is not None else None,
        )
        #### ์ฌ๊ธฐ ์์์ poses, focals, c ์ ๊ด๋?จ๋ ์ฐ์ฐ in models.py:
        # rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        # trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)      # ์ด translation์ด ์์ธ๊ตฐ.. 
        # self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)
        # if len(focal.shape) == 0:
        #     # Scalar: fx = fy = value for all views
        #     focal = focal[None, None].repeat((1, 2))
        # elif len(focal.shape) == 1:
        #     # Vector f: fx = fy = f_i *for view i*
        #     # Length should match NS (or 1 for broadcast)
        #     focal = focal.unsqueeze(-1).repeat((1, 2))
        # else:
        #     focal = focal.clone()
        # self.focal = focal.float()     # ๊ฐ์ฅ ๋ง์ง๋ง์ ๊ฐ๋ค์ -1์ด ๊ณฑํด์ง 

        # self.focal[..., 1] *= -1.0
        # if c is None:
        #     # Default principal point is center of image
        #     c = (self.image_shape * 0.5).unsqueeze(0)
        # elif len(c.shape) == 0:
        #     # Scalar: cx = cy = value for all views
        #     c = c[None, None].repeat((1, 2))
        # elif len(c.shape) == 1:
        #     # Vector c: cx = cy = c_i *for view i*
        #     c = c.unsqueeze(-1).repeat((1, 2))
        # self.c = c


        # all_rays <- ์๋ transformation์ด ์?์ฉ๋ ray, -> ๊ทธ๋ฌ๋ฉด ์ด ์์์ depth์ ๋ฐ๋ผ์ samplingํ๋ฉด ๋์ง ์์? -> ใใ ์ฌ๊ธฐ ์์์ transformed ray์ฌ์ฉํ๋ฉด ๋จ! 
        # pixelnerf๋ ์?๋ง.. viewer space์์ ์ฒ๋ฆฌ.. 

        #######################################################################################
        ############### all_rays๊ฐ ๋ค์ด๊ฐ๋ค!!!! NeRF์!!!!!!! ####################################
        #######################################################################################

        #######################################################################################
        ############### ์ฌ๊ธฐ์๋ถํฐ ๋ฐ๊พธ๊ธฐ!!! ####################################
        #######################################################################################

        # all_rays: ((SB, ray_batch_size, 8)) <- NV images์์์ ์?์ฒด rays์ SB๋งํผ์!
        render_dict = DotMap(render_par(all_rays, want_weights=True,)) # models.py์ forward ํจ์๋ฅผ ๋ณผ ๊ฒ 
        # Q. render_par์ output์ dictionary์ธ๊ฐ? 
        # render par ํจ์ ๋ฐ์ผ๋ก ์?๋ถ giraffe renderer๋ก ๋ฐ๊พธ๊ธฐ 
        coarse = render_dict.coarse
        fine = render_dict.fine
        using_fine = len(fine) > 0

        loss_dict = {}

        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine

        loss = rgb_loss
        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item()

        return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.randint(0, NV - curr_nviews)
        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        renderer.eval()
        source_views = (
            images_0to1[views_src]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            net.encode(
                test_images.unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
            )
            test_rays = test_rays.reshape(1, H * W, -1)
            render_dict = DotMap(render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)

            if using_fine:
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        renderer.train()
        return vis, vals


trainer = PixelNeRFTrainer()
trainer.start()
