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
from model import NeuralRenderer
import torchvision.transforms as transforms
from dotmap import DotMap
from PIL import Image
import pdb 
from torchvision.utils import save_image, make_grid

warnings.filterwarnings(action='ignore')

def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=32, help="Object batch size ('SB')"
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
        "--recon",
        type=float,
        default=1.,
        help="Loss of reconstruction error",
    )

    parser.add_argument(
        "--swap",
        type=float,
        default=1.,
        help="Weights of swap loss error",
    )

    parser.add_argument(
        "--disc_lr",
        type=float,
        default=1.,
        help="Discriminator learning rate ratio",
    )

    parser.add_argument(
        "--cam",
        type=float,
        default=1.,
        help="Loss of camera prediction error",
    )

    parser.add_argument(
        "--cycle",
        type=float,
        default=1.,
        help="Loss of camera prediction error",
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

# make_model: model에 대한 option. 
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

# Ours로 변경 예정!     # from_config: 모델 세팅 알려줌    
renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp,).to(
    device=device       # NeRFRenderer -> renderer setting 
)

# Parallize         # net: pixelNeRF -> pixelNeRF를 
render_par = renderer.bind_parallel(net, args.gpu_id).eval()   # -> _RenderWrapper를 선언함 -> 얘의 forward 함수가 class NeRFRenderer 실행하는거!
# self까지도 속성받아버림!
# renderer.bind_parallel -> _RenderWrapper(net, self, simple_output=simple_output)

nviews = list(map(int, args.nviews.split()))        # 1. 


class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)   # superclass에서의 init
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=device), strict=False
                )

        self.z_near = dset.z_near       # 일단은 그냥 두기 
        self.z_far = dset.z_far
        self.focal = torch.tensor([2.187719,]) * 10
        self.c = torch.tensor([8.000000, 8.000000])
        self.use_bbox = args.no_bbox_step > 0
        self.recon_loss = torch.nn.MSELoss()
        self.cam_loss = torch.nn.MSELoss()
        self.cycle_loss = torch.nn.MSELoss()
        # self.optim.add_param_group({'params': self.neural_renderer.parameters()})

    def compute_bce(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, targets)
        return loss

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0, mode=None):
        #######################################################################################
        ################### 여기서부터 잘 집중해서 읽어보기! ray 가져오는 부분!!! ########################
        #######################################################################################
        if is_train :
            # SB: number of batches 
            if "images" not in data:
                return {}
            all_images = data["images"].to(device=device)  # (B, 3, H, W)   # images: 128, 128

            B, _, H, W = all_images.shape   
            all_poses = data["poses"].to(device=device)  # (B, 4, 4)
            all_focals = data["focal"]  # (B)      # 각 batch sample마다의 focal length가 존재함 
            all_c = data.get("c")  # (B, 2)       # 아이고.. 생각해보면 각 sample마다 f, c가 다를텐데.. <- 같다!

            image_ord = torch.randint(0, 1, (B, 1))   # ours -> 계속 nviews=1일 예정! 

            # 원래는 object for문에 껴있었는데 그냥 바로 배치 단위로 
            images_0to1 = all_images * 0.5 + 0.5
            rgb_gt_all = (
                images_0to1.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (B, H, W, 3)

            # feat-W, feat-H 받아야 함! 
            feat_H = 16 # <- args로 조정 가능하도록!
            feat_W = 16 # <- args로 조정 가능하도록!    # 아 오키 이거 volume renderer 세팅 따라가고, 다른 부분 있으면 giraffe 모듈 가져오기 
        
            shape, appearance = net.encode(     # <- encode부분은 동일하게 가져오고, forward하는 부분 좀더 신경써서 가져오기!
                all_images,
                focal=self.focal.to(device=device),
                c=self.c.to(device=device)
            )   # encoder 결과로 self.rotmat, self.shape, self.appearance 예측됨 
            rotmat = net.rotmat
            
            ################################################
            ########################### for generated views 
            cam_rays = util.gen_rays(       # 여기서의 W, H 사이즈는 output target feature image의 resolution이어야 함!
                rotmat, feat_W, feat_H, self.focal, self.z_near, self.z_far, self.c       # poses에 해당하는 부분이 extrinsic으로 잘 반영되고 있음..!
            )  # (NV, H, W, 8)
            rays = cam_rays.view(B, -1, cam_rays.shape[-1]).to(device=device)      # (batch * num_ray * num_points, 8)

            featmap = render_par(rays, want_weights=True, shape=shape, appearance=appearance, training=True) # <-outputs.toDict()의 결과 
            rgb_fake = net.neural_renderer(featmap)

            ################################################
            ########################### for swapped views 
            swap_rot = rotmat.flip(0)
            swap_cam_rays = util.gen_rays(       # 여기서의 W, H 사이즈는 output target feature image의 resolution이어야 함!
                swap_rot.detach(), feat_W, feat_H, self.focal, self.z_near, self.z_far, self.c       # poses에 해당하는 부분이 extrinsic으로 잘 반영되고 있음..!
            )  # (NV, H, W, 8)
            swap_rays = swap_cam_rays.view(B, -1, swap_cam_rays.shape[-1]).to(device=device)      # (batch * num_ray * num_points, 8)

            swap_featmap = render_par(swap_rays, want_weights=True, shape=shape, appearance=appearance, training=True) # <-outputs.toDict()의 결과 
            rgb_swap = net.neural_renderer(swap_featmap)


            # neural renderer를 저 render par 프로세스 안에 넣기!
            # discriminator가 swap을 지날 예정!
            d_fake = self.discriminator(rgb_swap)
            loss_dict = {}
            if mode == 'generator':
                ######## for cycle appearance 
                cycle_appearance = appearance.flip(0)
                cycle_featmap = render_par(rays, want_weights=True,  shape=shape, appearance=cycle_appearance, training=True,) # <-outputs.toDict()의 결과 
                rgb_cycle = net.neural_renderer(cycle_featmap)

                new_shape, new_appearance = net.encode(     # <- encode부분은 동일하게 가져오고, forward하는 부분 좀더 신경써서 가져오기!
                    rgb_cycle,
                    focal=self.focal.to(device=device),
                    c=self.c.to(device=device)
                )   # encoder 결과로 self.rotmat, self.shape, self.appearance 예측됨  

                new_rotmat = net.rotmat
                cycle_loss = self.cycle_loss(rotmat, new_rotmat) + \
                                self.cycle_loss(shape, new_shape) + \
                                    self.cycle_loss(appearance, new_appearance)          # 아니.. shape과 appearance를 disentangle 보장은 못하지만 camera는 확실히 잡을 수 있도록 돕는다.. 

                rgb_loss = self.recon_loss(rgb_fake, all_images) # 아 오키. sampling된 points 갯수가 128개인가보군 
                # net attribute으로 rotmat있는지 확인 + 예측했던 rotmat과 같은지 확인 
                cam_loss = self.cam_loss(net.rotmat, all_poses) # 아 오키. sampling된 points 갯수가 128개인가보군 
                gen_swap_loss = self.compute_bce(d_fake, 1)
                loss_gen = rgb_loss * args.recon + cam_loss * args.cam + gen_swap_loss * args.swap + cycle_loss * args.cycle
                return loss_gen, rgb_loss.item(), cam_loss.item(), gen_swap_loss.item(), cycle_loss.item()

            elif mode =='discriminator':
                d_real = self.discriminator(all_images)
                disc_swap_loss = self.compute_bce(d_fake, 0)
                disc_real_loss = self.compute_bce(d_real, 1)
                loss_disc = (disc_swap_loss * args.swap + disc_real_loss * args.swap) / 2
                return loss_disc, disc_swap_loss.item(), disc_real_loss.item()
            else:
                pass
        else:
            # SB: number of batches 
            if "images" not in data:
                return {}
            all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)
            all_poses = data["poses"].to(device=device)
            
            SB, NV, _, H, W = all_images.shape      # SB: number of obj, NV: number of view     -> 4, 50, 3, 128, 128
            all_focals = data["focal"]  # (SB)      # 각 batch sample마다의 focal length가 존재함 
            all_c = data.get("c")  # (SB)

            if self.use_bbox and global_step >= args.no_bbox_step:
                self.use_bbox = False
                print(">>> Stopped using bbox sampling @ iter", global_step)

            all_rgb_gt = []
            all_rays = []

            curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
            if curr_nviews == 1:       # (0,) 을 batch size만큼 만들어준다!
                image_ord = torch.randint(0, NV, (SB, 1))   # ours -> 계속 nviews=1일 예정! 
            else: # Pass
                image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)

            val_num = 5
            ##### object마다의 Process 
            ##### 여기서는 RGB sampling하는 과정은 아예 빼고, extrinsic을 통한 camera ray를 가져올 것 pix_inds는 필요없음 
            for obj_idx in range(SB):       # batch 안의 index마다 pose가 다르기 때문!      # SB: 4     # meshgrid만 괜찮다면 batch 연산으로 큼지막하게 한번 가도 괜찮을듯 
                # batch size는 작은 편, 각 sample에 대해서 처리함 
                # 이거 자체가 하나의 batch로서 기능함 
                # 너무 메모리가 커서 조금 샘플링 해야할 것 같기도.. 
                indices = torch.randint(0, NV, (val_num,))      # (전체 251개의 view 중 5개 뽑기!)

                # 딱 5개만 뽑아냄!
                images = all_images[obj_idx][indices]  # (NV, 3, H, W)       # (50, 3, 128, 128)
                poses = all_poses[obj_idx][indices]  # (NV, 4, 4)            # (50, 4, 4)        # <- multi-view rotation
                
                focal = self.focal
                c = self.c
                if curr_nviews > 1: # Pass
                    # Somewhat inefficient, don't know better way
                    image_ord[obj_idx] = torch.from_numpy(          # 배치 안의 한 샘플에 대해 5개 중에 하나 뽑기!
                        np.random.choice(indices, curr_nviews, replace=False)       # 0부터 4중에 하나 고르기!  <- 각 batch마다 어떤 view에서 source image를 가져올지 결정!
                    )       # ex. image_ord[0] = 2 -> 0번째 샘플의 obj index는 2
                images_0to1 = images * 0.5 + 0.5

                feat_H, feat_W = 16, 16
                # ㅇㅇ 다 넣고 봐도 될 듯. 어차피 feature field에 대해서 보는거라! 
                cam_rays = util.gen_rays(       # 여기서의 W, H 사이즈는 output target feature image의 resolution이어야 함!
                    poses, feat_W, feat_H, focal, self.z_near, self.z_far, c=c        # poses에 해당하는 부분이 extrinsic으로 잘 반영되고 있음..!
                )  # (NV, H, W, 8)
                rgb_gt_all = images_0to1        # image는 encoder에 들어가는 그대로 넣어주면 됨
                rgb_gt_all = (
                    rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
                )  # (NV * H * W, 3)

                # 여기선 Ray sampling을 해서 pix_inds를 얻어내려고 하는데, 우리는 Feature map을 보고 하기 때문에 
                # pix_inds로 인덱싱해줄 대상이 없음. 그냥 이거 자체를 없애도 됨. 
                rgb_gt = rgb_gt_all  # (ray_batch_size, 3)
                rays = cam_rays.view(-1, cam_rays.shape[-1]).to(
                    device=device       # 그냥 어떤 resolution에 대해 생성하기 때문..
                )  # (ray_batch_size, 8)

                all_rgb_gt.append(rgb_gt)
                all_rays.append(rays)


            all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, 5*ray_batch_size, 3)     # 5장의 이미지
            all_rays = torch.stack(all_rays)  # (SB, 5*ray_batch_size, 8)

            image_ord = image_ord.to(device)    #  single-view이기 때문에 어차피 0으로 전부 indexing 되어있음 
            src_images = util.batched_index_select_nd(      # NS: number of samples 
                all_images, image_ord # 모든 이미지에 대해 랜덤하게 뽑은 source image를 가져오게 됨 
            )  # (SB, NS, 3, H, W) <- NV에서 NS로 바뀜 -> index_select_nd에 따라서 결정됨! <- ㅇㅋ 인정 어차피 한 obj 안에 50개 있으니까 
            
            src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4) <- 이 src poses를 예측해보자!
            # 4개의 batch, 각 batch의 NS개 중 일부만 골라서 poses로 처리 <- 오키.. <- 이거는 진짜 camera poses
            all_poses = all_images = None

            # 각 batch마다 하나의 sample src image를 고름 
            #######################################################################################
            ################### 여기까지 잘 집중해서 읽어보기! ray 가져오는 부분!!! ########################
            #######################################################################################

            # remove 
            ############### NeRF encoding하는 부분!!!!!!!!
            shape, appearance = net.encode(
                src_images,      # batch, 1, 3, 128, 128
                focal=self.focal.to(device=device),   # batch
                c=self.c.to(device=device) if all_c is not None else None,
            )

            # 하나의 source image에 대해 5개의 feature output을 만듦 -> 전체 sample에 대해서!
            # all_rays: ((SB, ray_batch_size, 8)) <- NV images에서의 전체 rays에 SB만큼을!
            feat_out = render_par(all_rays, val_num=val_num, want_weights=True, shape=shape, appearance=appearance, training=False) # models.py의 forward 함수를 볼 것 
            # render par 함수 밑으로 전부 giraffe renderer로 바꾸기 
            test_out = net.neural_renderer(feat_out)          

            # test out 있는 여기에 self.neural_renderer 놓기 
            loss_dict = {}
            test_out_pred = test_out.reshape(SB, -1, 3)

            rgb_loss = self.recon_loss(test_out_pred, all_rgb_gt)
            cam_loss = self.cam_loss(net.rotmat, src_poses)

            loss_dict["rc"] = rgb_loss.item() * args.recon
            loss_dict["cam"] = cam_loss.item() * args.cam
            loss = rgb_loss
            loss_dict["t"] = loss.item()

            return loss_dict


    def train_step(self, data, global_step):
        # discriminator가 먼저 update 
        dict_ = {}
        disc_loss, disc_swap, disc_real = self.calc_losses(data, is_train=True, global_step=global_step, mode='discriminator')
        disc_loss.backward()
        self.optim_d.step()
        self.optim_d.zero_grad()        

        # generator 그다음에 update 
        gen_loss, gen_rgb, gen_cam, gen_swap, gen_cycle = self.calc_losses(data, is_train=True, global_step=global_step, mode='generator')
        gen_loss.backward()
        self.optim.step()
        self.optim.zero_grad() 

        dict_['disc_loss'] = round(disc_loss.item(), 3)
        dict_['disc_swap'] = round(disc_swap, 3)
        dict_['disc_real'] = round(disc_real, 3)

        dict_['gen_loss'] = round(gen_loss.item(), 3)
        dict_['gen_rgb'] = round(gen_rgb, 3)
        dict_['gen_cam'] = round(gen_cam, 3)
        dict_['gen_swap'] = round(gen_swap, 3)
        dict_['gen_caycle'] = round(gen_cycle, 3)

        return dict_

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses


    # 얘네는 기존의 data loader 그대로 활용하도록 고고 
    def vis_step(self, data, global_step, epoch, batch, idx=None):
        if "images" not in data:
            return {}

        if idx is None:
            batch_indices = np.random.randint(0, data["images"].shape[0], 2)   # 16 = batch -> (16, 251, 3, 128, 128)
        else:
            print(idx)
            batch_indices = idx
        
        gt_list, rgb_list, cat_list = [], [], []
        test_rays_dest_list = []
        test_rays_src_list = []
        shape_list, appearance_list = [], []

        for batch_idx in range(len(batch_indices)):
            # 16개 batch objects 중에 하나의 batch index를 
            images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
            poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)

            # for swapped appearance # appearance target 
            images_cycle = data["images"][batch_idx+1].to(device=device)  # (NV, 3, H, W)
            poses_cycle = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)

            focal = self.focal  # (1)
            c = self.c
            feat_H, feat_W = 16, 16
            NV, _, H, W = images.shape
            cam_rays = util.gen_rays(   # (251개의 poses에 대해서 만듦..)
                poses, feat_W, feat_H, focal, self.z_near, self.z_far, c=c      # (251, 16, 16, 8)
            )  # (NV, H, W, 8)
            images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)       # (251, 3, 128, 128)

            val_num = 3

            # curr_nviews를 4개로 잡아볼까
            curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]        # curr_nviews = 1
            views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))   # NV: 251 -> ex.views_src: 여러 이미지들 나오는디요 시발
            view_dests = np.random.randint(0, NV - curr_nviews, val_num)  # ex. 63
            for vs in range(curr_nviews):
                view_dests += view_dests >= views_src[vs]
            views_src = torch.from_numpy(views_src)

            # set renderer net to eval mode
            renderer.eval()     # <- encoder는 왜 eval() 아니지         # renderer의 parameter 찾고 여기에 2DCNN 포함되는지 확인!
            source_views = (
                images_0to1[views_src].repeat(val_num, 1, 1, 1)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
                .reshape(-1, H, W, 3)       # (3, 128, 128, 3)
            )

            gt = images_0to1[view_dests].permute(0, 2, 3, 1).cpu().numpy().reshape(val_num, H, W, 3)     # (128, 128, 3)
            with torch.no_grad():       # cam_rays: (NV, 16, 16, 8)
                test_rays_dest = cam_rays[view_dests]  # (3, H, W, 8)    # -> (val_num, 16, 16, 8)
                test_rays_src = cam_rays[views_src].repeat(val_num, 1, 1, 1)  # (H, W, 8)    # -> (16, 16, 8)

                test_images_src = images[views_src].repeat(val_num, 1, 1, 1)  # (NS, 3, H, W)     # -> (3, 128, 128)
                test_images_dest = images[view_dests] # -> # -> (val_num, 3, 128, 128)

                ##### for reconstructed views 
                shape, appearance = net.encode(
                    test_images_src,  # (val_num, 3, 128, 128) 
                    poses[views_src].repeat(val_num, 1, 1),  # (val_num, 4, 4)
                    focal=self.focal.to(device=device),   
                    c=self.c.to(device=device),
                )

                test_rays_dest = test_rays_dest.reshape(val_num, feat_H * feat_W, -1)   # -> (1, 16*16, 8)
                test_rays_src = test_rays_src.reshape(val_num, feat_H * feat_W, -1)   # -> (1, 16*16, 8)
                                    # test_rays: 1, 16x16, 8

                feat_test_dest = render_par(test_rays_dest, val_num = 1, want_weights=True, shape=shape, appearance=appearance)   # -> (1, 16*16, 8)
                out_dest = net.neural_renderer(feat_test_dest)

                feat_test_src = render_par(test_rays_src, val_num = 1, want_weights=True, shape=shape, appearance=appearance,)   # -> (1, 16*16, 8)
                out_src = net.neural_renderer(feat_test_src)

                rgb_psnr = out_dest.cpu().numpy().reshape(val_num, H, W, 3)

                cat = torch.cat((test_images_src[[0]], test_images_dest.reshape(-1, 3, H, W), out_src[[0]].clamp_(0., 1.), out_dest.reshape(-1, 3, H, W).clamp_(0., 1.)), dim=0)

                # for visualization 
                test_rays_dest_list.append(test_rays_dest)
                test_rays_src_list.append(test_rays_src)
                shape_list.append(shape)
                appearance_list.append(appearance)
                cat_list.append(cat)

                # for calculation
                gt_list.append(gt)
                rgb_list.append(rgb_psnr)


        import pdb 
        pdb.set_trace()        
        feat_test_dest = render_par(test_rays_dest_list[0], val_num = 1, want_weights=True, shape=shape_list[0], appearance=appearance_list[-1])   # -> (1, 16*16, 8)
        out_dest_1 = net.neural_renderer(feat_test_dest)

        feat_test_src = render_par(test_rays_src_list[0], val_num = 1, want_weights=True, shape=shape_list[0], appearance=appearance_list[-1])   # -> (1, 16*16, 8)
        out_src_1 = net.neural_renderer(feat_test_src)

        feat_test_dest = render_par(test_rays_dest_list[-1], val_num = 1, want_weights=True, shape=shape_list[-1], appearance=appearance_list[0])   # -> (1, 16*16, 8)
        out_dest_2 = net.neural_renderer(feat_test_dest)

        feat_test_src = render_par(test_rays_src_list[-1], val_num = 1, want_weights=True, shape=shape_list[-1], appearance=appearance_list[0])   # -> (1, 16*16, 8)
        out_src_2 = net.neural_renderer(feat_test_src)

        cat = torch.cat((out_src_1[[0]], out_dest_1.reshape(-1, 3, H, W), out_src_2[[0]].clamp_(0., 1.), out_dest_2.reshape(-1, 3, H, W).clamp_(0., 1.)), dim=0)
        cat_list.append(cat)
        
        cat_new = torch.cat(cat_list, dim=0)
        image_grid = make_grid(cat_new, nrow=len(cat))  # row에 들어갈 image 갯수
        save_image(image_grid, f'visuals/{args.name}/{epoch}_{batch}_out.jpg')


        # source views, gt, test_out
        # 다른 batch 이미지들을 가져와야 함 
        # cat = torch.cat((test_images_src[[0]], test_images_dest.reshape(-1, 3, H, W), out_src[[0]].clamp_(0., 1.), out_dest.reshape(-1, 3, H, W).clamp_(0., 1.)), dim=0)
        # cat_cycle = torch.cat((test_images_src[[0]], test_images_dest.reshape(-1, 3, H, W), out_src[[0]].clamp_(0., 1.), out_dest.reshape(-1, 3, H, W).clamp_(0., 1.)), dim=0)
        # image_grid = make_grid(cat, nrow=len(cat))  # row에 들어갈 image 갯수
        # save_image(image_grid, f'visuals/{args.name}/{epoch}_{batch}_out.jpg')

        # for vals calculation 
        psnr_total = 0
        for idx in range(len(gt_list)):
            psnr = util.psnr(rgb_psnr[idx], gt[idx])
            psnr_total += psnr
        
        psnr = psnr_total / 2
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        renderer.train()
        return None, vals


trainer = PixelNeRFTrainer()
trainer.start()
