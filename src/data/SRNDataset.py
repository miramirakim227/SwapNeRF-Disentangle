import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import get_image_to_tensor_balanced, get_mask_to_tensor

##################################################################
################### MAKING #######################################
##################################################################

# car: (2459, 50)

class SRNDataset(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self, path, stage="train", image_size=(128, 128), world_scale=1.0,
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        self.stage = stage 
        self.base_path = path + "_" + stage
        self.dataset_name = os.path.basename(path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size        # (128, 128)
        self.world_scale = world_scale      # 1. 
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        if is_chair:
            self.z_near = 1.25
            self.z_far = 2.75
        else:
            self.z_near = 0.8
            self.z_far = 1.8
        self.lindisp = False

        # chair, car 둘다 이 intrinsic 가짐. 대신 chairs에 어떤 추가적인 preprocess 붙는지 좀더 봐보기 
        # 이거 원래 코드부분 보면서 generalizable하게 바꾸기 
        # self.focal 어떻게 조정해야 하나..?
        # self.focal, self.cx, self.cy = 131.250000, 8.000000, 8.000000      # <- 아니면 원래 표현방법대로 다시 돌아가기 
        self.focal, self.cx, self.cy = 2.187719, 8.000000, 8.000000      # <- 아니면 원래 표현방법대로 다시 돌아가기 

        self.rgb = sorted(
            glob.glob(os.path.join(self.base_path, "*", "rgb", "*"))
        )
        self.poses = sorted(
            glob.glob(os.path.join(self.base_path, "*", "pose", "*"))
        )


    def __len__(self):
        if self.stage == 'train':
            return len(self.rgb)
        else:
            return len(self.intrins)

    # 이 getitem하는 부분도 해당하는 부분들 이미지 등에서 한장씩만 가져올 수 있으면 좋을텐데.. 
    # focal length, center points 등등도.. <- 평균내서 돌려버릴까.. <- 오우쓑 이미 다 같음!
    def __getitem__(self, index):       # <- 아 미친.. 한 batch당 50개가 한번에 뽑힘.. 
        if self.stage == 'train':
            rgb_path = self.rgb[index]
            pose_path = self.poses[index]

            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)      # input 
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)     # input 

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            pose = pose @ self._coord_trans             # input

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)  # input 

            if img_tensor.shape[-2:] != self.image_size:
                scale = self.image_size[0] / img_tensor.shape[0]
                self.focal *= scale
                self.cx *= scale
                self.cy *= scale 

                img_tensor = F.interpolate(img_tensor, size=self.image_size, mode="area")
                mask_tensor = F.interpolate(mask_tensor, size=self.image_size, mode="area")

            if self.world_scale != 1.0:
                self.focal *= self.world_scale
                pose[:3, 3] *= self.world_scale
            focal = torch.tensor(self.focal, dtype=torch.float32)

            result = {
                # "path": dir_path,       # dir_path: 50개짜리 이미지 묶음 있는 디렉토리 
                "img_id": index,
                "focal": self.focal,
                "c": torch.tensor([self.cx, self.cy], dtype=torch.float32),
                "images": img_tensor,
                "masks": mask_tensor,
                "poses": pose,
            }

            return result
        else:
            intrin_path = self.intrins[index]
            dir_path = os.path.dirname(intrin_path)
            rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
            pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

            assert len(rgb_paths) == len(pose_paths)

            with open(intrin_path, "r") as intrinfile:
                lines = intrinfile.readlines()
                focal, cx, cy, _ = map(float, lines[0].split())
                height, width = map(int, lines[-1].split())

            all_imgs = []
            all_poses = []
            all_masks = []
            all_bboxes = []
            for rgb_path, pose_path in zip(rgb_paths, pose_paths):
                img = imageio.imread(rgb_path)[..., :3]
                img_tensor = self.image_to_tensor(img)
                mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
                mask_tensor = self.mask_to_tensor(mask)

                pose = torch.from_numpy(
                    np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
                )
                pose = pose @ self._coord_trans

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError(
                        "ERROR: Bad image at", rgb_path, "please investigate!"
                    )
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

                all_imgs.append(img_tensor)
                all_masks.append(mask_tensor)
                all_poses.append(pose)
                all_bboxes.append(bbox)

            # 50개를 쌓아버림

            all_imgs = torch.stack(all_imgs)
            all_poses = torch.stack(all_poses)
            all_masks = torch.stack(all_masks)
            all_bboxes = torch.stack(all_bboxes)


            if all_imgs.shape[-2:] != self.image_size:
                scale = self.image_size[0] / all_imgs.shape[-2]
                focal *= scale
                cx *= scale
                cy *= scale
                all_bboxes *= scale

                all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

            if self.world_scale != 1.0:
                focal *= self.world_scale
                all_poses[:, :3, 3] *= self.world_scale
            focal = torch.tensor(focal, dtype=torch.float32)


            # 음... 계속 새로운 instance가 들어올 수 있도록 encoder 부분 바꿔주기 

            result = {
                "path": dir_path,
                "img_id": index,
                "focal": focal,
                "c": torch.tensor([cx, cy], dtype=torch.float32),
                "images": all_imgs,
                "masks": all_masks,
                "bbox": all_bboxes,
                "poses": all_poses,
            }
            return result
    