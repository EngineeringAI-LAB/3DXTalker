from decalib.datasets import datasets 
from torch.utils.data import Dataset, DataLoader
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
from decalib.utils import util
from gdl.datasets.ImageTestDataset import TestData
from pathlib import Path
from gdl.models.DecaDecoder import Generator
from gdl.models.Renderer import SRenderY
import imageio

def deca_encode(input_img_path, deca_model):
    # testdata = datasets.TestData(img_path, iscrop=iscrop, face_detector=detector)
    test_dataset = TestData(input_img_path, face_detector="fan", max_detection=20)
    # images = testdata[0]['image'].to(deca_model.device)[None,...]
    # test_loader = DataLoader(testdata, batch_size=1, shuffle=False)
    for i in range(len(test_dataset)):
        batch = test_dataset[i]
        batch["image"] = batch["image"].to(deca_model.device)
        if len(batch["image"].shape) == 3:
            batch["image"] = batch["image"][None,...]
        with torch.no_grad():
            codedict = deca_model.encode(batch, training=False)
        shape_code = codedict['shapecode']
        exp_code = codedict['expcode']
        pose_code = codedict['posecode']
        detail_code = codedict['detailcode']
    return shape_code, exp_code, pose_code, detail_code


def deca_decode(deca_model, code_all_in_one, output_depth=False):
    shapecode = code_all_in_one[:, :100]
    expcode = code_all_in_one[:, 100:150]
    posecode = code_all_in_one[:, 150:156]
    detailcode = code_all_in_one[:, -128:]
    
    # shapecode_zero = torch.zeros_like(shapecode).to(shapecode.device)
    # posecode_zero = torch.zeros_like(posecode).to(posecode.device)
    verts, landmarks2d, landmarks3d, landmarks2d_mediapipe = deca_model.deca.flame(shape_params=shapecode, expression_params=expcode, pose_params=posecode)
    
    camcode = torch.from_numpy(np.array([[5.1146, 0.000324, 0.029371]])).float().to(verts.device)
    batch_size = verts.shape[0]
    trans_verts = util.batch_orth_proj(verts, camcode)
    trans_verts[:,:,1:] = -trans_verts[:,:,1:]

    opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            }
    
    albedo = torch.ones([batch_size, 3, deca_model.deca.config.uv_size, deca_model.deca.config.uv_size], device=verts.device) * 0.5

    codedict={'shapecode': shapecode,
              'expcode': expcode,
              'posecode': posecode,
            'detailcode': detailcode}
    detail_conditioning= ['jawpose', 'expression', 'detail']
    detail_conditioning_lst = deca_model._create_conditioning_lists(codedict, detail_conditioning)

    if isinstance(deca_model.deca.D_detail, Generator):
        uv_z = deca_model.deca.D_detail(torch.cat(detail_conditioning_lst, dim=1))

    
    render_model = SRenderY(image_size=512, obj_filename="/scratch3/wan451/3DTalk/The-Sound-of-Motion/data/head_template.obj",
                               uv_size=256).to(verts.device)
    render_model.eval()
    # render_model = deca_model.deca.render


    # ops = deca_model.deca.render(verts, trans_verts, albedo, h=512, w=512, background=None)
    with torch.no_grad():
        ops = render_model(verts, trans_verts, albedo)

    # output_depth = False
    # depth_video_output_path = '/scratch3/wan451/3DTalk/The-Sound-of-Motion/depth_output/EMOCA/my_depth_video.mp4'
    #if output_depth:
        # depth_video_output_path = '/scratch3/wan451/3DTalk/The-Sound-of-Motion/depth_output/my_depth_video_EMOCA.mp4'
    with torch.no_grad():
        depth_images = render_model.render_depth(trans_verts)[:,0,:,:]
        # shape_images = render_model.render_shape(verts, trans_verts)
            # normal_images = render_model.render_normal(trans_verts, normals=ops['normals'])
            # detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False)*alpha_images
            # shape_detail_images = render_model.render_shape(verts, trans_verts, detail_normal_images=detail_normal_images,)

    # def depthimgs_to_video(depth_array, depthvideo_output_path, fps=25):
    #     depth_gray = depth_array[:, 0, :, :]  # (N, H, W)
    #     depth_norm = (depth_gray - depth_gray.min()) / (depth_gray.max() - depth_gray.min() + 1e-8)
    #     depth_uint8 = (depth_norm * 255).astype(np.uint8)  # (N, H, W)
    #     depth_rgb_seq = np.stack([depth_uint8]*3, axis=-1)  # (N, H, W, 3)
    #     imageio.mimwrite(depthvideo_output_path, depth_rgb_seq, fps=fps, codec='libx264')

    # def rgbimgs_to_video(rgb_array, rgbvideo_output_path, fps=25):
    #     rgb_norm = (rgb_array - rgb_array.min()) / (rgb_array.max() - rgb_array.min() + 1e-8)
    #     rgb_uint8 = (rgb_norm * 255).astype(np.uint8)  # shape: (N, 3, H, W)
    #     rgb_uint8 = np.transpose(rgb_uint8, (0, 2, 3, 1))  # (N, H, W, 3)
    #     imageio.mimwrite(rgbvideo_output_path, rgb_uint8, fps=fps, codec='libx264')

        # depthimgs_to_video(depth_images.detach().cpu().numpy(), depthvideo_output_path=depth_video_output_path, fps=25)
        # rgbimgs_to_video(shape_images.detach().cpu().numpy(), rgbvideo_output_path='/scratch3/wan451/3DTalk/The-Sound-of-Motion/depth_output/my_rgb_video_EMOCA.mp4', fps=25)


    with torch.no_grad():
        uv_z = deca_model.deca.D_detail(torch.cat([posecode[:,3:], expcode, detailcode], dim=1))
        # uv_detail_normals = deca_model.deca.displacement2normal(uv_z, verts, ops['normals']) 

    # uv_texture = albedo

    B = verts.shape[0]
    # opdict['uv_texture'] = uv_texture                                          # [1, 3, 256, 256]
    opdict['normals'] = ops['normals']                                          # [1, 5023, 3]
    # opdict['uv_detail_normals'] = uv_detail_normals                            # [1, 3, 256, 256]
    opdict['displacement_map'] = uv_z + deca_model.deca.fixed_uv_dis[None,None,:,:] #[1, ,1, 256, 256]
    # texture = util.tensor2image(opdict['uv_texture'][0])                        # (256, 256, 3)
    # texture = texture[:,:,[2,1,0]]           
    # texture_tensor = torch.from_numpy(texture).unsqueeze(0).repeat(B, 1, 1, 1).to(verts.device)
    # faces_tensor = deca_model.render.faces.repeat(B, 1, 1).to(verts.device)
    dense_template_path = "/scratch3/wan451/3DTalk/The-Sound-of-Motion/assets/DECA/data/texture_data_256.npy"
    dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1').item()
    dense_vertices, dense_faces = util.upsample_mesh_batch(
                                            verts.detach(),
                                            opdict['normals'].detach(),
                                            opdict['displacement_map'].detach(), 
                                            dense_template)
    if output_depth:
        return verts, dense_vertices, depth_images
    else:
        return verts, dense_vertices,





    
