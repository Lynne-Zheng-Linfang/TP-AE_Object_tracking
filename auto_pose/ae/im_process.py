import os
import cv2
import numpy as np
import time
import glob
import traceback
import tensorflow as tf
import random

def get_intrinsic_params(K):
    K = np.array(K).flatten()
    param = {}
    param['fx'] = K[0]
    param['fy'] = K[4]
    param['cx'] = K[2]
    param['cy'] = K[5]
    return param

def get_enlarged_bbox(cam_params, t, diameter, enlarge_scale=1.3):
    """
    calculate the 2D bbox of a cube which parallal to the image plane and centered at t, with a side lengh of diameter*enlarge_scale
    return: topleft_x, topleft_y, width, hight
    """
    t = t.flatten()
    radius = diameter*enlarge_scale/2
    scale = 1 - radius/t[2]
    front_face_centre = t*scale 
    topleft_x = int((front_face_centre[0] - radius)*cam_params['fx']/front_face_centre[2] + cam_params['cx'])
    topleft_y = int((front_face_centre[1] - radius)*cam_params['fy']/front_face_centre[2] + cam_params['cy'])
    bottomright_x = int((front_face_centre[0] + radius)*cam_params['fx']/front_face_centre[2] + cam_params['cx']) 
    bottomright_y = int((front_face_centre[1] + radius)*cam_params['fy']/front_face_centre[2] + cam_params['cy']) 
    # t = t.flatten()
    # radius = diameter*enlarge_scale/2
    # topleft_x = int((t[0] - radius)*cam_params['fx']/(t[2] - radius) + cam_params['cx'])
    # topleft_y = int((t[1] - radius)*cam_params['fy']/(t[2] - radius) + cam_params['cy'])
    # bottomright_x = int((t[0] + radius)*cam_params['fx']/(t[2] - radius) + cam_params['cx']) 
    # bottomright_y = int((t[1] + radius)*cam_params['fy']/(t[2] - radius) + cam_params['cy']) 
    w = bottomright_x - topleft_x
    h = bottomright_y - topleft_y
    return topleft_x, topleft_y, w, h

def crop_image(image, cam_params, diameter, t, bbox_enlarge_level, depth=False):
    im_size = image.shape
    tl_x, tl_y, w, h = get_enlarged_bbox(cam_params, t, diameter,  enlarge_scale = bbox_enlarge_level)

    #Depth Image
    if depth:
        cropped_image = np.full((h,w,)+ (image.shape[2],), 10000)
    else:
        cropped_image = np.zeros((h,w,)+ (image.shape[2],))

    x_begin = 0
    y_begin = 0
    x_end = w
    y_end = h

    if tl_x < 0:
        x_begin = -tl_x
    
    if tl_y < 0:
        y_begin = -tl_y
    
    if (tl_x + w) > im_size[1]:
        x_end = im_size[1] - tl_x
    if (tl_y + h) > im_size[0]:
        y_end = im_size[0] - tl_y

    l_x = int(max(tl_x, 0))
    l_y = int(max(tl_y, 0))
    r_x = int(min(tl_x+w, im_size[1]))
    r_y = int(min(tl_y+h, im_size[0]))
    x_begin = int(x_begin)
    y_begin = int(y_begin)
    x_end = int(x_end)
    y_end = int(y_end)
    try:
        cropped_image[y_begin:y_end, x_begin:x_end] = image[l_y: r_y, l_x: r_x]
    except:
        print('Error: im_process.py: crop image out of range')
        print('lx', l_x)
        print('ly', l_y)
        print('tl_x', tl_x)
        print('r_x', r_x)
        print('r_y', r_y)
        print('im_size', im_size)
        print('tl_x, tl_y, w, h:', tl_x, tl_y, w, h)
        print('x_begin, x_end, y_begin, y_end:', x_begin, x_end, y_begin, y_end)
        exit()

    return cropped_image

def crop_real_image(rgb, depth, K, diameter, t, bbox_enlarge_level):
    image_h, image_w = depth.shape[:2]
    cam_params = get_intrinsic_params(K)
    tl_x, tl_y, crop_w, crop_h = get_enlarged_bbox(cam_params, t, diameter,  enlarge_scale = bbox_enlarge_level)

    rgb_crop = np.zeros((crop_h, crop_w, 3), dtype= np.uint8)
    depth_crop = np.zeros((crop_h, crop_w), dtype= np.float32)

    x_begin = 0
    y_begin = 0
    x_end = crop_w 
    y_end = crop_h

    if tl_x < 0:
        x_begin = -tl_x
    
    if tl_y < 0:
        y_begin = -tl_y
    
    if (tl_x + crop_w) > image_w:
        x_end = image_w - tl_x
    if (tl_y + crop_h) > image_h:
        y_end = image_h - tl_y

    l_x = int(max(tl_x, 0))
    l_y = int(max(tl_y, 0))
    r_x = int(min(tl_x+crop_w, image_w))
    r_y = int(min(tl_y+crop_h, image_h))
    x_begin = int(x_begin)
    y_begin = int(y_begin)
    x_end = int(x_end)
    y_end = int(y_end)
    try:
        rgb_crop[y_begin:y_end, x_begin:x_end] = rgb[l_y: r_y, l_x: r_x]
        depth_crop[y_begin:y_end, x_begin:x_end] = depth[l_y: r_y, l_x: r_x]
    except:
        print('Error: im_process.py: crop image out of range')
        print('lx', l_x)
        print('ly', l_y)
        print('tl_x', tl_x)
        print('r_x', r_x)
        print('r_y', r_y)
        print('im_size', image_h, image_w)
        print('tl_x, tl_y, crop_w, crop_h:', tl_x, tl_y, crop_w, crop_h)
        print('x_begin, x_end, y_begin, y_end:', x_begin, x_end, y_begin, y_end)
        exit()
    K_shape = K.shape
    K = K.flatten()
    CX_INDEX = 2
    CY_INDEX = 5
    K[CX_INDEX] -= tl_x
    K[CY_INDEX] -= tl_y
    K = K.reshape(K_shape)

    return (rgb_crop, depth_crop, K) 
    
def resize_image(image, new_size, interpolation=cv2.INTER_NEAREST):
    resized_im = cv2.resize(image, new_size, interpolation = interpolation)
    return resized_im

def batch_images_resize(images, new_size, interpolation=cv2.INTER_NEAREST):
    resized_images = np.empty((len(images),)+new_size+(images.shape[-1],), dtype=images.dtype) 
    for i in range(len(images)):
        resized_images[i] = cv2.resize(images[i], new_size, interpolation = interpolation)
    return resized_images

def depth_to_3D_coords(image, cam_params):
    h, w = image.shape

    x_map = np.tile(np.array(range(w)), (h,1))
    y_map = np.tile(np.array(range(h)).reshape(h,1), (1,w))

    real_x = (x_map - cam_params['cx'])*image/cam_params['fx']
    real_y = (y_map - cam_params['cy'])*image/cam_params['fy']
    new_image = np.stack((real_x, real_y, image), axis=2)
    return new_image

def depth_point_to_3D_coords( depth_point, K):
    pixel_x, pixel_y, depth_z = depth_point
    cam_params = get_intrinsic_params(K)
    real_x = (pixel_x - cam_params['cx'])*depth_z/cam_params['fx']
    real_y = (pixel_y- cam_params['cy'])*depth_z/cam_params['fy']
    return (real_x, real_y, depth_z)

def depth_translation(image, t):
    return image-np.array(t).reshape(1,3)

def scale_to_sphere(image, diameter, enlarge_scale):
    radius = diameter/2
    new_image = image.copy()
    result = np.linalg.norm(image, axis=2)
    mask = result > enlarge_scale*radius
    new_image[mask] = np.array([0,0,0])
    new_image = new_image/(enlarge_scale*diameter)
    return new_image, mask

def scale_to_unit_cube(image, diameter, enlarge_scale):
    new_image = image/(enlarge_scale*diameter)
    visable_mask = (np.absolute(new_image) <= 0.5).all(axis=2)
    new_image += 0.5
    new_image *= np.expand_dims(visable_mask, axis = 2)
    return new_image, visable_mask

def rotate_depth(image, R, mask):
    R = np.array(R).reshape(3,3)
    h, w, c = image.shape

    new_image = image.reshape(-1,c)
    new_image = (np.matmul(R.transpose(), new_image.transpose()).transpose()).reshape(h,w,c)
    return new_image

def depth_image_preprocess(depth_image, R, t, K, img_size, target_obj_diameter, bbox_enlarge_level ):
    camera_params = get_intrinsic_params(K)
    depth_im_3D = depth_to_3D_coords(depth_image, camera_params)
    depth_im_3D = depth_translation(depth_im_3D, t)
    cropped_depth = crop_image(depth_im_3D, camera_params, target_obj_diameter, t, bbox_enlarge_level, depth = True)
    resized_depth = resize_image(cropped_depth, img_size)
    rotated_depth = rotate_depth(resized_depth, R, None)
    cube_depth, visable_mask = scale_to_unit_cube(rotated_depth, target_obj_diameter, bbox_enlarge_level)
    final_depth = cube_depth*255
    return final_depth.astype(np.uint8), visable_mask

# TODO: depth should be transfered from sphere to unit cube
def irrelavent_depth_image_preprocess(depth_image, R, crop_center_x, crop_center_y, t, K, img_size, target_obj_diameter, bbox_enlarge_level, crop_size):
    half_crop_size = int(crop_size/2)
    camera_params = get_intrinsic_params(K)
    depth_im_3D = depth_to_3D_coords(depth_image, camera_params)
    depth_im_3D = depth_translation(depth_im_3D, t)

    depth_im_3D = depth_im_3D[crop_center_y-half_crop_size:crop_center_y+half_crop_size, crop_center_x-half_crop_size:crop_center_x+half_crop_size]

    resized_depth = resize_image(depth_im_3D, img_size)
    normalized_depth, mask = scale_to_sphere(resized_depth, target_obj_diameter, bbox_enlarge_level)
    final_depth = rotate_depth(normalized_depth, R, mask)
    final_depth = (final_depth + 0.5)*175.2
    final_depth[mask] = np.array([0,0,0])
    return final_depth.astype(np.uint8)

def rgb_image_preprocess(rgb, t, K, img_size, target_obj_diameter, bbox_enlarge_level):
    camera_params = get_intrinsic_params(K)
    cropped_rgb = crop_image(rgb, camera_params, target_obj_diameter, t, bbox_enlarge_level)
    resized_rgb = resize_image(cropped_rgb, img_size)
    return resized_rgb

def mask_image_preprocess(mask, t, K, img_size, target_obj_diameter, bbox_enlarge_level):
    camera_params = get_intrinsic_params(K)
    cropped_mask = crop_image(mask, camera_params, target_obj_diameter, t, bbox_enlarge_level)
    resized_mask = resize_image(cropped_mask, img_size)
    return resized_mask

def checkDepthImage(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)*0.1
    mask = img  > 0
    img[mask] = 255
    img[mask ==False] = 0

    cv2.imwrite('/home/linfang/depth_check.png', img.astype(np.uint8))

def getBackgroundImagesList(path, img_type):
    depth_list = []
    rgb_list = []
    gt_list = []
    info_list = []
    img_dirs = os.listdir(path)
    img_dirs.sort()
    for dir_name in img_dirs:
        dir_path = os.path.join(path, dir_name) 
        depth_list.append(getImageFilePathList(os.path.join(dir_path, 'depth'), img_type)) 
        rgb_list.append(getImageFilePathList(os.path.join(dir_path, 'rgb'), img_type))
        info_list.append(inout.load_info(os.path.join(dir_path, 'info.yml')))
        gt_list.append(inout.load_gt(os.path.join(dir_path, 'info.yml')))

def generateMasks(dir_path, img_type):
    return 0

def getImageFilePathList(dir_path, img_type):
    glob_path = dir_path+'/*'+img_type
    img_file_paths = glob.glob(glob_path)
    img_file_paths.sort()
    return img_file_paths


def moveObjImage(rgb, depth, ori_t, new_t, ori_K, new_K):
    ori_t = ori_t.flatten()
    new_t = new_t.flatten()
    ori_cam_params = get_intrinsic_params(ori_K.flatten())
    new_cam_params = get_intrinsic_params(new_K.flatten())
    center_alignd_rgb = alignImageCenter(rgb, ori_cam_params, new_cam_params) 
    center_alignd_depth = alignImageCenter(depth, ori_cam_params, new_cam_params) 
    depth_scale = new_t[2]/ori_t[2] 
    x_scale = ori_t[2]/new_t[2]*new_cam_params['fx']/ori_cam_params['fx']
    y_scale = ori_t[2]/new_t[2]*new_cam_params['fy']/ori_cam_params['fy']
    new_rgb = cv2.resize(rgb, (0,0), fx=x_scale, fy=y_scale, interpolation=cv2.INTER_NEAREST)
    new_depth = cv2.resize(depth, (0,0), fx=x_scale, fy=y_scale, interpolation=cv2.INTER_NEAREST)

def alignImageCenter(self, ori_img, ori_cam_params, new_cam_params):
    translation_matrix = np.array([
        [1, 0, new_cam_params['cx'] - ori_cam_params['cx']],
        [0, 1, new_cam_params['cy'] - ori_cam_params['cy']]
    ])
    rows, cols = ori_img.shape
    new_img = cv2.warpAffine(ori_img, translation_matrix, (cols, rows))
    return new_img
# getBackgroundImagesList('/home/linfang/Documents/Dataset/linemod_ori/test_kinect', '.png')

def rotate_image(ori_rgb, ori_depth, ori_R, ori_t, K, angle):
    cam_params = get_intrinsic_params(K)
    h, w = ori_rgb.shape[:2]

    R_3d = np.eye(3)
    R_2d = cv2.getRotationMatrix2D((cam_params['cx'], cam_params['cy']), angle, 1.0)
    rotated_rgb = cv2.warpAffine(ori_rgb, R_2d, (w,h))
    rotated_depth = cv2.warpAffine(ori_depth, R_2d, (w,h))

    R_3d[0:2, 0:2] = R_2d[0:2, 0:2]
    R_out = np.matmul(R_3d, ori_R)
    t_out = np.matmul(R_3d, ori_t.reshape((3,1)))
    return rotated_rgb, rotated_depth, R_out, t_out

def check_rotation_result(renderer, ori_rgb, rotated_rgb, rotated_R, rotated_t, K, target_obj_id):
    rgb_check, depth_check = renderer.render(
        obj_id=target_obj_id,
        W=ori_rgb.shape[1],
        H=ori_rgb.shape[0],
        K=K.copy(),
        R=rotated_R,
        t=rotated_t,
        near=10,
        far=10000,
        random_light=False
    )
    overlapping = cv2.addWeighted(rgb_check, 0.5, rotated_rgb, 0.8, 0)
    cv2.imshow('ori_rgb', ori_rgb)
    cv2.imshow('rotated_rgb', rotated_rgb)
    cv2.imshow('rgb_check', rgb_check)
    cv2.imshow('overlap',overlapping ) 
    cv2.waitKey(20000)

def depth_to_3D_coords_broadcast(images, Ks):
    '''
    Transfer depth images to 3D point clouds.
    The input depth images should not have the channel dimension.
    '''
    h, w = images.shape[1:]
    Ks = Ks.reshape((-1,9))
    cx = Ks[:, 2].reshape((-1,1,1))
    cy = Ks[:, 5].reshape((-1,1,1))
    fx = Ks[:, 0].reshape((-1,1,1))
    fy = Ks[:, 4].reshape((-1,1,1))

    x_map = np.expand_dims(np.tile(np.array(range(w)), (h,1)), axis=0)
    y_map = np.expand_dims(np.tile(np.array(range(h)).reshape(h,1), (1,w)), axis=0)

    real_xs = (x_map - cx)*images/fx
    real_ys = (y_map - cy)*images/fy
    new_images = np.stack((real_xs, real_ys, images), axis=3)
    return new_images

def get_enlarged_bbox_broadcast(Ks, ts, diameter, enlarge_scale=1.3):
    """
    calculate the 2D bbox of a cube which parallal to the image plane and centered at t, with a side lengh of diameter*enlarge_scale
    return: topleft_x, topleft_y, width, hight
    """
    radius = diameter*enlarge_scale/2
    ts = ts.reshape(-1,3,1)
    scale = 1 - radius/ts[:, 2, :]
    ts = ts*scale.reshape(-1,1,1)

    x = ts[:,0,:]
    y = ts[:,1,:]
    z = ts[:,2,:]

    Ks = Ks.reshape((-1,9,1))
    cx = Ks[:,2,:]
    cy = Ks[:,5,:]
    fx = Ks[:,0,:]
    fy = Ks[:,4,:]

    topleft_x = ((x - radius)*fx/(z) + cx).astype(int)
    topleft_y = ((y - radius)*fy/(z) + cy).astype(int)
    bottomright_x = ((x + radius)*fx/(z) + cx).astype(int)
    bottomright_y = ((y + radius)*fx/(z) + cy).astype(int)
    # ts = ts.reshape(-1,3,1)
    # x = ts[:,0,:]
    # y = ts[:,1,:]
    # z = ts[:,2,:]

    # Ks = Ks.reshape((-1,9,1))
    # cx = Ks[:,2,:]
    # cy = Ks[:,5,:]
    # fx = Ks[:,0,:]
    # fy = Ks[:,4,:]

    # radius = diameter*enlarge_scale/2

    # topleft_x = ((x - radius)*fx/(z - radius) + cx).astype(int)
    # topleft_y = ((y - radius)*fy/(z- radius) + cy).astype(int)
    # bottomright_x = ((x + radius)*fx/(z - radius) + cx).astype(int)
    # bottomright_y = ((y + radius)*fx/(z - radius) + cy).astype(int)
    w = bottomright_x - topleft_x
    h = bottomright_y - topleft_y
    if (h<0).any():
        print(w, topleft_x, topleft_y, bottomright_x, bottomright_y)
        print('ts:', ts.reshape(-1,3))
        print('Ks:', Ks.reshape(-1,9))
        exit()
    return (topleft_x, topleft_y, w, h)

def crop_and_resize_broadcast(images, Ks, diameter, ts, bbox_enlarge_level, patch_size, depth=False, crop_hs=None, crop_ws=None):
    im_h, im_w = images.shape[1:3]
    im_c = 3
    image_patchs = np.empty((len(images), ) + patch_size + (im_c,))

    if depth:
        x_map = np.tile(np.array(range(im_w)), (im_h,1))
        y_map = np.tile(np.array(range(im_h)).reshape(im_h,1), (1,im_w)) 
        Ks = Ks.reshape((-1,9))
        cx = Ks[:, 2]
        cy = Ks[:, 5]
        fx = Ks[:, 0]
        fy = Ks[:, 4]

    if (crop_hs is not None) and (crop_ws is not None):
        for i in range(len(images)):
            h, w = int(crop_hs[i]), int(crop_ws[i])
            if depth:
                new_image = np.stack((x_map, y_map, images[i]), axis=2)
            else:
                new_image = images[i]
            cropped_image = new_image[:h, :w]
            if depth:
                cropped_image[:,:,0] = (cropped_image[:,:,0] - cx[i])*cropped_image[:,:,2]/fx[i]
                cropped_image[:,:,1] = (cropped_image[:,:,1] - cy[i])*cropped_image[:,:,2]/fy[i]
            image_patchs[i] = resize_image(cropped_image, patch_size)
    else:
   
        tl_x, tl_y, w, h = get_enlarged_bbox_broadcast(Ks, ts, diameter, enlarge_scale= bbox_enlarge_level)

        crop_x_begin = np.maximum(tl_x, 0)
        crop_y_begin = np.maximum(tl_y, 0)
        crop_x_end = np.minimum(tl_x + w, im_w)
        crop_y_end = np.minimum(tl_y + h, im_h)

        fill_x_begin = np.maximum(-tl_x, 0)
        fill_y_begin = np.maximum(-tl_y, 0)
        fill_x_end = crop_x_end - crop_x_begin + fill_x_begin
        fill_y_end = crop_y_end - crop_y_begin + fill_y_begin

        for i in range(len(images)):
            try:
                cropped_image = np.zeros((int(h[i]),int(w[i]),)+ (im_c,))
            except:
                print(i, h[i], w[i], im_c)
                exit()
            l_x = int(crop_x_begin[i])
            l_y = int(crop_y_begin[i])
            r_x = int(crop_x_end[i])
            r_y = int(crop_y_end[i])

            x_begin = int(fill_x_begin[i])
            y_begin = int(fill_y_begin[i])
            x_end = int(fill_x_end[i])
            y_end = int(fill_y_end[i])
  
            if depth:
                new_image = np.stack((x_map, y_map, images[i]), axis=2)
            else:
                new_image = images[i]
            cropped_image[y_begin:y_end, x_begin:x_end] = new_image[l_y: r_y, l_x: r_x]
            if depth:
                cropped_image[:,:,0] = (cropped_image[:,:,0] - cx[i])*cropped_image[:,:,2]/fx[i]
                cropped_image[:,:,1] = (cropped_image[:,:,1] - cy[i])*cropped_image[:,:,2]/fy[i]
            image_patchs[i] = resize_image(cropped_image, patch_size)
    return image_patchs 

def crop_and_resize(rgb, depth, K, diameter, t, bbox_enlarge_level, patch_size):
    im_h, im_w = rgb.shape[:2]
    im_c = 6
    x_map = np.tile(np.array(range(im_w)), (im_h,1))
    y_map = np.tile(np.array(range(im_h)).reshape(im_h,1), (1,im_w)) 
    depth = np.stack((x_map, y_map, depth), axis=2)
    image = np.concatenate((rgb , depth), axis=2)

    K = get_intrinsic_params(K)
    tl_x, tl_y, w, h = get_enlarged_bbox(K, t, diameter, enlarge_scale = bbox_enlarge_level)

    l_x = int(max(tl_x, 0))
    l_y = int(max(tl_y, 0))
    r_x = int(min(tl_x + w, im_w))
    r_y = int(min(tl_y + h, im_h))

    x_begin = int(max(-tl_x, 0))
    y_begin = int(max(-tl_y, 0))
    x_end = int(r_x - l_x + x_begin)
    y_end = int(r_y - l_y + y_begin)

    cropped_image = np.zeros((int(h),int(w),)+ (im_c,))

    try:
        cropped_image[y_begin:y_end, x_begin:x_end] = image[l_y: r_y, l_x: r_x]
    except:
        print('tl_x, tl_y, w, h', tl_x, tl_y, w, h)
        print('l_x, l_y, r_x, r_y', l_x, l_y, r_x, r_y)
        print('x_begin, y_begin, x_end, y_end', x_begin, y_begin, x_end, y_end)
        exit()
    cropped_image[:,:,3] = (cropped_image[:,:,3] - K['cx'])*cropped_image[:,:,-1]/K['fx']
    cropped_image[:,:,4] = (cropped_image[:,:,4] - K['cy'])*cropped_image[:,:,-1]/K['fy']
    image_patch = resize_image(cropped_image, patch_size)
    return image_patch

def rotate_depth_broadcast(images, Rs):
    Rs = np.array(Rs).reshape(-1,3,3)
    h, w, c = images.shape[1:]
    new_images = images.reshape(-1, h*w, c)
    new_images = (np.matmul(Rs.transpose(0,2,1), new_images.transpose(0,2,1)).transpose(0,2,1)).reshape(-1,h,w,c)
    return new_images

def scale_to_unit_cube_broadcast(images, diameter, enlarge_scale):
    new_images = images/(enlarge_scale*diameter)
    invisable_masks = (np.absolute(new_images) > 0.5).any(axis=3)
    new_images += 0.5
    new_images[invisable_masks] = np.array([0,0,0]) 
    visable_mask = np.invert(invisable_masks)
    return new_images, visable_mask

def depth_image_preprocess_broadcast(depth_images, Rs, ts, Ks, patch_size, target_obj_diameter, bbox_enlarge_level, crop_hs=None, crop_ws=None, return_no_rot_patch = False):
    image_patchs = crop_and_resize_broadcast(depth_images, Ks, target_obj_diameter, ts, bbox_enlarge_level, patch_size, depth=True, crop_hs=crop_hs, crop_ws=crop_ws)
    visable_masks = np.expand_dims(image_patchs[:,:,:,-1]>0, axis=3)
    if return_no_rot_patch:
        image_patchs, no_rot_patchs = rigid_body_transformation_broadcast(image_patchs, Rs, ts, return_no_rot_patch=return_no_rot_patch)
        no_rot_patchs, _= scale_to_unit_cube_broadcast_new(no_rot_patchs, target_obj_diameter, visable_masks,bbox_enlarge_level)
        no_rot_patchs *= 255
    else:
        image_patchs = rigid_body_transformation_broadcast(image_patchs, Rs, ts)

    image_patchs, visable_masks= scale_to_unit_cube_broadcast_new(image_patchs, target_obj_diameter, visable_masks,bbox_enlarge_level)
    final_depths = image_patchs*255
    if return_no_rot_patch:
        return final_depths.astype(np.uint8), visable_masks.squeeze(axis=3), no_rot_patchs.astype(np.uint8)
    else:
        return final_depths.astype(np.uint8), visable_masks.squeeze(axis=3)

def depth_image_preprocess_broadcast_and_aug(depth_images, rgb_patchs,\
    aug_depth_patchs, Rs, ts, Ks, patch_size, target_obj_diameter,\
    bbox_enlarge_level, gt_mask = None, seg_rate = None):
    image_patchs = crop_and_resize_broadcast(depth_images, Ks, target_obj_diameter, ts, bbox_enlarge_level, patch_size, depth=True) 
    depth_visable_masks = np.expand_dims(image_patchs[:,:,:,-1]>0, axis=3)
    rgb_visable_masks = np.any(rgb_patchs.astype(bool), axis=3, keepdims=True)
    if gt_mask is not None:
        if seg_rate is not None:
            seg = (random.random() < seg_rate)
            if seg:
                depth_visable_masks *= np.expand_dims(gt_mask, axis = 3)
                rgb_visable_masks *= np.expand_dims(gt_mask, axis = 3)
        else:
            depth_visable_masks *= np.expand_dims(gt_mask, axis = 3)
            rgb_visable_masks *= np.expand_dims(gt_mask, axis = 3)

    obj_empty_area_masks = rgb_visable_masks*np.invert(depth_visable_masks)

    image_patchs -= ts.reshape((-1,1,1,3))
    image_patchs_no_rot, aug_vis_mask = merge_depth_with_aug_patchs_broadcast(aug_depth_patchs, image_patchs, depth_visable_masks, obj_empty_area_masks)
    image_patchs = rotate_depth_broadcast(image_patchs_no_rot, Rs)

    visable_masks = np.invert(obj_empty_area_masks)
    image_patchs, _= scale_to_unit_cube_broadcast_new(image_patchs, target_obj_diameter, visable_masks,bbox_enlarge_level)
    image_patchs[0] *= np.expand_dims(gt_mask[0], axis=2)
    final_depths = image_patchs*255
    image_patchs_no_rot, _= scale_to_unit_cube_broadcast_new(image_patchs_no_rot, target_obj_diameter, visable_masks,bbox_enlarge_level)
    image_patchs_no_rot[0] *= np.expand_dims(gt_mask[0], axis=2)
    final_depths_no_rot = image_patchs_no_rot*255
    return final_depths.astype(np.uint8), aug_vis_mask, final_depths_no_rot.astype(np.uint8)

def depth_patch_preprocess_broadcast_and_aug(depth_patchs, rgb_patchs,aug_depth_patchs, Rs, ts, patch_size, target_obj_diameter, bbox_enlarge_level ):
    depth_visable_masks = np.expand_dims(depth_patchs[:,:,:,-1]>0, axis=3)
    rgb_visable_masks = np.any(rgb_patchs.astype(bool), axis=3, keepdims=True)
    obj_empty_area_masks = rgb_visable_masks*np.invert(depth_visable_masks)

    depth_patchs -= ts.reshape((-1,1,1,3))
    image_patchs, aug_vis_mask = merge_depth_with_aug_patchs_broadcast(aug_depth_patchs, depth_patchs, depth_visable_masks, obj_empty_area_masks)
    image_patchs = rotate_depth_broadcast(image_patchs, Rs)

    visable_masks = np.invert(obj_empty_area_masks)
    image_patchs, visable_masks= scale_to_unit_cube_broadcast_new(image_patchs, target_obj_diameter, visable_masks,bbox_enlarge_level)
    final_depths = image_patchs*255
    return final_depths.astype(np.uint8), aug_vis_mask

def merge_depth_with_aug_patchs_broadcast(bg_depth, fg_depth, fg_mask, obj_empty_area_masks):
    fg_z_depth = np.expand_dims(fg_depth[:,:,:,-1], axis=3) 
    bg_z_depth = np.expand_dims(bg_depth[:,:,:,-1], axis=3) 
    occ_area_mask = fg_mask*(fg_z_depth-bg_z_depth <= 0)
    vis_mask = np.logical_or(occ_area_mask, obj_empty_area_masks)
    out_depth = np.where(vis_mask, fg_depth, bg_depth)
    vis_mask[0] = np.logical_or(fg_mask[0], obj_empty_area_masks[0])
    out_depth[0] = fg_depth[0]*vis_mask[0]
    return out_depth, vis_mask

def depth_patch_preprocess_broadcast(depth_images, Rs, ts,  target_obj_diameter, bbox_enlarge_level):
    visable_masks = np.expand_dims(depth_images[:,:,:,-1]>0, axis=3)
    image_patchs = rigid_body_transformation_broadcast(depth_images, Rs, ts)
    image_patchs, visable_masks= scale_to_unit_cube_broadcast_new(image_patchs, target_obj_diameter, visable_masks,bbox_enlarge_level)
    final_depths = image_patchs*255
    return final_depths.astype(np.uint8), visable_masks.squeeze(axis=3)
    
def rigid_body_transformation_broadcast(images, Rs, ts, return_no_rot_patch = False):
    images -=  ts.reshape((-1,1,1,3))
    new_images = rotate_depth_broadcast(images, Rs)
    if return_no_rot_patch:
        return new_images, images
    else:
        return new_images

def rgb_image_preprocess_broadcast(rgbs, ts, Ks, patch_size, target_obj_diameter, bbox_enlarge_level, crop_hs=None, crop_ws=None):
    image_patchs = crop_and_resize_broadcast(rgbs, Ks, target_obj_diameter, ts, bbox_enlarge_level, patch_size, crop_hs=crop_hs, crop_ws=crop_ws)
    return image_patchs.astype(np.uint8)

def scale_to_unit_cube_broadcast_new(images, diameter, vis_mask, enlarge_scale):
    new_images = images/(enlarge_scale*diameter)
    visable_masks = (np.absolute(new_images) <= 0.5).all(axis=3, keepdims=True) * vis_mask
    new_images += 0.5
    new_images *= visable_masks 
    return new_images, visable_masks

@tf.function
def image_preprocess_tf(rgb, depth, t, R, K, target_obj_diameter, bbox_enlarge_level, patch_size):
    K = get_intrinsic_params_tf(K) 
    depth_3D = depth_to_3D_coords_tf(depth, K)
    rgb_crop, depth_crop, depth_vis_mask_crop, crop_h, crop_w = crop_with_pad(rgb, depth_3D, K, t, target_obj_diameter, bbox_enlarge_level)
    depth_crop= depth_image_preprocess_tf(depth_crop, depth_vis_mask_crop, R, t, K, target_obj_diameter, bbox_enlarge_level, crop_h, crop_w)*255
    image_patch = tf.cast(tf.concat((rgb_crop, depth_crop), axis = -1), dtype= tf.uint8)
    image_patch = tf.image.resize(image_patch, patch_size, method='nearest')
    image_patch = tf.expand_dims(image_patch, axis=0) 
    return image_patch
    
@tf.function
def crop_with_pad(rgb, depth, K, t, target_obj_diameter, bbox_enlarge_level):
    rgb = tf.cast(rgb, dtype = tf.float32)
    topleft_x, topleft_y, crop_w, crop_h = get_enlarged_bbox_tf(K, t, target_obj_diameter, enlarge_scale=bbox_enlarge_level)

    topleft_w_pad = tf.math.maximum(0, -topleft_x)
    topleft_h_pad = tf.math.maximum(0, -topleft_y)
    img_h, img_w = rgb.get_shape()[0:2]
    target_w = tf.math.maximum(topleft_x + img_w + topleft_w_pad, img_w + topleft_w_pad)
    target_h = tf.math.maximum(topleft_y + img_h + topleft_h_pad , img_h+topleft_h_pad)

    depth_mask = tf.expand_dims(get_visable_mask(depth[:,:,-1]), axis= 2)
    stacked_image = tf.concat((rgb, depth, depth_mask), axis = -1)

    img_resized=tf.image.pad_to_bounding_box(
        stacked_image, topleft_h_pad, topleft_w_pad, target_h, target_w
    )
    crop_topleft_x = topleft_x + topleft_w_pad
    crop_topleft_y = topleft_y + topleft_h_pad
    cropped_image = tf.image.crop_to_bounding_box(img_resized, crop_topleft_y, crop_topleft_x, crop_h, crop_w)

    rgb = cropped_image[:,:,0:3]
    depth = cropped_image[:,:, 3:6]
    mask = cropped_image[:,:,6:]

    return (rgb, depth, mask, crop_h, crop_w) 

@tf.function
def depth_image_preprocess_tf(depth, mask, R, t, K, diameter, enlarge_scale, h, w):
    depth = depth - tf.reshape(t, (1, 3))
    depth = rotate_depth_tf(depth, R, h, w)
    depth, vis_mask = scale_to_unit_cube_tf(depth, mask, diameter, enlarge_scale)
    return depth

@tf.function
def get_visable_mask(image):
    dims = len(image.get_shape())
    if dims == 3:
        mask = tf.reduce_any(image>0, axis=-1, keepdims=True)
        c = image.get_shape()[-1]
        mask = tf.tile(mask, [1,1,c])
    else:
        mask = image > 0
    return tf.cast(mask, dtype = tf.float32)

@tf.function
def get_intrinsic_params_tf(K):
    K = tf.reshape(K, (-1,))
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]
    return (fx, fy, cx, cy) 

@tf.function
def depth_to_3D_coords_tf(image, cam_params):
    image = tf.cast(image, dtype=tf.float32)
    fx, fy, cx, cy = cam_params

    cols, rows = image.get_shape()[:2]
    img_mask_x = tf.range(0, cols, dtype=tf.float32)
    img_mask_y = tf.range(0, rows, dtype=tf.float32)
    y_map, x_map = tf.meshgrid(img_mask_x, img_mask_y, indexing='ij')

    real_x = (x_map - cx)*image/fx
    real_y = (y_map - cy)*image/fy
    new_image = tf.stack((real_x, real_y, image), axis=2)
    return new_image

@tf.function
def get_enlarged_bbox_tf(cam_params, t, diameter, enlarge_scale=1.3):
    """
    calculate the 2D bbox of a cube which parallal to the image plane and centered at t, with a side lengh of diameter*enlarge_scale
    return: topleft_x, topleft_y, width, hight
    """
    t = tf.reshape(t,(-1,))
    fx, fy, cx, cy = cam_params
    radius = diameter*enlarge_scale/2
    scale = 1 - radius/t[2]
    front_face_centre = t*scale 

    # print(t)
    topleft_x = tf.cast((t[0] - radius)*fx/(t[2]) + cx, dtype= tf.int32)
    topleft_y = tf.cast((t[1] - radius)*fy/(t[2]) + cy, dtype= tf.int32)
    bottomright_x = tf.cast((t[0] + radius)*fx/(t[2]) + cx, dtype= tf.int32)
    bottomright_y = tf.cast((t[1] + radius)*fy/(t[2]) + cy, dtype= tf.int32)
    # w = tf.math.maximum(bottomright_x - topleft_x, bottomright_y - topleft_y)
    # h = w
    # t = tf.reshape(t,(-1,))
    # fx, fy, cx, cy = cam_params
    # radius = diameter*enlarge_scale/2
    # # print(t)
    # topleft_x = tf.cast((t[0] - radius)*fx/(t[2] - radius) + cx, dtype= tf.int32)
    # topleft_y = tf.cast((t[1] - radius)*fy/(t[2] - radius) + cy, dtype= tf.int32)
    # bottomright_x = tf.cast((t[0] + radius)*fx/(t[2] - radius) + cx, dtype= tf.int32)
    # bottomright_y = tf.cast((t[1] + radius)*fy/(t[2] - radius) + cy, dtype= tf.int32)
    w = bottomright_x - topleft_x
    h = bottomright_y - topleft_y

    return (topleft_x, topleft_y, w, h)

@tf.function
def rotate_depth_tf(image, R, h, w):
    R = tf.cast(tf.reshape(R,(3,3)), dtype= tf.float32)
    c = image.get_shape()[2]

    new_image = tf.reshape(image,(-1,c))
    new_image = tf.reshape(tf.transpose(tf.matmul(tf.transpose(R), tf.transpose(new_image))),(h,w,c))
    return new_image

@tf.function
def scale_to_unit_cube_tf(image, depth_vis_mask, diameter, enlarge_scale):
    new_image = image/(enlarge_scale*diameter)
    visable_mask = tf.cast(tf.reduce_all(tf.math.abs(new_image) <= 0.5, axis=2, keepdims=True), dtype=tf.float32)

    dims = len(depth_vis_mask.get_shape())
    if dims < 3:
        detph_vis_mask = tf.expand_dims(depth_vis_mask, axis=2)
    detph_vis_mask = tf.cast(depth_vis_mask, dtype = tf.float32)

    visable_mask = visable_mask* depth_vis_mask 
    new_image += 0.5
    new_image = new_image*visable_mask
    return new_image, visable_mask

def get_crop_coords(topleft_x, topleft_y, crop_w, crop_h, image_w, image_h):
    x_begin = np.maximum(topleft_x, 0)
    y_begin = np.maximum(topleft_y, 0)
    x_end = np.minimum(x_begin + crop_w, image_w)
    y_end = np.minimum(y_begin + crop_h, image_h)
    return (x_begin, y_begin, x_end, y_end)

@tf.function
def depth_to_3D_coords_broadcast_tf(images, Ks):
    if len(images.get_shape()) < 3:
        images = tf.expand_dims(images, axis = 0)
        Ks = tf.expand_dims(Ks, axis = 0)
    
    rows, cols = images.get_shape()[1:3]
    img_mask_x = tf.range(0, rows, dtype=tf.float32)
    img_mask_y = tf.range(0, cols, dtype=tf.float32)
    y_map, x_map = tf.meshgrid(img_mask_x, img_mask_y, indexing='ij')

    Ks = tf.reshape(Ks, (-1,9,1,1))
    cx = Ks[:, 2]
    cy = Ks[:, 5]
    fx = Ks[:, 0]
    fy = Ks[:, 4]

    real_x = (x_map - cx)*images/fx
    real_y = (y_map - cy)*images/fy
    image_3D = tf.stack((real_x, real_y, images), axis=3)
    return image_3D 

@tf.function
def to_3D_patchs_broadcast_tf(rgbs, depths, Ks, target_shape):
    depths = tf.cast(depths, dtype= tf.float32)
    if len(rgbs.get_shape()) == 3:
        rgbs = tf.expand_dims(rgbs, axis = 0)
        depths = tf.expand_dims(depths, axis = 0)
    depths = depth_to_3D_coords_broadcast_tf(depths, Ks)
    depths = tf.image.resize(depths, target_shape, method='nearest')
    rgbs = tf.image.resize(rgbs, target_shape, method='nearest')
    return rgbs, depths

def to_3D_patch(rgb, depth, K, target_shape):
    cam_params = get_intrinsic_params(K)
    depth = depth_to_3D_coords(depth, cam_params) 
    depth = resize_image(depth, target_shape) 
    rgb =resize_image(rgb, target_shape)
    return rgb, depth

@tf.function
def rotation_depth_process_tf(depths, Ks, Rs, ts, target_obj_diameter, bbox_enlarge_level, target_shape):
    depths = depth_to_3D_coords_broadcast_tf(depths, Ks)
    depths = tf.image.resize(depths, target_shape, method='nearest')
    depths, _ = depth_patch_preprocess_broadcast_tf(depths, Rs, ts, target_obj_diameter, bbox_enlarge_level)
    return depths

def rotation_depth_process(depths, Ks, Rs, ts, target_obj_diameter, bbox_enlarge_level, target_shape):
    depths = depth_to_3D_coords_broadcast(depths, Ks)
    depths = batch_images_resize(depths, target_shape)
    depths, _ = depth_patch_preprocess_broadcast(depths, Rs, ts, target_obj_diameter, bbox_enlarge_level)
    return depths

@tf.function
def depth_patch_preprocess_broadcast_tf(depth_images, Rs, ts,  target_obj_diameter, bbox_enlarge_level):
    visable_masks = tf.expand_dims(depth_images[:,:,:,-1]>0, axis=3)
    image_patchs = rigid_body_transformation_broadcast_tf(depth_images, Rs, ts)
    image_patchs, visable_masks= scale_to_unit_cube_broadcast_new_tf(image_patchs, target_obj_diameter, visable_masks,bbox_enlarge_level)
    final_depths = image_patchs*255
    return tf.cast(final_depths, dtype=tf.uint8), tf.squeeze(visable_masks, axis=3)

@tf.function
def rotate_depth_broadcast_tf(images, Rs):
    Rs = tf.reshape(Rs, (-1,3,3))
    h, w, c = images.get_shape()[1:]
    new_images = tf.reshape(images,(-1, h*w, c))
    new_images = tf.transpose(tf.matmul(tf.transpose(Rs, (0,2,1)), tf.transpose(new_images,(0,2,1))),(0,2,1))
    new_images = tf.reshape(new_images, (-1,h,w,c))
    return new_images

@tf.function
def rigid_body_transformation_broadcast_tf(images, Rs, ts):
    images -=  tf.reshape(ts,(-1,1,1,3))
    new_images = rotate_depth_broadcast_tf(images, Rs)
    return new_images

@tf.function
def scale_to_unit_cube_broadcast_new_tf(images, diameter, vis_mask, enlarge_scale):
    new_images = images/(enlarge_scale*diameter)
    visable_masks = tf.reduce_all((tf.math.abs(new_images) <= 0.5),axis=3, keepdims=True)
    visable_masks = tf.math.logical_and(visable_masks, vis_mask)
    new_images += 0.5
    new_images *= tf.cast(visable_masks, dtype=tf.float32)
    return new_images, visable_masks

def keep_t_within_image(t, K, im_W, im_H):
    new_t = t.copy()
    K = get_intrinsic_params(K)
    t = t.flatten()
    x = int(t[0]*K['fx']/t[2] + K['cx'])
    if x < 0:
        new_t[0] = (-K['cx'])*t[2]/K['fx']
    if x > im_W:
        new_t[0] = (im_W-K['cx'])*t[2]/K['fx']

    y = int(t[1]*K['fy']/t[2] + K['cy'])
    if y < 0:
        new_t[1] = (-K['cy'])*t[2]/K['fy']
    if y > im_H:
        new_t[1] = (im_H - K['cy'])*t[2]/K['fy']
    return new_t 

def keep_ts_within_image(ts, K, im_W, im_H):
    #TODO: need to be modified to broadcast
    new_ts = np.empty_like(ts)
    for i in range(len(ts)):
        new_ts[i] = keep_t_within_image(ts[i], K, im_W, im_H)
    return new_ts
