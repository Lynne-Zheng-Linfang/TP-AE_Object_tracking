# -*- coding: utf-8 -*-

import numpy as np
import glob

# from utils import lazy_property

class Renderer(object):

    def __init__(self, model_path, store_path):
        self.model_path = model_path
        self.dataset_path = store_path

    @property
    def renderer(self):
        from meshrenderer import meshrenderer_phong

        model_paths = glob.glob(self.model_path)
        if model_paths == []:
            print('No model file found in model path! Please check with your model path.')
            exit()
        model_paths.sort()
        renderer = meshrenderer_phong.Renderer(
            model_paths,
            1,
            self.dataset_path,
            1
        )
        return renderer

    @property
    def clip_near(self):
        return 10
    @property
    def clip_far(self):
        return 10000 

    def render_image(self, obj_id, H,W, R, t, K, random_light = False):
        K = K.reshape((3,3)) 
        R = np.array(R).reshape((3,3)) 
        t = np.array(t).flatten()
        rgb, depth = self.renderer.render(
            obj_id = obj_id,
            W = W,
            H = H,
            R = R,
            t = t,
            K = K,
            near = self.clip_near,
            far = self.clip_far,
            random_light = random_light 
        )
        return rgb, depth
