import os
import trimesh
import pickle
import smplx
import pyrender
import torch

from vis_tool import sp_animation

BODY_MODEL_DIR = "./smpl_model/"

VIS_MODEL = 'scenepic' # scenepic or pyrender

if __name__ == '__main__':

    # data_dir = '/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/preprocess/0000/'
    scene_dir = '/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main_old/ase/preprocess/0000/scene_obj/'
    # input_path = '/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/preprocess/0000/mesh/'

    data_dir = "/media/srtp/新加卷/SRTP_MotionGeneration/AMP_zjy/ase_cnn/poselib/cmu_smpl"
    input_path = "/media/srtp/新加卷/SRTP_MotionGeneration/AMP_zjy/ase_cnn/poselib/cmu_smpl/mesh/"
    model_path = "/media/srtp/新加卷/SRTP_MotionGeneration/AMP_zjy/ase_cnn/preprocess/smpl_model/"
    # Load human body meshes
    # 先建一个空的tensor，再cat，或者直接导入obj  
    # 或者先合并一个pkl，用np        
    scene_files = os.listdir(scene_dir)
    bodymesh = []
    files = os.listdir(input_path)
    files.sort()
    print(files)
    i = 0
    for mesh_file in files:
        mesh_file = os.path.join(input_path,mesh_file)
        mesh = trimesh.load(mesh_file)
        bodymesh.append(mesh)

      
    # vis

    if VIS_MODEL == 'pyrender':
        # Setup renderer
        scene = pyrender.Scene()
        spcaing = 2
        for i in range(bodymesh):

            # scene 1: source_object + ori_body
            s1_o = source_object.copy()
            s1_b = ori_bodymesh[i].copy()

            scene.add(pyrender.Mesh.from_trimesh(s1_o))
            scene.add(pyrender.Mesh.from_trimesh(s1_b))

            # scene 2: target_object + ori_body
            s2_o = target_object.copy().apply_translation([spcaing, 0, 0])
            s2_b = ori_bodymesh[i].copy().apply_translation([spcaing, 0, 0])

            scene.add(pyrender.Mesh.from_trimesh(s2_o))
            scene.add(pyrender.Mesh.from_trimesh(s2_b))

            # scene 3: target_object + new_body
            s3_o = target_object.copy().apply_translation([2*spcaing, 0, 0])
            s3_b = new_bodymesh[i].copy().apply_translation([2*spcaing, 0, 0])

            scene.add(pyrender.Mesh.from_trimesh(s3_o))
            scene.add(pyrender.Mesh.from_trimesh(s3_b))

            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
            scene.clear()

    elif VIS_MODEL == 'scenepic':       
        anim = sp_animation(framerate=30)
        i = 0
        for file in scene_files:
            file = os.path.join(scene_dir,file)
            scene_object = trimesh.load(file)
            matrix=[[1,0,0,0],[0,1,0,-0.10],[0,0,1,0],[0,0,0,1]]
            scene_object = scene_object.apply_transform(matrix)
            anim.add_static_mesh(scene_object, 'object')
        for i in range(len(files)):
            # # scene 1: source_object + ori_body
            s = bodymesh[i].copy()
            anim.add_frame([s], ['scene_body'])


        anim.save_animation(os.path.join(data_dir, 'mesh_demo.html'))
        print('scenepic animation saved!')
