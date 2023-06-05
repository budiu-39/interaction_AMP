
from isaacgym.torch_utils import *
import torch
import json
import numpy as np
import collections
import os

from scipy.spatial.transform import Rotation as R
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# 目的是为了产生带角速度的smpl参考运动
def from_pkl():
    motion_file='preprocess/0000/params'
    tpose_file='data/smpl_tpose.npy'
    local_rotation=[]
    root_translation=[]
    arrr={}
    arrt={}
    source_motion=collections.OrderedDict()
    files = os.listdir(motion_file)
    files.sort()
    print(files)
    for pkl in files:
        pkl = os.path.join(motion_file,pkl)
        motion = np.load(pkl,allow_pickle=True)
        rot=np.array(motion['person00']['pose'])
        rot=rot.reshape(-1,3)
        r = R.from_rotvec(rot)
        local_rotation.append(r.as_quat())

        root_translation.append(motion['person00']['transl'])

    arrr['arr'] = np.array(local_rotation)
    arrr['context'] = {'dtype':'float32'}
    source_motion['rotation'] = arrr
    arrt['arr'] = np.array(root_translation)
    arrt['context'] = {'dtype':'float32'}
    print(root_translation)
    source_motion['root_translation'] = arrt
    source_motion['is_local']=True,
    source_motion['fps']= 30
    source_motion['__name__']='SkeletonMotion'

    np.save('preprocess/0000/smpl_rot.npy',source_motion)

    source_tpose = SkeletonState.from_file(tpose_file)
    data=np.load("preprocess/0000/smpl_rot.npy", allow_pickle = True)
    d=data.item()
    rotation=torch.tensor(d['rotation']['arr'])
    root_translation=torch.tensor(d['root_translation']['arr'])
    source_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=source_tpose.skeleton_tree,r=rotation,t=root_translation, is_local=True)
    source_motion = SkeletonMotion.from_skeleton_state(source_state, fps=30)
    source_motion.to_file('poselib/data/0000_smpl_motion.npy')

def from_file():

    motion_file = 'poselib/data/smpl_model'
    tpose_file = 'poselib/data/tpose/smpl_tpose.npy'
    output_path = "poselib/data/cmu_smpl/mesh/"

    local_rot = []
    local_rotation = []
    source_motion = collections.OrderedDict()
    gender = "male"
    os.makedirs(output_path, exist_ok=True)
    data1 = np.load("poselib/data/cmu_smpl/poses.npy", allow_pickle=True)
    deta2 = np.load("poselib/data/cmu_smpl/betas.npy", allow_pickle=True)

    poses = data1[:,:72].reshape(-1,24,3)
    for i in poses:
        local_rot.append(R.from_rotvec(i).as_matrix())
    local_rot = np.array(local_rot)
    root_rot = np.linalg.inv(local_rot[0,0]).dot(R.from_rotvec([0,-np.pi/2,0]).as_matrix()) #.dot(R.from_euler('xyz',[0,90,0], degrees = True).as_matrix())
    local_rotat = []
    local_rot[:,0] = local_rot[:,0].dot(root_rot)
    for i in local_rot:
        local_rotation.append(R.from_matrix(i).as_quat())
        local_rotat.append(R.from_matrix(i).as_rotvec())

    betas = torch.tensor(deta2[:10], dtype=torch.float32).reshape(1, 10)
    trans = np.load("poselib/data/cmu_smpl/trans.npy", allow_pickle=True)
    trans[:, [1, 2]] = trans[:, [2, 1]]
    trans[:, 2] = -trans[:, 2]
    trans = torch.matmul(torch.tensor(trans),torch.tensor(root_rot))

    # trans[:] = trans[:].dot(R.from_euler('xyz',[90,0,0], degrees = True).as_matrix()).dot(R.from_euler('xyz',[0,-90,0], degrees = True).as_matrix())
    # trans[:] = trans[:]  #.dot(R.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix())
    translation = torch.tensor(trans)

    # global_orient = torch.tensor(data1[:, :3])
    # root_translation.append(motion['person00']['transl'])

    # arrr['arr'] = np.array(poses)
    # arrr['context'] = {'dtype': 'float32'}
    source_motion['rotation'] = torch.tensor(local_rotation)

    source_motion['root_translation'] =translation
    source_motion['is_local'] = True,
    source_motion['fps'] = 120
    source_motion['__name__'] = 'SkeletonMotion'

    source_tpose = SkeletonState.from_file(tpose_file)
    # data = np.load("/media/srtp/新加卷/SRTP_MotionGeneration/ASE-main/ase/preprocess/0000/smpl_rot.npy",
    #                allow_pickle=True)
    d = source_motion
    source_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=source_tpose.skeleton_tree, r=d['rotation'], t=d['root_translation'], is_local=True)
    source_motion = SkeletonMotion.from_skeleton_state(source_state, fps=120)
    source_motion.to_file('poselib/data/cmu_smpl/smpl_motion.npy')
    plot_skeleton_motion_interactive(source_motion)

if __name__ == '__main__':
    from_file()
    # from_pkl()

