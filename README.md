# interaction_AMP
这个项目是基于AMP代码 https://nv-tlabs.github.io/ASE/ 改进完成的，最终效果为实现了基于物理的人体-物体交互运动生成。  

项目的原理是Sigraph 2023的ynthesizing Physical Character-Scene Interactions中提出的基于对抗模仿强化学习的交互运动生成。  
（只复现了该文中的与椅子的交互任务，不过其它任务的原理是一致的。
  
  
最终实现的效果如下：  
当输入为单一风格的运动片段时，多个角色以相同的方式与物体发生交互，且运动轨迹不具有泛化性。  

![image](https://github.com/budiu-39/interaction_AMP/blob/main/single_reference.gif)   
  
    
当输入为多个风格的运动片段时，多个角色以不同的风格与物体发生交互，运动轨迹具有良好的泛化性。  
（在距离交互物体一定范围内，角色的初始位置与朝向被随机初始化后，最终都可以坐在椅子上。  

![image](https://github.com/budiu-39/interaction_AMP/blob/main/multi_reference.gif)

除了人体-物体交互运动，俺还在人体-地形交互方面（与交互运动的原理相似）做过一些尝试
  
固定轨迹上楼梯:

![image](https://github.com/budiu-39/interaction_AMP/blob/main/terrain_1.gif)


地形中自主探索：  

![image](https://github.com/budiu-39/interaction_AMP/blob/main/terrain_2.gif)

部分代码见 https://github.com/budiu-39/AMP_terrain  
不过，基于地形的三维人体自主运动生成目前应该还没有一个泛化性良好的解决方案。  


 
