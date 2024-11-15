# interaction_AMP
### codebase
This project is an improvement based on the AMP code from here https://nv-tlabs.github.io/ASE/, resulting in a system that generates **physically-based human-object interaction motion**.  

### Project Principles
The methodology is based on the adversarial imitation learning approach for interaction motion generation proposed in the SIGGRAPH 2023 paper, Synthesizing Physical Character-Scene Interactions. \\
(Note: only the chair interaction task from the paper was reproduced, though the principles are consistent across other tasks.)  
  
  
### Implementation Results 
1. **When the input is a single-style motion clip**  
Multiple characters interact with the object in the same style, with limited generalizability in their motion paths.  

![image](https://github.com/budiu-39/interaction_AMP/blob/main/single_reference.gif)   


2. **When the input includes multiple styles of motion clips**  
Multiple characters interact with the object in various styles, demonstrating good generalizability in their motion paths.
Generalizability here means that when the character's initial position and orientation are 

![image](https://github.com/budiu-39/interaction_AMP/blob/main/multi_reference.gif)

### Other Attempts
In addition to human-object interaction motion, I also explored** human-terrain interaction** (similar principles to interaction motion) with some experiments.
  
1. **Ascending stairs along a fixed path:**

![image](https://github.com/budiu-39/interaction_AMP/blob/main/terrain_1.gif)


2. **Autonomous exploration on terrain:**

![image](https://github.com/budiu-39/interaction_AMP/blob/main/terrain_2.gif)

Code for human-terrain interaction can be found here: https://github.com/budiu-39/AMP_terrain  

### Summary
Reproducing human-object interaction motion generation based on the paper’s theoretical foundation is relatively straightforward, though transferring this approach to a complex terrain setting has been less effective.  \\
Generating generalized 3D human motion for terrain-based interactions currently lacks a robust solution, making it a promising direction for further exploration!  


 
