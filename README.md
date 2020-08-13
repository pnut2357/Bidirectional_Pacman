# CSE 571 Spring 2020 Final Project

## Team name and members
### Quad Js
- Jaehyuk Choi
- Jinyung Hong
- Eunsom Jeon
- Yongbaek Cho


## Description of Topic
### Bi-directional search
We apply the variants of MM including MMε and Factional MM (fMM(p)) to solve the POSITION SEARCH problems in Pacman domain. 

## Contributions of each team member
- Jaehyuk Choi : Analyzed conventional heuristics on MM's behavior and developed customized one for better performance of MM. Contributed to algorithm design, introduction, and technical approach so that how the combined MM impletments the variants of MM.   
- Jinyung Hong : I developed MM and MMε. I participated in writing the introduction and technical approach in our report.
- Eunsom Jeon : I developed fMM and heuristics. I participated in writing the technical approach heuristics, the results and discussion, and conclusion in our report.
- Yongbaek Cho : I experimented with algorithms and analyzed it. I created tables and graphs participated in writing the Results and discussion in our report. 


## Dependencies
- Python 2

## How to run

```python
python pacman.py -l [Maze] -p SearchAgent -a fn=[search_algorithm],heuristic=[Heuristic],useEpsilon=[useEpsilon],useFractional=[useFractional],p=[pValue]
```

- Maze (Required) : {Tiny, Small, Medium, Big}
- search_algorithm (Required) : {bfs, dfs, astar, ucs, bis}
    - bfs = breadthFirstSearch
    - dfs = depthFirstSearch
    - astar = aStarSearch
    - ucs = uniformCostSearch
    - bis = bidirectionalSearch
                     
- Heuristic (Optional) : 
    - default: nullHeuristic, 
    - {manhattanHeuristic, octileDistance, cosineDistance, digonalDistance, customizedHeuristic, mazeHeuristic}
    
- useEpsilon (Optional) : 
    - default : False
    - {True, False}
    
- useFractional (Optional) :
    - default : False
    - {True, False}
    
- pValue<sup>1</sup> (Optional) :
    - default : 0.5 
    - [0, 1]

<sup>1</sup> The meaning of pValue:
- pValue = 0.5 -> MM
- pValue = 0 -> reverse A*
- pValue = 1 -> Forward A*

Example)

- MM0 (MM Without heuristic)
```python
python pacman.py -l tinyMaze -p SearchAgent -a fn=bis
```

- MM With manhattan heuristic
```python
python pacman.py -l tinyMaze -p SearchAgent -a fn=bis,heuristic=manhattanHeuristic
```

- MMε without heuristic
```python
python pacman.py -l smallMaze -p SearchAgent -a fn=bis,useEpsilon=True
```

- MMε with manhattan heuristic
```python
python pacman.py -l smallMaze -p SearchAgent -a fn=bis,heuristic=manhattanHeuristic,useEpsilon=True
```

- Fractional MM without heuristic
```python
python pacman.py -l smallMaze -p SearchAgent -a fn=bis,useFractional=True,p=0.8
```

- Fractional MM with manhattan heuristic
```python
python pacman.py -l tinyMaze -p SearchAgent -a fn=bis,heuristic=manhattanHeuristic,useFractional=True,p=0.3
```

## Comparison table of Heuristics
![heuristic formula](https://user-images.githubusercontent.com/43649503/80334192-52960980-8805-11ea-9d13-3d26e6ffac5b.png)



<img width="551" alt="table 22" src="https://user-images.githubusercontent.com/43649503/80453318-d969e580-88dc-11ea-8da5-5bb4e87485ae.png">

<img width="785" alt="table2" src="https://user-images.githubusercontent.com/43649503/80334072-f59a5380-8804-11ea-8acb-560321025916.png">




## Graph
![unInformed](https://user-images.githubusercontent.com/43649503/80542826-96525580-8962-11ea-9a0e-168adb0fc119.png)
![222](https://user-images.githubusercontent.com/43649503/79622864-eeb96580-80cd-11ea-890b-0c8d0f24c5c5.png)
![3333](https://user-images.githubusercontent.com/43649503/79622865-efea9280-80cd-11ea-894e-c146dcdc0239.png)
![graph1](https://user-images.githubusercontent.com/43649503/80450979-9a856100-88d7-11ea-9cc6-0b6e218cc826.png)
![ranking](https://user-images.githubusercontent.com/43649503/80451496-de2c9a80-88d8-11ea-9eb5-54c63fd11ea8.png)

## References
- [AIMA examples](https://github.com/aimacode/aima-python)
- Video [AAAI Presentation - MM: Bidirectional Search That Is Guaranteed to Meet in the Middle](https://youtu.be/VCSFyj9Yy0c)
- Paper [Bidirectional Search That Is Guaranteed to Meet in the Middle](https://webdocs.cs.ualberta.ca/%7Eholte/Publications/MM-AAAI2016.pdf)
- Paper [Extended Abstract: An Improved Priority Function for Bidirectional Heuristic Search](https://www.aaai.org/ocs/index.php/SOCS/SOCS16/paper/viewFile/13959/13257)
- Paper [A Brief History and Recent Achievements in Bidirectional Search](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17232)
