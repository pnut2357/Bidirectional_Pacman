# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    from util import Stack
    parent_map = {}
    actions = []
    initial_state = problem.getStartState()
    if problem.isGoalState(initial_state):
        return actions
    else:
        frontier = Stack()
        frontier.push(initial_state)
        explored = []
        while not frontier.isEmpty():
            node = frontier.pop()
            if problem.isGoalState(node):
                state = node
                while state in parent_map.keys():
                    (state, direction, cost) = parent_map[state]
                    actions.append(direction)
                actions = actions[::-1]
                return actions
            if not node in explored:
                explored.append(node)
                for (state, direction, cost) in problem.getSuccessors(node):
                    if not state in explored:
                        frontier.push(state)
                        parent_map[state] = (node, direction, cost)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    from util import Queue
    parent_map = {}
    actions = []
    initial_state = problem.getStartState()
    if problem.isGoalState(initial_state):
        return actions
    else:
        frontier = Queue()
        frontier.push(initial_state)
        explored = []
        while not frontier.isEmpty():
            node = frontier.pop()
            explored.append(node)
            if problem.isGoalState(node):
                state = node
                while state in parent_map.keys():
                    (state, direction, cost) = parent_map[state]
                    actions.append(direction)
                actions = actions[::-1]
                return actions
            for (state, direction, cost) in problem.getSuccessors(node):
                if not ((state in explored) or (state in frontier.list)):
                    frontier.push(state)
                    parent_map[state] = (node, direction, cost)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    from util import PriorityQueue
    parent_map = {}
    actions = []
    initial_state = problem.getStartState()
    if problem.isGoalState(initial_state):
        return actions
    else:
        frontier = PriorityQueue()
        cumulative_cost = 0
        frontier.push(initial_state, cumulative_cost)
        explored = []
        while not frontier.isEmpty():
            node = frontier.pop()
            if problem.isGoalState(node):
                state = node
                while state in parent_map.keys():
                    (state, direction, cost) = parent_map[state]
                    actions.append(direction)
                actions = actions[::-1]
                return actions
            if node in parent_map.keys():
                (_, _, cumulative_cost) = parent_map[node]
            if not node in explored:
                explored.append(node)
                for (state, direction, cost) in problem.getSuccessors(node):
                    if not ((state in explored) or (state in frontier.heap)):
                        total_cost = cumulative_cost + cost
                        frontier.push(state, total_cost)
                        if state in parent_map.keys():
                            (node_temp, direction_temp, cost_temp) = parent_map[state]
                            if cost_temp > total_cost:
                                parent_map[state] = (node, direction, total_cost)
                        else:
                            parent_map[state] = (node, direction, total_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    from util import PriorityQueue
    parent_map = {}
    actions = []
    initial_state = problem.getStartState()
    if problem.isGoalState(initial_state):
        return actions
    else:
        frontier = PriorityQueue()
        f_value = 0 + heuristic(initial_state, problem)
        frontier.push(initial_state, f_value)
        explored = []
        while not frontier.isEmpty():
            node = frontier.pop()
            if problem.isGoalState(node):
                state = node
                while state in parent_map.keys():
                    (state, direction, cost) = parent_map[state]
                    actions.append(direction)
                actions = actions[::-1]
                return actions
            if node in parent_map.keys():
                (_, _, cumulative_cost) = parent_map[node]
            else:
                cumulative_cost = 0
            if not node in explored:
                explored.append(node)
                for (state, direction, cost) in problem.getSuccessors(node):
                    if not ((state in explored) or (state in frontier.heap)):
                        total_cost = cumulative_cost + cost
                        f_value = total_cost + heuristic(state, problem)
                        frontier.push(state, f_value)
                        if state in parent_map.keys():
                            (node_temp, direction_temp, cost_temp) = parent_map[state]
                            if cost_temp > f_value:
                                parent_map[state] = (node, direction, total_cost)
                        else:
                            parent_map[state] = (node, direction, total_cost)


# ______________________________________________________________________________
# Bidirectional Search
# Pseudocode from https://webdocs.cs.ualberta.ca/%7Eholte/Publications/MM-AAAI2016.pdf

def bidirectionalSearch(problem, heuristic=nullHeuristic, useEpsilon=False, useFractional=False, pv=0.5):
    if useEpsilon:
        e = 1
    else:
        e = 0
    initialState = problem.getStartState()
    goalState = problem.goal

    import copy
    problemF = copy.deepcopy(problem)
    problemB = copy.deepcopy(problem)
    problemB.startState = goalState
    problemB.goal = initialState

    gF, gB = {initialState: 0}, {goalState: 0}
    openF, openB = [initialState], [goalState]
    closedF, closedB = [], []
    parentF, parentB = {}, {}
    U = float('inf')

    if useFractional:
        p = pv
    else:
        p = 0.5

    def extend(U, open_dir, open_other, g_dir, g_other, closed_dir, parent, search_direction):
        """Extend search in given direction"""
        n = find_key(C, open_dir, g_dir, search_direction)

        open_dir.remove(n)
        closed_dir.append(n)

        for (c, direction, cost) in problem.getSuccessors(n):
            if c in open_dir or c in closed_dir:
                if g_dir[c] <= g_dir[n] + cost:
                    continue
                open_dir.remove(c)

            g_dir[c] = g_dir[n] + cost
            open_dir.append(c)
            parent[c] = (n, direction)

            if c in open_other:
                U = min(U, g_dir[c] + g_other[c])

        return U, open_dir, closed_dir, g_dir

    def find_min(open_dir, g, search_direction):
        """Finds minimum priority, g and f values in open_dir"""
        # pr_min_f isn't forward pr_min instead it's the f-value
        # of node with priority pr_min.
        pr_min, pr_min_f = float('inf'), float('inf')
        for n in open_dir:
            if search_direction == 'F':
                f = g[n] + heuristic(n, problemF)
            else:
                f = g[n] + heuristic(n, problemB)

            if useEpsilon:
                pr = max(f, 2 * g[n] + 1)
            elif useFractional:
                if search_direction == 'F':
                    pr = max(f, g[n] / float(p))
                else:
                    minus_p = 1-float(p)    
                    pr = max(f, g[n] / minus_p)
            else:
                pr = max(f, 2 * g[n])
            pr_min = min(pr_min, pr)
            pr_min_f = min(pr_min_f, f)

        return pr_min, pr_min_f, min(g.values())

    def find_key(pr_min, open_dir, g, search_direction):
        """Finds key in open_dir with value equal to pr_min
        and minimum g value."""
        m = float('inf')
        node = None
        for n in open_dir:
            if search_direction == 'F':
                if useEpsilon:
                    pr = max(g[n] + heuristic(n, problemF), 2 * g[n] + 1)
                elif useFractional:
                    pr = max(g[n] + heuristic(n, problemF), g[n] / float(p))
                else:
                    pr = max(g[n] + heuristic(n, problemF), 2 * g[n])
            else:
                if useEpsilon:
                    pr = max(g[n] + heuristic(n, problemB), 2 * g[n] + 1)
                elif useFractional:
                    minus_p = 1-float(p)    
                    pr = max(g[n] + heuristic(n, problemB), g[n] / minus_p)
                else:
                    pr = max(g[n] + heuristic(n, problemB), 2 * g[n])
            if pr == pr_min:
                if g[n] < m:
                    m = g[n]
                    node = n

        return node

    while openF and openB:
        pr_min_f, f_min_f, g_min_f = find_min(openF, gF, 'F')
        pr_min_b, f_min_b, g_min_b = find_min(openB, gB, 'B')
        C = min(pr_min_f, pr_min_b)

        if U <= max(C, f_min_f, f_min_b, g_min_f + g_min_b + e):
            # Get an interesect node between openF and openB
            intersect_list = list(set(openF) & set(openB))
            # intersect exists
            if intersect_list:
                intersect = intersect_list[0]
            # intersect does NOT exist
            else:
                if len(openF) <= len(openB):
                    intersect = openF[0]
                    openB.append(intersect)
                else:
                    intersect = openB[0]
                    openF.append(intersect)
            actionsF = []
            # Get actions from intersect to initState
            state = intersect
            while state in parentF.keys():
                (state, direction) = parentF[state]
                actionsF.append(direction)
                if state == initialState:
                    break
            actionsF = actionsF[::-1]
            # Get actions from intersect to goalState
            actionsB = []
            state = intersect
            while state in parentB.keys():
                (state, direction) = parentB[state]
                actionsB.append(direction)
                if state == goalState:
                    break
            # solution
            solution = actionsF
            for action in actionsB:
                if action == 'North':
                    solution.append('South')
                if action == 'East':
                    solution.append('West')
                if action == 'South':
                    solution.append('North')
                if action == 'West':
                    solution.append('East')
            # return U
            return solution

        if C == pr_min_f:
            # Extend forward
            U, openF, closedF, gF = extend(U, openF, openB, gF, gB, closedF, parentF, 'F')
        else:
            # Extend backward
            U, openB, closedB, gB = extend(U, openB, openF, gB, gF, closedB, parentB, 'B')

    return float('inf')


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
###
bis = bidirectionalSearch
