# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        food_distances = getFoodDistance(newPos, newFood)
        ghost_distances = getGhostDistance(newPos, newGhostStates, max(newScaredTimes) if newScaredTimes else 0)

        return score + food_distances + ghost_distances

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:  # pacman
                return max_value(state, agentIndex, depth)
            else:  # ghost
                return min_value(state, agentIndex, depth)

        def max_value(state, agentIndex, depth):
            v = float("-inf")
            for act in state.getLegalActions(agentIndex):
                nxt_agent = (agentIndex + 1) % state.getNumAgents()

                v = max(v, value(state.generateSuccessor(agentIndex, act), nxt_agent, depth if nxt_agent > 0 else depth + 1))
            return v

        def min_value(state, agentIndex, depth):
            v = float("inf")
            for act in state.getLegalActions(agentIndex):
                nxt_agent = (agentIndex + 1) % state.getNumAgents()

                v = min(v, value(state.generateSuccessor(agentIndex, act), (agentIndex + 1) % state.getNumAgents(), depth if nxt_agent > 0 else depth + 1))
            return v

        best_act = None
        best_val = float("-inf")

        for act in gameState.getLegalActions(0):
            act_val = value(gameState.generateSuccessor(0, act), 1, 0)

            if act_val > best_val:
                best_val = act_val
                best_act = act

        return best_act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state)
            v = float("-inf")
            for act in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, act)
                v = max(v, min_value(successor, depth + 1, agentIndex + 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state)
            v = float("inf")
            for act in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, act)
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, max_value(successor, depth + 1, 0, alpha, beta))
                else:
                    v = min(v, min_value(successor, depth + 1, agentIndex + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        alpha = float("-inf")
        beta = float("inf")
        best_act = None
        best_score = float("-inf")

        for act in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, act)
            score = min_value(successor, 1, 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_act = act
            alpha = max(alpha, best_score)

        return best_act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def value(self, gameState, agentIndex, currentDepth):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:  # pacman's turn
            return self.max_value(gameState, agentIndex, currentDepth)
        else:  # ghosts' turn
            return self.exp_value(gameState, agentIndex, currentDepth)

    def max_value(self, gameState, agentIndex, currentDepth):
        v = float("-inf")
        legal_act = gameState.getLegalActions(agentIndex)
        for act in legal_act:
            v = max(v, self.value(gameState.generateSuccessor(agentIndex, act), (currentDepth + 1) % gameState.getNumAgents(), currentDepth + 1))
        return v

    def exp_value(self, gameState, agentIndex, currentDepth):
        v = 0
        legal_act = gameState.getLegalActions(agentIndex)
        prob = 1.0 / len(legal_act)
        for act in legal_act:
            v += prob * self.value(gameState.generateSuccessor(agentIndex, act), (currentDepth + 1) % gameState.getNumAgents(), currentDepth + 1)
        return v

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        max_val = float("-inf")
        max_act = None
        for act in gameState.getLegalActions(0):  # pacman
            val = self.value(gameState.generateSuccessor(0, act), 1, 1)
            if val > max_val:
                max_val = val
                max_act = act
        return max_act


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()
    food_distance = getFoodDistance(newPos, newFood)
    ghost_distance = getGhostDistance(newPos, newGhostStates, min(newScaredTimes) if newScaredTimes else 0)
    food_encouragement = currentGameState.getNumFood() * 10

    return score + food_distance + ghost_distance - food_encouragement


# Abbreviation
better = betterEvaluationFunction

# helper functions:
def getFoodDistance(newPos, newFood):
    food_distances = [manhattanDistance(newPos, food) for food in newFood.asList()]
    if food_distances:
        return 1.0 / min(food_distances)
    return 0

def getGhostDistance(newPos, newGhostStates, scareTime):
    ghost_distances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if ghost.scaredTimer == 0]
    score = 0
    if ghost_distances:
        min_distance = min(ghost_distances)
        if min_distance > 0:
            score -= 2.0 / min_distance

    for ghost_state in newGhostStates:
        if ghost_state.scaredTimer > 0:
            scared_ghost_distance = manhattanDistance(newPos, ghost_state.getPosition())
            if scared_ghost_distance > 0:
                score += scareTime / scared_ghost_distance

    return score