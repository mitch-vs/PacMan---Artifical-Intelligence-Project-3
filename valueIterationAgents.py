# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for _ in range(self.iterations):
            v_values = util.Counter()

            for s in states:
                if not self.mdp.isTerminal(s):
                    max_q_value = -float('inf')
                    actions = self.mdp.getPossibleActions(s)

                    for action in actions:
                        q_value = self.computeQValueFromValues(s, action)
                        # max_q_value = max(max_q_value, q_value)
                        if q_value > max_q_value:
                            max_q_value = q_value

                    v_values[s] = max_q_value

            self.values = v_values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        v = 0
        for s_prime, t in self.mdp.getTransitionStatesAndProbs(state, action):
            v += t * (self.mdp.getReward(state, action, s_prime) + (self.discount * self.getValue(s_prime)))
        
        return v

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possible_actions = self.mdp.getPossibleActions(state)

        optimal_action = None
        optimal_action_value = None

        if self.mdp.isTerminal(state):
            return None

        for action in possible_actions:
            action_q_value = self.computeQValueFromValues(state, action)
            if optimal_action_value is None or action_q_value > optimal_action_value:
                optimal_action = action
                optimal_action_value = action_q_value

        return optimal_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        num_iterations = 0
        states = self.mdp.getStates()

        while num_iterations < self.iterations:
            for s in states:
                if not self.mdp.isTerminal(s):
                    max_q_value = -float('inf')
                    actions = self.mdp.getPossibleActions(s)
                    v_values = util.Counter()

                    for action in actions:
                        q_value = self.computeQValueFromValues(s, action)
                        if q_value > max_q_value:
                            max_q_value = q_value

                    self.values[s] = max_q_value

                num_iterations += 1
                if num_iterations >= self.iterations:
                    return

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        # Initialize predecessor dict and with empty sets
        predecessors = {s: set() for s in states}
        
        # Add to sets in predecessor dict as needed
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            for action in actions:
                for s_prime, t in self.mdp.getTransitionStatesAndProbs(s, action):
                    if t > 0:
                        predecessors[s_prime].add(s)

        priority_queue = util.PriorityQueue()

        # Loop over non terminal states
        for s in states:
            if not self.mdp.isTerminal(s):
                max_q_value = -float('inf')
                actions = self.mdp.getPossibleActions(s)

                for action in actions:
                    q_value = self.computeQValueFromValues(s, action)
                    if q_value > max_q_value:
                        max_q_value = q_value

                # Calculate difference between max Q value for state s and the existing value for s.
                # Push the negative of the absolute difference of this to the priprity queue
                diff = abs(self.values[s] - max_q_value)
                priority_queue.update(s, -diff)

        # For each iteration pop a state s off the priority queue and do stuff!
        for _ in range(self.iterations):
            if priority_queue.isEmpty():
                return
            
            s = priority_queue.pop()

            # If s is not a terminal state, compute the value of it and update self.values[s]
            if not self.mdp.isTerminal(s):
                max_q_value = -float('inf')
                actions = self.mdp.getPossibleActions(s)

                for action in actions:
                    q_value = self.computeQValueFromValues(s, action)
                    if q_value > max_q_value:
                        max_q_value = q_value

                self.values[s] = max_q_value

            # For each predecessor p of s, if the absolute value of the difference between value of 
            # p and self.values[p] is greater than self.theta, push p to the priprity queue with 
            # priority -diff.
            for p in predecessors[s]:
                max_q_value = -float('inf')
                actions = self.mdp.getPossibleActions(p)

                for action in actions:
                    q_value = self.computeQValueFromValues(p, action)
                    if q_value > max_q_value:
                        max_q_value = q_value

                diff = abs(self.values[p] - max_q_value)
                if diff > self.theta:
                    priority_queue.update(p, -diff)
