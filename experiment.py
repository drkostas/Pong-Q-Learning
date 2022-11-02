""" 
Author: Konstantinos Georgiou
Date: 10/10/2022
Description: This file contains the code for a RL agent 
             that finds the optimal policy for the zoo-monkey problem
             using MDP Value Iteration.
Example Usage: python experiment.py 2 4 0.5 -100 0.9
               Which means: N_min = 2, N_max = 4, p = 0.5, R_over = -100, gamma = 0.9
"""


from collections import defaultdict
import sys
import random
import math
from itertools import combinations
import pickle
import time
from pprint import pprint
from typing import Dict, List, Tuple, Union, Any


class Agent:

    def __init__(self, N_min: int, N_max: int, p: float, R_over: float, 
                 gamma: float, max_iterations: int, theta: float) -> None:
        """ Initializes the agent"""
        self.N_min = N_min
        self.N_max = N_max
        self.p = p
        self.R_over = R_over
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.theta = theta
        # Generate the possible states and actions
        self.possible_states, self.possible_actions = self.get_possible_states_and_actions()

    def get_possible_states_and_actions(self) -> Tuple[List[int], Dict[int, List[int]]]:
        """ Generates all possible states and actions for the current problem"""
        # The states are between [0, 3*N_max] (max possible monkeys)
        possible_states = list(range(0, 3*self.N_max+1))
        # action is the number of monkeys to release; ranges from [0, state]
        possible_actions = {}
        for s in possible_states:
            possible_actions[s] = list(range(s, -1, -1))
        return possible_states, possible_actions

    def value_iteration(self, V0: float) -> Tuple[Dict[int, float], # V
                                            Dict[int, Dict], # Q
                                            List[Dict], # history
                                            int # Total Iterations
                                            ]:
        """ Performs value iteration on the state value function
        Args:
            V0: The starting value for the state value function
        Returns:
            V: The state value function
            Q: The action value function
            history: History of the iterations (used for plotting)
            Total Iterations: The total number of iterations"""


        # --- Initialize values before Value Iteration --- #
        new_state = random.choice(range(self.N_min, 2*self.N_max+1)) # state: number of monkeys
        print(f"Initialized with {new_state} Monkeys")
        V = self.initialize_V(V0) # V is a dictionary that maps states to V[s]
        Q = defaultdict(dict) # Q is a dictionary that maps states to a dictionary of actions to expected returns
        history = []  # history is a list of dicts that contains the history of policies and V (for plotting)

        # --- Loop until convergence or max iterations or zoo fails --- #
        iteration = 0
        Delta = None
        while not self.should_terminate(new_state, iteration, Delta):
            iteration += 1
            Delta = 0.0
            state = new_state
            # Loop through all possible states
            for s in self.possible_states:
                v_current = V[s]
                # Calculate the expected return for each possible action
                for a in self.possible_actions[s]:
                    # Update the action value function for the current state and action
                    Q[s][a] = self.get_expected_return(s, a, V)
                # Update the state value function for the current state
                V[s] = max(Q[s].values())
                Delta = max(Delta, abs(v_current - V[s]))

            # Get the best action for the current state and value function
            action = self.get_best_action(state, self.possible_actions[state], V)           
            monkeys_left = state - action  # Release the monkeys
            monkeys_born = self.calculate_kids(num_monkeys=monkeys_left)  # Calculate the number of kids born
            new_state = monkeys_left + monkeys_born  # Add the kids to the state

            # Store some historical data for plotting
            V_hist= {f"V[{s}]": V[s] for s in self.possible_states}
            Q_hist = {f"Q[{s}, {a}]": Q[s][a] for s in Q for a in Q[s]}
            history.append({"Delta": Delta, "s": state, 
                            "a": action, "s (after release)": monkeys_left, 
                            "s'": new_state, **V_hist, **Q_hist})
        
        return V, Q, history, iteration

    def initialize_V(self, V0: float) -> Dict[int, float]:
        """ Initializes the state value function"""
        # default value is V0 if state is within [0, 2*N_max], otherwise 0
        V = defaultdict(lambda: 0.0)
        V0 = float(V0)
        for i in range(0, 2*self.N_max+1):
            V[i] = V0
        return V

    def get_expected_return(self, state: int, action: int, V: Dict[int, float]) -> float:
        """ Calculates the expected return
            given a state and action and the 
            current state value function"""

        # Calculate the current reward
        reward = self.calculate_reward(state)
        # reward = self.calculate_reward(state - action)  # alternative reward function
        # Calculate the number of monkeys after the action and before the kids
        state_prime = state - action
        if state_prime > 2*self.N_max:
            return 0.0
        # Calculate every possible sub action and the corresponding
        # probability based on the number of pairs
        sub_states, probabilities = self.get_sub_states_and_probabilities(state_prime)
        # Calculate the expected return
        expected_return = 0.0
        for sub_state, prob in zip(sub_states, probabilities):
            if sub_state <= 2*self.N_max:
                # Non-terminal state
                reward = self.calculate_reward(sub_state)
                expected_return += prob * (reward + self.gamma * V[sub_state])
            else:
                # Terminal state
                expected_return += prob * self.R_over
        return expected_return

    def get_sub_states_and_probabilities(self, state_prime: int) -> Tuple[List[int], List[float]]:
        """ Calculates the sub actions and probabilities given a state"""
        # Number of pairs is the number of monkeys divided by 2 and rounded down
        pairs = math.floor(state_prime/2)
        # Initialize
        sub_states = []
        probabilities = [0.0 for _ in range(0, pairs+1)]
        # Loop through all pairs
        for pair_ind in range(0, pairs+1):
            # The sub action is the number of monkeys plus the current pair index
            sub_states.append(state_prime+pair_ind)
            # Calcualte the possible combinations for (pairs, pair_ind)
            num_combs = len(list(combinations(range(pairs), pair_ind)))
            # Calculate the probability of the current pair index
            probabilities[pair_ind] = num_combs * \
                (self.p**pair_ind)*(1-self.p)**(pairs-pair_ind)
        if sum(probabilities) != 1.0:
            raise ValueError("Probabilities do not sum to 1")
        return sub_states, probabilities

    def calculate_reward(self, state: int) -> Union[float, int]:
        """ Calculates the reward given a number of monkeys (state)"""

        if self.N_min <= state <= self.N_max:
            # Reward = Number of monkeys if number of monkeys
            # is within the range [N_min, N_max]
            return state
        elif state > 2*self.N_max:
            # Reward = R_over if number of monkeys
            # is greater than 2*N_max
            return self.R_over
        else:
            # Reward = 0 if number of monkeys is less
            # than N_min or greater than N_max
            return 0

    def get_best_action(self, state: int, actions: List[int], V: Dict[int, float]) -> int:
        """ Calculates the best action given a state and the current state value function"""
        actions = actions[::-1]
        # Calculate the expected return for each possible action
        exp_returns = [self.get_expected_return(state, a, V) for a in actions]
        # Get the action with the highest expected return
        action = actions[self.argmax(exp_returns)]
        return action

    def calculate_policy(self, V: Dict[int, float]) -> Dict[int, int]:
        """ Calculates the policy given the state value function"""

        # Get the best action for each state
        best_actions = {s: self.get_best_action(s, self.possible_actions[s], V) 
                        for s in self.possible_states if s <= 2*self.N_max}
        return best_actions
    
    def calculate_kids(self, num_monkeys: int) -> int:
        """ Calculates the number of kids given a number of monkeys"""
        kids = 0
        for _ in range(math.floor(num_monkeys/2)):
            # Generate a random number between 0 and 1
            # and add a kid if the number is greater than p
            if self.p >= random.uniform(0, 1):
                kids += 1
        return kids

    def should_terminate(self, num_monkeys: int, iteration: int,
                         Delta: Union[float, None]) -> bool:
        """ Checks if the agent should terminate the value iteration"""
        if Delta is None:
            return True if num_monkeys > 2*self.N_max or iteration >= self.max_iterations else False
        else:
            return True if num_monkeys > 2*self.N_max or iteration >= self.max_iterations or Delta < self.theta else False

    def clean(self, V: Dict[int, Any], Q: Dict[int, Any]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """ Cleans the dictionaries by removing the keys that are not needed"""
        V = {k: v for k,v in V.items() if k <= 2*self.N_max}
        Q = {k: v for k,v in Q.items() if k <= 2*self.N_max}
        return V, Q

    @staticmethod
    def argmax(array):
        """ Returns the index of the maximum value in an array"""
        return array.index(max(array))


def save_file(data: Any, file_name: str) -> None:
    """ Saves a python object to a pickle file"""
    with open(f"outputs/{file_name}", 'wb') as f:
        pickle.dump(data, f)


def main(args):
    """ Main function"""
    print("------ Initializing ------")
    # --- Args Loading and Error Checking --- #
    if len(args) >= 5:
        # Load the arguments
        N_min, N_max, p, R_over, gamma = args[:5]
        # Cast the arguments to the correct type
        N_min, N_max, p, R_over, gamma = int(N_min), int(
            N_max), float(p), float(R_over), float(gamma)
        if len(args) > 5:
            # If more than the 5 required arguments are given,
            # set the requested values for the maxnumber of iterations
            # and initiual values of V[s]
            max_iterations = int(args[5])
            V0 = int(args[6])
            theta = float(args[7])
        else:
            # Else, set the default values
            max_iterations = 200
            V0 = 20
            theta = 0.01
    else:
        # If not enough arguments are given, print an error message
        raise Exception("Invalid number of arguments")

    # Initialize the agent
    agent = Agent(N_min, N_max, p, R_over, gamma, max_iterations, theta)
    print("------ Starting ------")
    # Perform Value Iteration and get the new value function
    start_t = time.time()
    V, Q, history, iterations = agent.value_iteration(V0=V0)
    policy = agent.calculate_policy(V)
    end_t = time.time()
    V, Q = agent.clean(V, Q)

    # Print the results
    print("------ Results ------")
    print("Policy (key is number of monkeys, value is num of monkeys to release):")
    pprint(policy)
    print()
    print("Value Function (key is s, value is V[s])")
    pprint(V)
    print()
    print("Action Value Function (key is s, second key is s', value is Q):")
    pprint(Q)
    print()
    print(f"Finished in {end_t-start_t:.4f} seconds and {iterations} episodes")
    print()
    print("Settings used:")
    print(f"N_min: {N_min}, N_max: {N_max}, p: {p}, R_over: {R_over}, gamma: {gamma}, "\
          f"max iterations: {max_iterations}, V0: {V0}, theta: {theta}")
    print()

    # Save the results for plotting
    save_file(history, "history.pkl")
    save_file(policy, "policy.pkl")
    save_file(V, "V.pkl")
    save_file(Q, "Q.pkl")
    info = {'N_min': N_min, 'N_max': N_max, 'p': p, 'R_over': R_over,
            'gamma': gamma, 'iterations_perc': f"{iterations}/{max_iterations}", 
            'iterations': iterations, "theta": theta,
            'time': f"{end_t-start_t:.4f}", 'V0': V0}
    save_file(info, "info.pkl")


if __name__ == "__main__":
    main(sys.argv[1:])
