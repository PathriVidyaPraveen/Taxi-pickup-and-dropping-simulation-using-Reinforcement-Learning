# 🤖 Q-Learning with Gymnasium Environments

This project demonstrates the implementation of **Q-learning**, a fundamental algorithm in reinforcement learning, to train agents in discrete [Gymnasium](https://gymnasium.farama.org/) environments.  
The goal is to maximize the cumulative rewards obtained by the agent through interaction with its environment.

---

## ✨ Features
- **Q-Table Initialization**: Sets up the Q-table for state-action value storage.  
- **ε-Greedy Exploration**: Balances exploration of new actions and exploitation of known optimal actions.  
- **Q-Value Updates**: Implements the Bellman equation to iteratively update Q-values.  
- **Training & Evaluation**: Trains the agent over multiple episodes and evaluates its performance.  
- **Learning Curve Plotting**: Visualizes the agent's learning progress (rewards vs. episodes).  
- **Simulation**: Allows for the simulation of the trained agent's policy.  

---

## 🚀 Environments Explored
This project demonstrates Q-learning on discrete Gymnasium environments:

- **Taxi-v3**:  
  A grid-world environment where the agent controls a taxi that must:
  - Pick up a passenger at one location  
  - Drop them off at another  
  - Learn navigation, pick-up, and drop-off actions to maximize rewards.  

---

## 🛠️ Installation

Clone the repository and install the necessary dependencies:

```bash
# Clone the repository (replace with your repo URL)
git clone <your-repository-url>
cd <your-project-directory>
```

```bash

# Install Python dependencies
pip install gymnasium numpy matplotlib
```
💡 Q-Learning Concepts
Q-Table
The Q-table is a lookup table where each entry Q(s,a) represents the estimated maximum future reward for taking action a in state s. The agent learns to populate this table over time.

ε-Greedy Exploration
This strategy ensures the agent explores new actions while also exploiting its current knowledge.

With probability ε, the agent chooses a random action (exploration).

With probability 1−ε, the agent chooses the action with the highest Q-value (exploitation).

ε typically decays over time, leading to more exploitation as the agent learns.

Q-Value Update Rule
The Q-values are updated using the following formula:

 Q(s,a)←Q(s,a)+α[r+γmax a′Q(s′,a′)−Q(s,a)]  
 
Where:

s: Current state

a: Action taken

r: Reward received

s′: Next state

a′: All possible actions from the next state

α (Learning Rate): How much new information overrides old information (0 to 1).

γ (Discount Factor): Importance of future rewards (0 to 1).

📝 Usage
Run the Q-learning agent on the Taxi-v3 environment using the provided script or Jupyter Notebook.

Each episode involves the taxi picking up a passenger and dropping them at the destination.

📊 Results
After training, you will see:

Learning Curve Plot: A graph showing rewards vs. episodes.

Simulation Output: A sequence of states, actions, and rewards as the trained agent navigates the environment.

For environments like Taxi-v3, env.render() will visually display the agent's actions in the ASCII grid.

Taxi-v3 Learning Curve Example:

Early episodes show erratic performance due to exploration.

Over time, the curve stabilizes as the agent learns an optimal policy.

🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests to:

Improve the code

Add more environments

Enhance analysis and visualizations
