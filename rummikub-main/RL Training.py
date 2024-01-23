env = RummikubEnv()
agent = RummikubAgent()

for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False

    while not done:
        action = agent.sample_action(state)
        next_state, reward = env.step(state, action)
        agent.update(state, action, reward, next_state)
        state = next_state

