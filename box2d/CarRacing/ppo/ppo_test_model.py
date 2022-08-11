import torch

from ppo_agent import CarRacingPPO

car_racing_ppo = CarRacingPPO(cuda=True, seed=0, torch_deterministic=False)
car_racing_ppo.load_agent("./models/best_model_1.pt")
# car_racing_ppo.model.load_state_dict("./models/best_model.pt")
# car_racing_ppo.model.eval()
episodes = 5
env = car_racing_ppo.make_env(100, 0, True, 1, "test_best_1")()

for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _, __, value = car_racing_ppo.get_action_and_value(
            torch.tensor([obs], dtype=torch.float).to(car_racing_ppo.device))

        obs, r, done, info = env.step(action.cpu().detach().numpy()[0])
    print(f"Episode {episode + 1} ends with {info['episode']['r']} reward")
