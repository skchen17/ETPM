import random
import sys
import torch

from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2c.model import BehavioralModel
from etrcm.stage2c.rollout import play_experience, probe_policy
from etrcm.stage2c.world import Experience, consequence, surface

c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = BehavioralModel(Stage14Config(**c["config"]), "full")
m.load_state_dict(c["model"])
m.eval()
rng = random.Random(912)
with torch.no_grad():
    state = m.initial_state(2)
    for t in range(16):
        experiences = []
        for z in (0, 1):
            action = t % 2
            features = surface(rng, "train")
            experiences.append(Experience(*features, action, consequence(rng, z, action)))
        state, _, _, _ = play_experience(m, state, experiences)
    p, action_logits, forecasts, *_ = probe_policy(m, state, [surface(rng, "novel")] * 2)
    print("choice", p.tolist())
    print("forecasts", forecasts.softmax(-1).tolist())
    print("H", state.H.norm(dim=(-2, -1)).tolist())
