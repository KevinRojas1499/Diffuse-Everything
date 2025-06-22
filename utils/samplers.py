import torch
import abc
from functools import partial
from tqdm import tqdm

from utils.sde_lib import SDE
from utils.graph_lib import Absorbing

@torch.no_grad()
def get_euler_maruyama(shape, sde : SDE , model, steps=100, device='cuda', labels=None, clip=None, cfg_scale=0.):
    T = sde.T
    use_cfg = cfg_scale!=0.
    rho = 7
    step_indices = torch.arange(steps, dtype=torch.float32, device=device)
    t_steps = (T + step_indices / (steps - 1) * (sde.delta ** (1 / rho) - T**(1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    time_pts = t_steps
    # time_pts = torch.linspace(1, sde.delta, steps + 1, device=device)

    x_t = sde.prior_sampling((shape),device=device)
    pbar = tqdm(range(steps),leave=False)
    for i in pbar:
        t = time_pts[i].expand(x_t.shape[0])
        dt = time_pts[i + 1] - time_pts[i]
        score = model(x_t, t, labels, clip=clip, cfg=use_cfg, scale=cfg_scale)
        score *= -1./sde.marginal_prob_std(t).view(-1, *([1]*len(score.shape[1:])))
        # x_t = sde.update_fn(x_t, t, score, dt, method='euler')
        t_hat = time_pts[i+1].expand(x_t.shape[0])
        old_drift = sde.drift(x_t,t, score,method='probability')
        x_hat = x_t + old_drift * dt

        score_hat = model(x_hat, t, labels, clip=clip, cfg=use_cfg, scale=cfg_scale)
        new_drift = sde.drift(x_hat,t_hat,score_hat,method='probability')
        x_t = x_t + .5 * dt * (new_drift + old_drift)

    pbar.close()
    return x_t

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@torch.no_grad()    
def get_pc_sampler(model, graph : Absorbing, shape, steps, device):
    x = graph.sample_limit(*shape).to(device)
    eps = graph.delta
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    for i in tqdm(range(steps)):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        score = model(x,graph.sigma_int(t)).exp()
        x = graph.update_fn(score, x, t, dt)
    return x.clamp(0,graph.dim-2) # Make sure we never return the absorbed state

@torch.no_grad()
def get_multimodal_sampler(model, sde : SDE, graph : Absorbing,
        cont_shape, disc_shape, steps, device=torch.device('cpu'), mode='multimodal',cond=None,cfg_scale=1., return_traj=False,
        guidance_left=0., guidance_right=1.,guidance_final_time=1., tau=True):
    """
        For conditional we have 3 options
        1. 'multimodal' - unconditional generation
        2. 'continuous' - condition on the discrete data (text)
        3. 'discrete' - condition on the continuous data (image)
    """
    # For conditional we have 3 options:
    eps = sde.delta # Assume == graph.delta
    
    data_t = sde.prior_sampling(cont_shape, device)
    label_t = graph.sample_limit(*disc_shape).to(device)
    timesteps = torch.linspace(1, eps, steps, device=device)
    timesteps = torch.cat([timesteps, torch.zeros_like(timesteps[:1])]) # t_N = 0
    ones = torch.ones(data_t.shape[0], device=device)
    eps_vect =  torch.ones(data_t.shape[0], device=device) * eps
    update_label, update_data = True, True
    def_guid_scale = cfg_scale 
    if mode == 'continuous':
        assert cond is not None, 'We need a condition for conditional sampling'
        assert cond.shape == disc_shape, f'Condition does not match the appropiate shape, expected {disc_shape} got {cond.shape}'
        t_disc = eps_vect
        label_t = cond
        update_label = False
    elif mode == 'discrete':
        assert cond is not None, 'We need a condition for conditional sampling'
        assert cond.shape == cont_shape, f'Condition does not match the appropiate shape, expected {cont_shape} got {cond.shape}'
        t_cont = eps_vect
        data_t = cond
        update_data = False
    
    if return_traj:
        trajectories = [[data_t, label_t]]
    for i, (t_cur, t_next) in tqdm(enumerate(zip(timesteps[:-1], timesteps[1:])),leave=False):
        dt = t_next - t_cur
        t_cur = t_cur * ones
        t_next = t_next * ones
        if update_data:
            t_cont = t_cur
        if update_label:
            t_disc = t_cur

        guidance_scale = def_guid_scale if t_cur[0].item() > guidance_left and t_cur[0].item() < guidance_right else 1.
        score_cont, score_disc = model(data_t, label_t, t_cont, t_disc, scale=guidance_scale, mode=mode, guidance_final_time=guidance_final_time)
        if update_data:
            score_cont *= -1./sde.marginal_prob_std(t_cont).view(-1, *([1]*len(score_cont.shape[1:])))
            old_drift = sde.drift(data_t,t_cont,score_cont,method='probability')
            x_hat = data_t + old_drift * dt
            if i < steps - 1:
                score_cont_hat, _ = model(x_hat, label_t, t_next, t_disc, scale=guidance_scale, mode=mode, guidance_final_time=guidance_final_time)
                score_cont_hat *= -1./sde.marginal_prob_std(t_next).view(-1, *([1]*len(score_cont.shape[1:])))
                new_drift = sde.drift(x_hat,t_next,score_cont_hat,method='probability')
                data_t = data_t + .5 * dt * (new_drift + old_drift)
            else:
                data_t = x_hat
        if update_label:
            score_disc = score_disc.exp() 
            if i < steps - 1:
                label_t = graph.update_fn(score_disc, label_t, t_disc, abs(dt), tau=tau)
            else:
                label_t = graph.denoise(score_disc, label_t, t_disc)
        if return_traj:
            trajectories.append([data_t, label_t])
        
    if return_traj:
        return data_t, label_t.clamp(0,graph.dim-2), trajectories
    else:
        return data_t, label_t.clamp(0,graph.dim-2)
    
def get_sampling_fn(model, sde, graph, device):
    return partial(get_multimodal_sampler, model=model, sde=sde, graph=graph, device=device)