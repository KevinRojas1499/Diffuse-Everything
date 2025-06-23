import torch
import torch.nn as nn

import utils.graph_lib as Graphs
import utils.sde_lib as SDEs
from models.mmdit import MMDiTModel

class Preconditioner(nn.Module):
    def __init__(self, net, sde : SDEs.SDE, graph : Graphs.Absorbing, mode='multimodal') -> None:
        super().__init__()
        self.net = net
        self.sde = sde
        self.graph = graph
        self.mode = mode
    
    def forward_with_cfg(self, x, disc, t_cont, t_disc, scale=3., mode='multimodal', guidance_final_time=1.):
            cond_cont, cond_disc = self(x,disc,t_cont, t_disc, scale=1.)
            final_time = torch.ones_like(t_disc)
            if mode == 'multimodal':
                t_pert = final_time
                sigma_int = self.graph.sigma_int(t_pert)
                perturbed = self.graph.sample_transition(disc,sigma_int.view(-1,1))
                uncond_cont, _ = self(x,perturbed,t_cont=t_cont, t_disc=t_pert, scale=1.)
                cont_score = scale * cond_cont + (1-scale) * uncond_cont
                return cont_score, cond_disc
            elif mode == 'continuous':
                t_pert = final_time * guidance_final_time 
                sigma_int = self.graph.sigma_int(t_pert)
                perturbed = self.graph.sample_transition(disc,sigma_int.view(-1,1))
                uncond_cont, _ = self(x,perturbed,t_cont=t_cont, t_disc=t_pert, scale=1.)
                cont_score = scale * cond_cont + (1-scale) * uncond_cont
                return cont_score, cond_disc
            elif mode == 'discrete':
                t_pert = final_time * guidance_final_time 
                mean, var = self.sde.marginal_prob(x,t_pert.view(-1,1,1,1))
                perturbed = mean + var**.5 * torch.randn_like(mean)
                _, uncond_disc = self(perturbed,disc,t_cont=t_pert, t_disc=t_disc, scale=1.)
                disc_score = scale * cond_disc + (1-scale) * uncond_disc
                return cond_cont, disc_score

    def forward(self, x, disc, t_cont, t_disc, scale=1., mode=None, guidance_final_time=1.,**kwargs):
        if not self.training and scale != 1.:
            if mode is None:
                mode = self.mode
            return self.forward_with_cfg(x,disc, t_cont, t_disc, scale=scale, mode=mode, guidance_final_time=guidance_final_time)
        else:
            sigma_disc = self.graph.sigma_int(t_disc)
            esigm1_log = torch.where(sigma_disc < 0.5, torch.expm1(sigma_disc), sigma_disc.exp() - 1).log().to(x.dtype).view(-1,1,1)
            mask = disc != 49407
            cont_score, disc_score = self.net(text_tokens=disc, image=x, 
                                            image_time_cond=self.sde.marginal_prob_std(t_cont), 
                                            text_time_cond=t_disc, 
                                            text_mask=mask,
                                            detach_hidden=[False, False])
            disc_score = disc_score - esigm1_log
            return cont_score, disc_score
        

def get_model(name, img_res, vocab_size,context_len, num_channels, net_opts, project_hidden=False):
    if name == 'mmdit':
        return MMDiTModel(img_res, img_channels=num_channels, vocab_size=vocab_size, context_len=context_len, project_hidden=project_hidden, **net_opts)
    
def get_preconditioned_model(net, sde, graph, mode='multimodal'):
    return Preconditioner(net,sde,graph, mode)