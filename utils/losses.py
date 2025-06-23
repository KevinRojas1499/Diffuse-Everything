import torch
import torch.nn.functional as F
import utils.sde_lib as SDEs
import utils.graph_lib as graphs

def dsm_loss(sde : SDEs.SDE,data, model ,labels=None):
    eps = sde.delta
    times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
    shaped_t = times.reshape(-1,1,1,1) if len(data.shape) > 2 else times.reshape(-1,1)
    mean, variance = sde.marginal_prob(data,shaped_t)
    noise = torch.randn_like(mean,device=data.device)
    perturbed_data = mean + variance**.5 * noise
    flatten_error = ((variance**.5 * model(perturbed_data,times, labels) + noise)**2).view(data.shape[0],-1)
    
    return torch.mean(flatten_error)

def discrete_loss(graph : graphs.Graph, data, model, cond=None):
    """
    Batch shape: [B, L] int. D given from graph
    """
    eps = graph.delta
    t = (1 - eps) * torch.rand(data.shape[0], device=data.device) + eps
    sigma = graph.sigma(t)
    sigma_int = graph.sigma_int(t)
    perturbed_data = graph.sample_transition(data, sigma_int[:, None])
    log_score = model(perturbed_data, sigma_int)
    loss = graph.score_entropy(log_score, sigma_int[:, None], perturbed_data, data)

    loss = (sigma[:, None] * loss).sum(dim=-1)
    return loss.mean()

def combined_loss(sde : SDEs.SDE, graph : graphs.Absorbing, data, label, model, hidden_img, hidden_text): 
    eps = sde.delta
    times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
    times_disc = torch.ones_like(times) * eps
    shaped_t = times.reshape(-1,1,1,1) if len(data.shape) > 2 else times.reshape(-1,1)
    
    mean, variance = sde.marginal_prob(data,shaped_t)
    noise = torch.randn_like(mean,device=data.device)
    perturbed_data = mean + variance**.5 * noise
    
    sigma_int = graph.sigma_int(times_disc)
    perturbed_label = graph.sample_transition(label, sigma_int[:, None])
    
    cont_noise, log_score_disc, img_hidden, text_hidden = model(perturbed_data, perturbed_label, times, times_disc)
        
    # Continuous loss 
    flatten_error_dsm = ((cont_noise - noise)**2).view(data.shape[0],-1).mean(-1)

    # Discrete loss 
    error_disc_dsm = F.cross_entropy(log_score_disc.reshape(-1, log_score_disc.shape[-1]), 
                                     label.flatten().long(), reduction='none')
    # error_disc_dsm = graph.score_entropy(log_score_disc, sigma_int[:, None], perturbed_label, label)
    # error_disc_dsm = (sigma[:, None] * error_disc_dsm).sum(dim=-1)

    # Hidden states losses
    
    # Image hidden loss calculation (if hidden_img is not None)
    if hidden_img is not None and img_hidden is not None:
        normalized = F.normalize(img_hidden, dim=-1)
        hidden_img_normalized = F.normalize(hidden_img, dim=-1)
        img_hidden_loss = -(normalized * hidden_img_normalized).sum(dim=-1).mean()
    else:
        img_hidden_loss = torch.tensor(0.0, device=data.device, requires_grad=True)

    # Text hidden loss calculation (if hidden_text is not None)
    if hidden_text is not None and text_hidden is not None:
        normalized_text = F.normalize(text_hidden, dim=-1)
        hidden_text_normalized = F.normalize(hidden_text, dim=-1)
        text_hidden_loss = -(normalized_text * hidden_text_normalized).sum(dim=-1).mean()
    else:
        text_hidden_loss = torch.tensor(0.0, device=data.device, requires_grad=True)

    return flatten_error_dsm.mean(), error_disc_dsm, img_hidden_loss, text_hidden_loss
    
def get_repa_loss(hidden_rep, true_hidden):
    if hidden_rep is None or true_hidden is None:
        return 0
    normalized = F.normalize(hidden_rep, dim=-1)
    true_hidden_normalized = F.normalize(true_hidden, dim=-1)
    return -(normalized * true_hidden_normalized).sum(dim=-1).mean()

def get_loss(modality, sde : SDEs.SDE=None, graph : graphs.Absorbing=None, use_all_times=False): 
    def loss(data, label, model):
        eps = sde.delta
        if use_all_times or modality == 'multimodal':
            times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
            times_disc = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
        if modality == 'continuous':
            times = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
            times_disc = torch.ones_like(times) * eps
        elif modality == 'discrete':
            times_disc = (torch.rand((data.shape[0]),device=data.device) * (1-eps) + eps) * sde.T
            times = torch.ones_like(times_disc) * eps

        shaped_t = times.reshape(-1,1,1,1) if len(data.shape) > 2 else times.reshape(-1,1)
    
        # Perturb data
        mean, variance = sde.marginal_prob(data,shaped_t)
        noise = torch.randn_like(mean,device=data.device)
        perturbed_data = mean + variance**.5 * noise
        
        sigma_int = graph.sigma_int(times_disc)
        perturbed_label = graph.sample_transition(label, sigma_int[:, None])
        
        # Be careful, in the single modality case one of the two scores will not be accurate
        cont_noise, log_score_disc = model(perturbed_data, perturbed_label, times, times_disc)
        # Get model output and compute losses
        log_dir = {}
        if modality == 'continuous':
            dsm_loss = ((cont_noise - noise)**2).view(data.shape[0],-1).mean()
            log_dir['cont_loss'] = dsm_loss
        elif modality == 'discrete':
            dsm_loss_disc = F.cross_entropy(log_score_disc.reshape(-1, log_score_disc.shape[-1]), 
                                            label.flatten().long(), reduction='none').mean()
            log_dir['disc_loss'] = dsm_loss_disc
        else:
            dsm_loss = ((cont_noise - noise)**2).view(data.shape[0],-1).mean()
            dsm_loss_disc = F.cross_entropy(log_score_disc.reshape(-1, log_score_disc.shape[-1]), 
                                            label.flatten().long(), reduction='none').mean()
            log_dir['cont_loss'] = dsm_loss
            log_dir['disc_loss'] = dsm_loss_disc

        return log_dir
    
    return loss
