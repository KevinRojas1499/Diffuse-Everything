"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def scale(self, t):
    """Scale as in EDM"""
    pass

  @abc.abstractmethod
  def sigma(self, t):
    """Sigma as in EDM"""
    pass
  
  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def marginal_prob_std(self, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass
  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def update_fn(self, x, t, score, dt, method):
    """Update the backwards process, usually its a Euler-Maruyama step"""
    pass

  @abc.abstractmethod
  def denoise(self, x, t, noise):
    """Removes the noise from a noisy input"""
  
  @abc.abstractmethod
  def drift(self, x, t, score, method):
    """Returns the drift either for the forward/backward/probability"""
  
class VP(SDE):

  def __init__(self,T=1.,delta=1e-5, linear_start=0.00085, linear_end=0.0120):
    super().__init__()
    self._T = T
    self.delta = delta
    self.linear_start = linear_start
    self.linear_end = linear_end
  
  @property
  def T(self):
    return self._T

  def beta(self, t):
    return 500 * (self.linear_start**.5 * (1-t) + t * self.linear_end**.5)**2
  
  
  def beta_int(self, t):
    dif = self.linear_end**.5 - self.linear_start**.5
    return 500 * ( (self.linear_start**.5 * (1-t) + t * self.linear_end**.5)**3 /(3 * dif) - self.linear_start**1.5/(3 * dif) )
  
  def scale(self, t):
    big_beta = self.beta_int(t)
    return torch.exp(-big_beta)
  
  def sigma(self,t):
    big_beta = self.beta_int(t)
    return (torch.exp(2 * big_beta) - 1)**.5
  
  def marginal_prob(self, x, t):
    # If    x is of shape [B, H, W, C]
    # then  t is of shape [B, 1, 1, 1] 
    # And similarly for other shapes
    big_beta = self.beta_int(t)
    return torch.exp(-big_beta) * x, 1 - torch.exp(-2 * big_beta)
  
  def marginal_prob_std(self, t):
    return (1 - torch.exp(-2 * self.beta_int(t)))**.5

  def perturb_x(self, x, t, dt):
    t_shaped = t.reshape(-1, *([1] * len(x.shape[1:])))
    bt = self.beta_int(t_shaped)
    bs = self.beta_int(t_shaped + dt)
    exp = torch.exp(bt - bs)
    return x * exp + torch.randn_like(x) * (1 - exp**2)**.5

  def prior_sampling(self, shape, device):
    return torch.randn(*shape, dtype=torch.float, device=device)

  
  def drift(self, x, t, score, method='probability'):
    t_shaped = t.reshape(-1, *([1] * len(x.shape[1:])))
    bt = self.beta(t_shaped)
    if method == 'forward':
      return -bt * x
    elif method == 'backward':
      return  -bt * (x + 2 * score) 
    else:
      return -bt * (x + score)
  
  def diffusion(self, x, t):
    t_shaped = t.reshape(-1, *([1] * len(x.shape[1:])))
    bt = self.beta(t_shaped)
    return (2 * bt).sqrt()

  def denoise(self, x,t, noise):
    scale =  self.scale(t).view(-1, *([1]*len(x.shape[1:])))
    return (x - (1- scale**2).sqrt() * noise)/(scale + 1e-6)
       
  def update_fn(self, x, t, score, dt, method='probability'):
    t_shaped = t.reshape(-1, *([1] * len(x.shape[1:])))
    bt = self.beta(t_shaped)
    if method == 'probability':
      return x - bt * (x + score) * dt 
    else:
      return x - bt * (x + 2 * score) * dt + (2 * bt * abs(dt))**.5 * torch.randn_like(x)

def get_sde(sde_name):
  if sde_name == 'vp':
    return VP()