import abc
import torch
import torch.nn.functional as F

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    

def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass


    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        self.delta = 1e-5

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True
    
    def sigma(self, t):
        return (1 - self.delta) / (1 - (1 - self.delta) * t)
    
    def sigma_int(self, t):
        return -torch.log1p(-(1 - self.delta) * t)

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma_int):
        sigma_int = unsqueeze_as(sigma_int, i[..., None])
        edge = (-sigma_int).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma_int).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma_int):
        move_chance = 1 - (-sigma_int).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def update_fn(self, score, x, t, step_size, tau=True):
        ones = [1] * len(score.shape[1:])
        dsigma = self.sigma(t)


        rev_rate = step_size * dsigma.view(-1,*ones) * self.reverse_rate(x, score)
        if tau:
            # Tau Leaping
            rev_rate[:, :, -1] = 0  
            rev_rate[x != self.dim -1] = 0
            rev_rate.scatter_(-1, x[..., None], torch.zeros_like(rev_rate))

            diffs = torch.arange(self.dim, device=x.device).view(1, 1, self.dim) - x.unsqueeze(-1)
            jump_nums = torch.distributions.poisson.Poisson(rev_rate).sample()
            jump_nums[jump_nums.sum(dim = -1) > 1] = 0 
            overall_jump = torch.sum(jump_nums * diffs, dim=-1)
            x = (x + overall_jump).to(torch.int64)
        else:
            # Euler
            x = self.sample_rate(x, rev_rate)
            
        return x
    
    def denoise(self, score, x, t):
        sigma_int = self.sigma_int(t)[0].view(1,1)

        stag_score = self.staggered_score(score, sigma_int)
        probs = stag_score * self.transp_transition(x, sigma_int)
        probs = probs[..., :-1]
        
        return sample_categorical(probs)
    
    def score_entropy(self, score, sigma_int, x, x0, return_full_entropy=False):
        rel_ind = (x == self.dim - 1)
        
        esigm1 = torch.where(
            sigma_int < 0.5,
            torch.expm1(sigma_int),
            torch.exp(sigma_int) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        entropy = torch.zeros(*x.shape, device=x.device)
        if return_full_entropy:
            #positive term
            pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

            # constant term
            const = ratio * (ratio.log() - 1)
            entropy[rel_ind] += pos_term - neg_term + const
        else:
            entropy[rel_ind] += -neg_term
            
        return entropy
    