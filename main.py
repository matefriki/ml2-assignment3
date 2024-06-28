""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib.backends.backend_pdf import PdfPages

import tqdm

def get_sigmas(sigma_1, sigma_L, L):
    # geometric progression for noise levels from \sigma_1 to \sigma_L 
    return torch.tensor(np.exp(np.linspace(np.log(sigma_1),np.log(sigma_L), L)))

def generate_data(n_samples):
    """ Generate data from 3-component GMM

        Requirements for the plot: 
        fig1 
            - this plot should contain a 2d histogram of the generated data samples

    """
    # n_samples = 10
    x, mu, sig, a = torch.zeros((n_samples,2)), None, None, None
    fig1 = plt.figure(figsize=(5,5))
    plt.title('Data samples')

    """ Start of your code
    """

    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Set the seed
    set_seed(321354645)

    K = 3
    a = np.array([1/3, 1/3, 1/3])
    mu = np.array([[1,1], [3,1], [2,3]]) # this is not mu, because apparently mu has already been defined (line 32)
    mu = mu/4
    sig = np.array([0.01*np.eye(2) for k in range(K)])
    to_sample = np.array(np.floor(n_samples*a), dtype=np.int32)
    to_sample[-1] = n_samples - np.sum(to_sample[:-1])

    # we can use the same decomposition for sampling because all components have the same covariance
    L = np.linalg.cholesky(sig[0] + 1e-6 * np.eye(sig[0].shape[0]))
    idx = 0
    x_np = np.zeros((n_samples, 2))
    for k in range(K):
        z = np.random.normal(size=(to_sample[k], mu[0].shape[0], 1))
        x_np[idx:idx + to_sample[k]] = mu[k] + np.matmul(L, z).reshape(to_sample[k], mu[0].shape[0])
        idx += to_sample[k]

    x = torch.from_numpy(x_np)
    plt.hist2d(x_np[:,0], x_np[:,1], bins=128, cmap='viridis')  # maybe it should be bins=32
    plt.colorbar()  # Add a colorbar to a plot

    """ End of your code
    """

    return x, (mu, sig, a), fig1

def dsm(x, params):
    """ Denoising score matching
    
        Requirements for the plots:
        fig2
            - ax2[0] contains the histogram of the data samples
            - ax2[1] contains the histogram of the data samples perturbed with \sigma_1
            - ax2[2] contains the histogram of the data samples perturbed with \sigma_L
        fig3
            - this plot contains the log-loss over the training iterations
        fig4
            - ax4[0,0] contains the analytic density for the data samples perturbed with \sigma_1
            - ax4[0,1] contains the analytic density for the data samples perturbed with an intermediate \sigma_i 
            - ax4[0,2] contains the analytic density for the data samples perturbed with \sigma_L

            - ax4[1,0] contains the analytic scores for the data samples perturbed with \sigma_1
            - ax4[1,1] contains the analytic scores for the data samples perturbed with an intermediate \sigma_i 
            - ax4[1,2] contains the analytic scores for the data samples perturbed with \sigma_L
        fig5
            - ax5[0,0] contains the learned density for the data samples perturbed with \sigma_1
            - ax5[0,1] contains the learned density for the data samples perturbed with an intermediate \sigma_i 
            - ax5[0,2] contains the learned density for the data samples perturbed with \sigma_L

            - ax5[1,0] contains the learned scores for the data samples perturbed with \sigma_1
            - ax5[1,1] contains the learned scores for the data samples perturbed with an intermediate \sigma_i 
            - ax5[1,2] contains the learned scores for the data samples perturbed with \sigma_L
    """

    fig2, ax2 = plt.subplots(1,3,figsize=(10,3))
    ax2[0].hist2d(x.cpu().numpy()[:,0],x.cpu().numpy()[:,1],128), ax2[0].set_title(r'data $x$')
    ax2[1].set_title(r'data $x$ with $\sigma_{1}$')
    ax2[2].set_title(r'data $x$ with $\sigma_{L}$')

    fig3, ax3 = plt.subplots(1,1,figsize=(5,3))
    ax3.set_title('Log loss over training iterations')

    # plot analytic density/scores (fig4) vs. learned by network (fig5)
    fig4, ax4 = plt.subplots(2,3,figsize=(16,10))
    fig5, ax5 = plt.subplots(2,3,figsize=(16,10))

    mu, sig, a = params

    """ Start of your code
    """
    n_samples = x.shape[0]
    space_dimension = x.shape[1]
    sigma_1 = 0.01
    sigma_L = 0.4 # 0.3
    L = 30

    Net = None # TODO: replace with torch.nn module
    sigmas_all = get_sigmas(sigma_1, sigma_L, L) # DONE: replace with the L noise levels
    print("Sigmas: ", sigmas_all)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # TASK 3.1
    x_bar = torch.zeros((L, n_samples, space_dimension))
    for l in range(L):
        z = np.random.normal(size=(n_samples, space_dimension))
        x_bar[l] = x + sigmas_all[l]*z
    ax2[1].hist2d(x_bar[0].cpu().numpy()[:,0],x_bar[0].cpu().numpy()[:,1],128)
    xlim1 = ax2[1].get_xlim()
    ylim1 = ax2[1].get_ylim()
    ax2[2].hist2d(x_bar[-1].cpu().numpy()[:,0],x_bar[-1].cpu().numpy()[:,1],128)
    # ax2[2].set_xlim(xlim1)
    # ax2[2].set_ylim(ylim1)
    # print(x_bar.shape)

    # TASK 3.2

    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.W1 = nn.Linear(input_size, hidden_size, bias=True)
            self.W2 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.W3 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.W4 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.W5 = nn.Linear(hidden_size, output_size, bias=True)

        def forward(self, ipt):
            tmp = F.elu(self.W1(ipt))
            tmp = F.elu(self.W2(tmp))
            tmp = F.elu(self.W3(tmp))
            tmp = F.elu(self.W4(tmp))
            return self.W5(tmp)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    classifier_net = SimpleMLP(input_size=3,hidden_size=64,output_size=1).to(device)
    print('Learnable params=%i' %count_parameters(classifier_net))

    # TASK 3.3

    # Repeat sigma to match the size of x_bar
    sigma_expanded = sigmas_all.view(L, 1, 1).expand(-1, n_samples, 1)
    # Concatenate x_bar and sigma_expanded along the last dimension
    x_bar_reshaped = torch.cat((x_bar, sigma_expanded), dim=2)
    # Reshape x_fin to have shape [50000, 3]
    x_bar_reshaped = x_bar_reshaped.view(-1, space_dimension+1).float()
    # x_bar_reshaped.requires_grad_(True)
    # print(x_bar_reshaped.shape)  # This should print torch.Size([50000, 3])

    class DSMLoss(torch.nn.Module):
        def __init__(self):
            super(DSMLoss, self).__init__()

        def forward(self, x_bar, x_original, sigma, gradients):
            # Calculate the analytical gradient of the log probability
            analytical_grad = -1 * (x_bar - x_original) / (sigma**2)
            # Calculate the loss
            loss = torch.mean(sigma**2 * torch.sum((analytical_grad - gradients) ** 2, dim=1))
            return loss

    # Usage:
    # Assuming `predictions` are outputs from your model and other tensors are prepared

    def noisy_shape(noisy_input, noise_value):
        auxN, auxM = noisy_input.shape  # Get the shape of the input tensor
        sigma_col = torch.full((auxN, 1), noise_value)  # Create a column tensor filled with the value s
        z = torch.cat((noisy_input, sigma_col), dim=1)  # Concatenate x with the sigma column
        return z

    def print_parameters(net, msg="Parameters"):
        print(msg)
        for name, param in net.named_parameters():
            if "W1" in name:
                if param.requires_grad:
                    print(name, param.data)

    n_epochs = 100
    optimizer = optim.Adam(classifier_net.parameters(), lr=.001, weight_decay=1e-4)

    # criterion = nn.MSELoss()
    loss_function = DSMLoss()
    loss_all = []

    # print_parameters(classifier_net)

    for epoch in tqdm.tqdm(range(n_epochs)):
        running_loss = 0.0
        for l in np.random.permutation(range(L)):
            sigma = sigmas_all[l]
            z = np.random.normal(size=(n_samples, space_dimension))
            noisy_input = (x + sigma*z).float()
            noisy_input.requires_grad_(True)
            noisy_input_reshaped = noisy_shape(noisy_input,sigma)
            noisy_input_reshaped = noisy_input_reshaped.to(device)
            noisy_input = noisy_input.to(device)



            predictions = classifier_net.forward(noisy_input_reshaped)
            grad_outputs = torch.ones_like(predictions)
            gradients = torch.autograd.grad(predictions, noisy_input_reshaped, grad_outputs, create_graph=True)[0][:,:-1]
            # print(gradients)
            # print(gradients.shape)
            loss = loss_function(noisy_input, x.to(device), sigma, gradients)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # print("Gradient of first layer weights after backward: ", classifier_net.W1.weight.grad)
            optimizer.step()
            running_loss += loss.item()
        loss_all.append(running_loss)

    # print_parameters(classifier_net)

    ax3.plot(np.log(np.asarray(loss_all))
             ) # TODO: maybe, plot log loss instead of loss

    # TASK 3.4

    # Define the Gaussian Mixture Model PDF function
    def gmm_pdf(mu, sigma, pi, x):
        res = 0.0
        for i in range(len(mu)):
            res += pi[i] * (1/(2 * np.pi * sigma)) * np.exp(-np.linalg.norm(x - mu[i])**2 / (2 * sigma))
        return res
    
    def get_arrow_scale(magnitude, scale_factor=15):
        return np.max(magnitude)*scale_factor

    xlim1 = np.array(ax2[1].get_xlim())
    ylim1 = np.array(ax2[1].get_ylim())

    xlimL = np.array(ax2[2].get_xlim())
    ylimL = np.array(ax2[2].get_ylim())

    xlims = [xlim1, (xlim1 + xlimL)/2, xlimL]
    ylims = [ylim1, (ylim1 + ylimL)/2, ylimL]
    xticks = [np.linspace(lim[0], lim[1], num=5) for lim in xlims]
    yticks = [np.linspace(lim[0], lim[1], num=5) for lim in ylims]
    noiselevels = [sigma_1,sigmas_all[int(L/3)],sigma_L] # DONE: replace these with the chosen noise levels for plotting the density/scores
    

    for nl in range(3):

        x1 = np.linspace(xlims[nl][0], xlims[nl][1], num=32)
        y1 = np.linspace(ylims[nl][0], ylims[nl][1], num=32)

        X, Y = np.meshgrid(x1, y1)
        Z = np.zeros_like(X)

        # Calculate the GMM PDF for each point in the meshgrid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = gmm_pdf(mu, 0.01 + noiselevels[nl], a, np.array([X[i, j], Y[i, j]]))

        ax4[0,nl].contourf(X, Y, Z, levels=50, cmap='viridis')
        ax4[0, nl].set_xticks(xticks[nl]) # TODO: for some reason, the ticks are not working
        ax4[0, nl].set_yticks(yticks[nl])
        ax4[0, nl].set_xlim(xlims[nl])
        ax4[0, nl].set_ylim(ylims[nl])

        # Compute gradients
        Gx, Gy = np.gradient(Z, x1, y1)
        magnitude = np.hypot(Gx, Gy)
        scale = get_arrow_scale(magnitude)
        ax4[1,nl].quiver(X, Y, Gx, Gy, magnitude, angles = 'uv', scale=scale, cmap='viridis')


   # TASK 3.5
   # Energy_function
    def energy_pdf(energy_net, sigma, x):
        x_tens = torch.tensor([x[0], x[1], sigma]).float().to(device)
        res = energy_net.forward(x_tens)
        return res

    arrowscales = [500,300,200]
    for nl in range(3):

        x1 = np.linspace(xlims[nl][0], xlims[nl][1], num=32)
        y1 = np.linspace(ylims[nl][0], ylims[nl][1], num=32)

        X, Y = np.meshgrid(x1, y1)
        Z = np.zeros_like(X)

        # Calculate the GMM PDF for each point in the meshgrid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = energy_pdf(classifier_net, noiselevels[nl], np.array([X[i, j], Y[i, j]]))

        ax5[0,nl].contourf(X, Y, Z, levels=50, cmap='viridis')
        ax5[0, nl].set_xticks(xticks[nl]) # TODO: for some reason, the ticks are not working
        ax5[0, nl].set_yticks(yticks[nl])
        ax5[0, nl].set_xlim(xlims[nl])
        ax5[0, nl].set_ylim(ylims[nl])

        # Compute gradients
        Gx, Gy = np.gradient(Z, x1, y1)
        magnitude = np.hypot(Gx, Gy)
        scale = get_arrow_scale(magnitude)
        ax5[1,nl].quiver(X, Y, Gx, Gy, magnitude, angles = 'uv', scale=scale, cmap='viridis')

    Net = classifier_net
    """ End of your code
    """

    for idx, noiselevel in enumerate(noiselevels):
        ax4[0,idx].set_title(r'$\sigma$=%f' %noiselevel)
        ax5[0,idx].set_title(r'$\sigma$=%f' %noiselevel)

        ax4[0,idx].set_xticks([]), ax4[0,idx].set_yticks([])
        ax4[1,idx].set_xticks([]), ax4[1,idx].set_yticks([])
        ax5[0,idx].set_xticks([]), ax5[0,idx].set_yticks([])
        ax5[1,idx].set_xticks([]), ax5[1,idx].set_yticks([])

    ax4[0,0].set_ylabel('analytic density'), ax4[1,0].set_ylabel('analytic scores')
    ax5[0,0].set_ylabel('learned density'), ax5[1,0].set_ylabel('learned scores')

    return Net, sigmas_all, (fig2, fig3, fig4, fig5)

def sampling(Net, sigmas_all, n_samples):
    """ Sampling from the learned distribution
    
        Requirements for the plots:
            fig6
                - ax6[0] contains the histogram of the data samples
                - ax6[1] contains the histogram of the generated samples
    
    """

    fig6, ax6 = plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    ax6[0].set_title(r'data $x$')
    ax6[1].set_title(r'samples')

    """ Start of your code
    """
    epsilon = 0.01
    T = 10
    space_dimension = x.shape[1]


    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    Net.to(device)
    Net.eval()

    x_samples = torch.randn(n_samples, space_dimension, device=device).float()

    sigmas_all = sigmas_all.to(device).float()
    for sigma in reversed(sigmas_all):
        sigma = sigma.float()
        alpha_i = epsilon * (sigma ** 2) / (sigmas_all[0] ** 2)
        for t in range(T):
            x_samples.requires_grad_(True)
            sigma_expanded = sigma.repeat(n_samples, 1).float()

            z_t = torch.randn(n_samples, space_dimension, device=device).float()

            # Step
            score = Net(torch.cat([x_samples, sigma_expanded], dim=1))
            grad = torch.autograd.grad(score.sum(), x_samples, create_graph=True)[0]
            x_samples = x_samples.detach() + 0.5 * alpha_i * grad + torch.sqrt(alpha_i) * z_t

            # Memory management
            x_samples = x_samples.detach()
            del z_t, sigma_expanded, score, grad
            if device == torch.device('cuda'):
                torch.cuda.empty_cache()
    x_samples = x_samples.cpu().numpy()
    x_np = x.numpy()

    ax6[0].hist2d(x_np[:, 0], x_np[:, 1], bins=128, cmap='viridis')
    ax6[1].hist2d(x_samples[:, 0], x_samples[:, 1], bins=128, cmap='viridis')

    """ End of your code
    """

    return fig6

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    # generate data
    x, params, fig1 = generate_data(n_samples=10000)

    # denoising score matching
    Net, sigmas_all, figs = dsm(x=x, params=params)


    # sampling
    fig6 = sampling(Net=Net, sigmas_all=sigmas_all, n_samples=5000)

    pdf.savefig(fig1)
    for f in figs:
        pdf.savefig(f)

    pdf.savefig(fig6)

    pdf.close()

