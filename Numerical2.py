import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



DIM = 10
BATCH_SIZE = 1024       
N_ITER = 15000
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)


def exact_u_torch(x):
    return torch.sum(torch.sin(np.pi * x), dim=1, keepdim=True)

def source_f_torch(x):
    # -Laplace u = pi^2 * u
    return (np.pi ** 2) * exact_u_torch(x)



class HighDimNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_in = nn.Linear(dim, 128) 
        self.block1 = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.block2 = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.fc_out = nn.Linear(128, 1)
        self.act = nn.Tanh()
        
    def forward(self, x):
        out = self.act(self.fc_in(x))
        out = out + self.block1(out)
        out = out + self.block2(out)
        return self.fc_out(out)



if __name__ == "__main__":
    print(f"--- Solving {DIM}-Dimensional Poisson using Deep Ritz ---")
    model = HighDimNet(DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    error_history = []
    
    start_time = time.time()
    for it in range(N_ITER + 1):
        optimizer.zero_grad()
        
        # 1. 域内采样
        x_domain = torch.rand(BATCH_SIZE, DIM, requires_grad=True, device=DEVICE)
        u_domain = model(x_domain)
        grads = torch.autograd.grad(u_domain, x_domain, torch.ones_like(u_domain), create_graph=True)[0]
        grad_sq = torch.sum(grads**2, dim=1)
        f_val = source_f_torch(x_domain)
        energy_loss = torch.mean(0.5 * grad_sq - f_val * u_domain.squeeze())
        # 2. 边界采样
        x_bd = torch.rand(BATCH_SIZE // 2, DIM, device=DEVICE)
        random_dims = torch.randint(0, DIM, (BATCH_SIZE // 2,))
        random_vals = torch.randint(0, 2, (BATCH_SIZE // 2,)).float().to(DEVICE)
        for i in range(BATCH_SIZE // 2):
            x_bd[i, random_dims[i]] = random_vals[i]
            
        u_bd = model(x_bd)
        target_bd = exact_u_torch(x_bd)
        bd_loss = torch.mean((u_bd - target_bd)**2)
        loss = energy_loss + 1000 * bd_loss
        loss.backward()
        optimizer.step()
        
        if it % 50 == 0:
            with torch.no_grad():
                x_test = torch.rand(3000, DIM, device=DEVICE)
                u_pred_test = model(x_test)
                u_true_test = exact_u_torch(x_test)
                l2_err = torch.norm(u_pred_test - u_true_test) / torch.norm(u_true_test)
                error_history.append(l2_err.item())
                
        if it % 500 == 0:
            print(f"Iter {it}: L2 Error = {l2_err.item():.2%}")

    print(f"Training finished in {time.time()-start_time:.2f}s")


    grid_n = 50
    x = np.linspace(0, 1, grid_n)
    y = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(x, y)
    
    input_tensor = np.zeros((grid_n * grid_n, DIM))
    input_tensor[:, 0] = X.flatten()
    input_tensor[:, 1] = Y.flatten()
    input_tensor[:, 2:] = 0.5 # 固定其他8个维度在中心
    
    input_torch = torch.tensor(input_tensor, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        u_exact_val = exact_u_torch(input_torch).cpu().numpy().reshape(grid_n, grid_n)
        u_pred_val = model(input_torch).cpu().numpy().reshape(grid_n, grid_n)


    fig = plt.figure(figsize=(18, 5))

    # 1. Exact Solution Slice
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(X, Y, u_exact_val, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title("Exact Solution (Slice at $x_{3..10}=0.5$)", fontsize=14)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlim(np.min(u_exact_val)-1, np.max(u_exact_val)+1) # 自动调整Z轴
    ax1.view_init(elev=30, azim=-45)

    # 2. DRM Prediction Slice
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, u_pred_val, cmap='plasma', edgecolor='none', alpha=0.9)
    ax2.set_title(f"Deep Ritz Method (Dim={DIM})", fontsize=14)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlim(np.min(u_exact_val)-1, np.max(u_exact_val)+1)
    ax2.view_init(elev=30, azim=-45)

    # 3. Error Convergence
    ax3 = fig.add_subplot(1, 3, 3)
    iters = np.arange(0, len(error_history) * 50, 50)
    ax3.semilogy(iters, error_history, 'r-', linewidth=2)
    ax3.set_title("Relative $L_2$ Error Convergence", fontsize=14)
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Relative Error (Log Scale)")
    ax3.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("high_dim_analysis.pdf", format='pdf', bbox_inches='tight')
    print("Saved high_dim_analysis.pdf")
    plt.show()