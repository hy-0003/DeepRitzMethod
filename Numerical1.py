import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


GRID_SIZE = 60          
N_ITER = 10000           # DRM 迭代次数
BATCH_SIZE = 1024        # DRM 采样点
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


np.random.seed(42)
torch.manual_seed(42)


def exact_u(x, y):
    return np.sin(3 * np.pi * x) * np.sin(3 * np.pi * y)

def source_f_numpy(x, y):
    return 18 * (np.pi**2) * np.sin(3 * np.pi * x) * np.sin(3 * np.pi * y)

def source_f_torch(x):
    return 18 * (np.pi**2) * torch.sin(3 * np.pi * x[:, 0]) * torch.sin(3 * np.pi * x[:, 1])



class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()
    def forward(self, x):
        res = x
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        return out + res

class DeepRitzNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(2, 20)
        self.blocks = nn.Sequential(ResBlock(20), ResBlock(20), ResBlock(20)) 
        self.fc_out = nn.Linear(20, 1)
    def forward(self, x):
        x = torch.tanh(self.fc_in(x))
        x = self.blocks(x)
        return self.fc_out(x)

def train_drm(beta=500):
    print(f"--- Training DRM with Beta = {beta} ---")
    model = DeepRitzNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()
    for it in range(N_ITER + 1):
        # 1. Domain Sampling
        x_domain = torch.rand(BATCH_SIZE, 2, requires_grad=True, device=DEVICE)
        u_domain = model(x_domain)
        
        grads = torch.autograd.grad(u_domain, x_domain, torch.ones_like(u_domain), create_graph=True)[0]
        grad_u_sq = torch.sum(grads**2, dim=1)
        f_val = source_f_torch(x_domain)
        
        # Energy Loss
        energy_loss = torch.mean(0.5 * grad_u_sq - f_val * u_domain.squeeze())
        
        # 2. Boundary Sampling
        x_bd = torch.rand(BATCH_SIZE // 4, 2, device=DEVICE)
        x_bd[:BATCH_SIZE//16, 0] = 0.0 # Left
        x_bd[BATCH_SIZE//16:2*BATCH_SIZE//16, 0] = 1.0 # Right
        x_bd[2*BATCH_SIZE//16:3*BATCH_SIZE//16, 1] = 0.0 # Bottom
        x_bd[3*BATCH_SIZE//16:, 1] = 1.0 # Top
        
        u_bd = model(x_bd)
        bd_loss = torch.mean(u_bd**2)
        
        # Total Loss
        loss = energy_loss + beta * bd_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    print(f"Finished in {time.time()-start_time:.2f}s")
    return model


def solve_fem(n):
    h = 1.0 / (n - 1)
    
    # 1D Laplace Operator
    data = [np.ones(n), -2*np.ones(n), np.ones(n)]
    offsets = [-1, 0, 1]
    D1 = sp.diags(data, offsets, shape=(n, n))
    D1 = -1 * D1 / (h**2)
    
    I = sp.eye(n)
    A = sp.kron(I, D1) + sp.kron(D1, I)
    
    # Right hand side
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    b = source_f_numpy(X, Y).flatten()
    
    # Boundary Conditions
    mask = np.zeros((n, n), dtype=bool)
    mask[0, :] = True; mask[-1, :] = True
    mask[:, 0] = True; mask[:, -1] = True
    bd_indices = np.where(mask.flatten())[0]
    
    A = A.tolil()
    for idx in bd_indices:
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = 0
    A = A.tocsr()
    
    u_fem = spla.spsolve(A, b).reshape(n, n)
    return u_fem, X, Y


if __name__ == "__main__":
    u_fem, X, Y = solve_fem(GRID_SIZE)
    model_std = train_drm(beta=500)
    

    xy_tensor = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        u_drm = model_std(xy_tensor).cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
    u_exact = exact_u(X, Y)


    fig1 = plt.figure(figsize=(20, 4.5))
    
    # Subplot 1: Exact
    ax1 = fig1.add_subplot(1, 4, 1, projection='3d')
    ax1.plot_surface(X, Y, u_exact, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title('Exact Solution', fontsize=14)
    ax1.axis('off')
    
    # Subplot 2: FEM
    ax2 = fig1.add_subplot(1, 4, 2, projection='3d')
    ax2.plot_surface(X, Y, u_fem, cmap='inferno', edgecolor='none', alpha=0.9)
    ax2.set_title('FEM (Grid)', fontsize=14)
    ax2.axis('off')

    # Subplot 3: DRM
    ax3 = fig1.add_subplot(1, 4, 3, projection='3d')
    ax3.plot_surface(X, Y, u_drm, cmap='plasma', edgecolor='none', alpha=0.9)
    ax3.set_title('Deep Ritz (Beta=500)', fontsize=14)
    ax3.axis('off')

    # Subplot 4: 1D Cross-Section (y=0.5)
    ax4 = fig1.add_subplot(1, 4, 4)
    mid_idx = GRID_SIZE // 2
    x_line = X[mid_idx, :]
    
    ax4.plot(x_line, u_exact[mid_idx, :], 'k-', linewidth=2, label='Exact')
    ax4.plot(x_line, u_fem[mid_idx, :], 'b--', linewidth=2, label='FEM')
    ax4.plot(x_line, u_drm[mid_idx, :], 'r:', linewidth=2, label='DRM')
    
    ax4.set_title('Cross-Section at y=0.5', fontsize=14)
    ax4.set_xlabel('x')
    ax4.set_ylabel('u(x, 0.5)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    

    print("\n=== Experiment 2: Beta Sensitivity ===")
    
    betas = [100, 500, 5000]
    models = []
    
    for b in betas:
        models.append(train_drm(beta=b))
        
    fig2 = plt.figure(figsize=(18, 5))
    for i, model in enumerate(models):
        with torch.no_grad():
            u_pred = model(xy_tensor).cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
            
        ax = fig2.add_subplot(1, 3, i+1, projection='3d')
        surf = ax.plot_surface(X, Y, u_pred, cmap='viridis', edgecolor='none', alpha=0.9)
        
        beta_val = betas[i]
        title_str = f'Beta = {beta_val}'
        
        ax.set_title(title_str, fontsize=14)
        ax.set_zlim(-1.2, 1.2) # 固定坐标轴以便对比
        ax.axis('off')

    plt.suptitle("Effect of Boundary Penalty Parameter (Beta)", fontsize=16)
    plt.tight_layout()
    plt.show()