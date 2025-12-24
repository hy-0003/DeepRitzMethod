import numpy as np
import scipy.sparse as sp
import sys

def run_high_dim_fem_challenge(dim=10, grid_points=10, memory_limit_gb=32.0):
    """
    尝试构建一个 D 维 Poisson 方程的线性系统 A u = b
    """
    print(f"==================================================")
    print(f"挑战任务: 使用 FEM/FDM 求解 {dim} 维 Poisson 方程")
    print(f"每维网格点数: {grid_points}")
    print(f"==================================================")

    # 1. 计算总自由度 (Degrees of Freedom)
    # N = n^d
    try:
        total_dof = grid_points ** dim
    except OverflowError:
        total_dof = float('inf')
    
    print(f"总未知数数量 (DOF): {grid_points}^{dim} = {total_dof:e}")

    # 2. 估算所需内存
    # 只需要存一个解向量 u (float64 = 8 bytes)
    # 即使不存矩阵，光存结果向量需要的内存：
    vector_memory_bytes = total_dof * 8 
    vector_memory_gb = vector_memory_bytes / (1024**3)
    
    # 估算稀疏矩阵 A 的内存 (CSR 格式)
    # 在 D 维拉普拉斯算子中，每一行大约有 2*D + 1 个非零元素
    # 每个非零元素需要: 8 bytes (value) + 4 bytes (col_index) = 12 bytes
    # 另外需要 row_ptr 数组 (忽略不计)
    nnz_per_row = 2 * dim + 1
    matrix_memory_bytes = total_dof * nnz_per_row * 12
    matrix_memory_gb = matrix_memory_bytes / (1024**3)

    total_memory_gb = vector_memory_gb + matrix_memory_gb

    print(f"\n--- 内存需求预估 ---")
    print(f"解向量 u 需要内存: {vector_memory_gb:.2f} GB")
    print(f"稀疏矩阵 A 需要内存: {matrix_memory_gb:.2f} GB")
    print(f"总计最低内存需求: {total_memory_gb:.2f} GB")
    print(f"你的内存限制: {memory_limit_gb:.2f} GB")

    # 3. 熔断保护
    if total_memory_gb > memory_limit_gb:
        print("\n[错误] 内存需求过大！停止运行以保护系统。")
        print("原因: 维数灾难 (Curse of Dimensionality)。")
        print("传统网格方法无法处理 d > 3 或 d > 4 的问题。")
        return False

    # 4. 如果你设的网格很小（比如每维2个点），代码会尝试真的去跑
    print("\n内存检查通过！正在尝试构建矩阵（这可能需要一些时间）...")
    
    try:
        # 构建 1D 拉普拉斯算子 Dxx
        h = 1.0 / (grid_points - 1)
        data = [np.ones(grid_points), -2*np.ones(grid_points), np.ones(grid_points)]
        offsets = [-1, 0, 1]
        D1 = sp.diags(data, offsets, shape=(grid_points, grid_points))
        D1 = -1 * D1 / (h**2) # 1D 负拉普拉斯

        # 利用 Kronecker 积构建 ND 拉普拉斯
        # A_nd = A_1d (x) I (x) ... + I (x) A_1d (x) ...
        # 这是一个递归过程
        A = D1
        I = sp.eye(grid_points)
        
        # 循环构建高维矩阵
        for i in range(1, dim):
            # A_new = A_old (kron) I + I_old (kron) D1
            # 注意维度变化
            current_size = grid_points ** i
            I_current = sp.eye(current_size)
            
            # 这一步在维数高时会极慢且爆内存
            A = sp.kron(A, I) + sp.kron(I_current, D1)
            print(f"  ...已构建 {i+1} 维矩阵")

        print("矩阵构建完成！")
        return True

    except MemoryError:
        print("\n[致命错误] MemoryError: 真实分配内存失败！")
        return False
    except Exception as e:
        print(f"\n[错误] {e}")
        return False

if __name__ == "__main__":
    # --- 实验 1: 稍微正常一点的维度 (比如 4维, 每维10个点) ---
    print("\n>>> 尝试运行 4维 FEM (作为对比)...")
    run_high_dim_fem_challenge(dim=4, grid_points=10)

    # --- 实验 2: 真正的 10维 挑战 ---
    print("\n" + "="*60)
    print(">>> 尝试运行 10维 FEM (Deep Ritz 的主场)...")
    print("="*60)
    
    # 即使每维只取 5 个点，10维也是天文数字
    run_high_dim_fem_challenge(dim=10, grid_points=10)