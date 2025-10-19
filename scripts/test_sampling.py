import os, sys, torch, random
sys.path.append('src')
from models.meta_controller import MetaController

def main():
    torch.manual_seed(0); random.seed(0)
    task_feat_dim, layer_feat_dim, num_layers = 3, 2, 8
    mc = MetaController(task_feat_dim, layer_feat_dim, 64,
                        ranks=[1,2,4,8], bitwidths=[2,4,8], sparsities=[0.0,0.5],
                        num_layers=num_layers)
    task_f = torch.randn(1, task_feat_dim)
    layer_f = torch.randn(1, num_layers, layer_feat_dim)
    r_p, b_p, s_p = mc(task_f, layer_f)

    # Softmax with temperature
    temp = 3.0
    def temp_softmax(p): return torch.softmax(torch.log(p + 1e-9)/temp, dim=-1)
    r_p, b_p, s_p = map(temp_softmax, (r_p, b_p, s_p))

    from torch.distributions import Categorical
    samples = []
    for k in range(20):
        r_idx, b_idx, s_idx = [], [], []
        for i in range(num_layers):
            rd, bd, sd = Categorical(r_p[0,i]), Categorical(b_p[0,i]), Categorical(s_p[0,i])
            r_idx.append(int(rd.sample())); b_idx.append(int(bd.sample())); s_idx.append(int(sd.sample()))
        triplet = (tuple(r_idx[:4]), tuple(b_idx[:4]), tuple(s_idx[:4]))
        samples.append(triplet)
        print(f"Sample {k+1}: ranks={r_idx[:4]}, bits={b_idx[:4]}, sparsity={s_idx[:4]}")
    uniq = len(set(samples))
    print(f"Unique first-4-layer triplets in 20 samples: {uniq}")

if __name__ == "__main__":
    main()
