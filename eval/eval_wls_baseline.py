import os, numpy as np

def angle_rmse_rad(a, b):
    d = (a - b + np.pi) % (2*np.pi) - np.pi
    return np.sqrt(np.mean(d**2))

def main(path):
    D = np.load(path)
    x_true = D["x"]       # [Nwin, W, 2N]
    x_wls  = D["x_wls"]   # [Nwin, W, 2N]
    Nwin, W, Ddim = x_true.shape
    nbus = Ddim // 2

    th_true = x_true[..., :nbus]
    vm_true = x_true[..., nbus:]
    th_wls  = x_wls[...,  :nbus]
    vm_wls  = x_wls[...,  nbus:]

    th_rmse_rad = angle_rmse_rad(th_wls, th_true)
    vm_rmse     = np.sqrt(np.mean((vm_wls - vm_true)**2))

    print(f"[{os.path.basename(path)}] θ-RMSE = {np.degrees(th_rmse_rad):.3f} deg | |V|-RMSE = {vm_rmse:.4f} p.u.")

if __name__ == "__main__":
    base = "data/windows_ieee33"
    for tag in ["W24","W96"]:
        for part in ["train","val","test"]:
            p = os.path.join(base, f"{tag}_{part}.npz")
            if os.path.exists(p):
                main(p)
