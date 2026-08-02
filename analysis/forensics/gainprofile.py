import numpy as np
for c in (1,2,3):
    for tag in ("kinv","kfwd"):
        k = np.load(f"/tmp/{tag}_case{c}.npy")
        K = k.shape[0]; N = 256
        H = np.abs(np.fft.fftshift(np.fft.fft2(k, (N,N))))
        fy = np.fft.fftshift(np.fft.fftfreq(N)); fx = fy
        R = np.hypot(*np.meshgrid(fy,fx,indexing='ij'))
        out=[]
        for lo,hi in [(0,.01),(.01,.03),(.03,.06),(.06,.10),(.10,.15),(.15,.25),(.25,.40)]:
            m=(R>=lo)&(R<hi); out.append(H[m].mean())
        print(f"case{c} {tag}  DC={k.sum():6.3f} | " + " ".join(f"{v:5.3f}" for v in out))
    print()
print("频带(cyc/px): <.01  .01-.03  .03-.06  .06-.10  .10-.15  .15-.25  .25-.40")
