import json, numpy as np
d = json.load(open("/home/ubuntu/workspace/contest/CSIG-2026/Moebius/analysis_tone/t1.json"))
for name, rec in d.items():
    print(f"===== {name}  shape={rec['shape']}  "
          f"lq_rgb_allzero={rec['lq_rgb_allzero_frac']*100:.3f}%  "
          f"lq_any_zero={rec['lq_any_zero_frac']*100:.3f}%")
    if "gt_at_lq_rgb0" in rec:
        g = rec["gt_at_lq_rgb0"]
        print(f"   GT@LQ全零: mean={np.round(g['mean'],2)} med={g['med']} p90={g['p90']} max={g['max']} "
              f"frac(GT任一通道>=8)={g['frac_gt_gte8']*100:.2f}%")
        print(f"   黑像素4邻域黑数 实测={rec['black_nbr_mean']:.3f} 随机期望={rec['black_nbr_expect_random']:.4f}")
    for cn in "RGB":
        c = rec[cn]
        print(f"  [{cn}] lq0={c['lq_zero_frac']*100:7.4f}% gt0={c['gt_zero_frac']*100:7.4f}% | "
              f"lq255={c['lq_255_frac']*100:7.4f}% gt255={c['gt_255_frac']*100:7.4f}% | "
              f"mean {c['lq_mean']:6.2f}->{c['gt_mean']:6.2f} std {c['lq_std']:6.2f}->{c['gt_std']:6.2f} | "
              f"affine a={c['affine'][0]:.4f} b={c['affine'][1]:+.3f}")
        print(f"       p01 {c['lq_p01']:.0f}->{c['gt_p01']:.0f}  p99 {c['lq_p99']:.0f}->{c['gt_p99']:.0f} | "
              f"GT@lq==0: mean={c['gt_at_lq0_mean']} med={c['gt_at_lq0_med']} p90={c['gt_at_lq0_p90']} max={c['gt_at_lq0_max']}"
              f" | GT@lq==255: mean={c['gt_at_lq255_mean']} min={c['gt_at_lq255_min']}")
    # 传递曲线（中位数）打印关键点
    print("   传递曲线 median(GT|LQ=v) - v :")
    for cn in "RGB":
        cur = rec[cn]["curve"]
        row = []
        for v in [0,1,2,4,8,12,16,24,32,48,64,96,128,160,192,224,240,248,252,254,255]:
            e = cur[v]
            row.append(f"{v}:{'--' if e[3] is None else f'{e[3]-v:+.1f}'}")
        print(f"    [{cn}] " + " ".join(row))
    print("   传递曲线 样本数（log10）:")
    cur = rec["G"]["curve"]
    print("    [G] " + " ".join(f"{v}:{np.log10(max(cur[v][1],1)):.1f}" for v in [0,1,2,4,8,16,32,64,128,192,240,254,255]))
