"""eval_p0 variant that dumps per-item scores, so two configs can be compared with a paired test."""
import argparse, json, os, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, "/root/.cache/huggingface/csig/code/tools")
sys.path.insert(0, "/root/.cache/huggingface/csig/code/HYPIR")
import iqa
from HYPIR.enhancer.sd2 import SD2Enhancer

CSIG = "/root/.cache/huggingface/csig"
LORA_MODULES = ["to_k","to_q","to_v","to_out.0","conv","conv1","conv2","conv_shortcut","conv_out","proj_in","proj_out","ff.net.2","ff.net.0.proj"]
PROMPT = ("A high-resolution photograph with fine natural detail: modern Chinese city "
          "high-rise buildings with glass curtain walls and tiled facades, construction "
          "cranes and shop signage, distant hills, and close-up green foliage with flowers.")

def load(p):
    return torch.from_numpy(cv2.imread(p)).cuda().flip(-1).permute(2,0,1)[None].float()/255.0

ap = argparse.ArgumentParser()
ap.add_argument("weight_path"); ap.add_argument("--tag", required=True)
a = ap.parse_args()
ed = os.path.join(CSIG, "data/eval_p0")
meta = json.load(open(os.path.join(ed, "meta.json")))["items"]
en = SD2Enhancer(base_model_path=os.path.join(CSIG,"weights"), weight_path=a.weight_path,
                 lora_modules=LORA_MODULES, lora_rank=256, model_t=200, coeff_t=200, device="cuda")
en.init_models()
t0 = time.time(); items = []
for i, it in enumerate(meta):
    gt = load(os.path.join(ed,"gt",it["name"])); lq = load(os.path.join(ed,"lq",it["name"]))
    with torch.no_grad():
        out = en.enhance(lq, prompt=PROMPT, upscale=1, patch_size=512, stride=256, return_type="pt").clamp(0,1)
    _,_,p_o = iqa.score(out*2-1, gt*2-1)
    _,_,p_l = iqa.score(lq*2-1, gt*2-1)
    _,_,p_g = iqa.score(gt*2-1, gt*2-1)
    items.append(dict(name=it["name"], cat=it["cat"], p_out=float(p_o), p_lq=float(p_l), p_gt=float(p_g)))
el = time.time()-t0
v = np.array([[x["p_out"],x["p_lq"],x["p_gt"]] for x in items])
ret = (v[:,0].mean()-v[:,1].mean())/(v[:,2].mean()-v[:,1].mean())
print("tag=%s retained=%.4f p_out=%.4f  elapsed=%.1fs (%.2fs/item)" % (a.tag, ret, v[:,0].mean(), el, el/len(meta)))
json.dump(dict(tag=a.tag, weight=a.weight_path, retained=float(ret), items=items),
          open("/tmp/evaldump_%s.json" % a.tag, "w"))
