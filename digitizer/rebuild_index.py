import json, glob, os
import numpy as np
out=r"F:\nds\output\unet_data"
index=[]; tot={"train":0,"val":0}
for split in ("train","val"):
    for fn in sorted(glob.glob(os.path.join(out,split,"*.npz"))):
        try:
            with np.load(fn, mmap_mode="r") as z:
                n=z["imgs"].shape[0]
        except Exception as e:
            print("skip",os.path.basename(fn),e); continue
        well=os.path.basename(fn).split("__")[0]
        index.append({"npz":fn,"split":split,"well":well,"n_tiles":int(n)})
        tot[split]+=int(n)
json.dump(index, open(os.path.join(out,"index.json"),"w",encoding="utf-8"), ensure_ascii=False)
print("rebuilt index.json: files",len(index),"| tiles train",tot["train"],"val",tot["val"])
gb=(tot["train"]+tot["val"])*256*256*3/1e9
print(f"approx RAM for all tiles (uint8 img): {gb:.1f} GB (+masks ~{gb/3:.1f} GB)")
print("wells:", len({r["well"] for r in index}))
