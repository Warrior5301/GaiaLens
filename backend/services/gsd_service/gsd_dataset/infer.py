import argparse
import numpy as np
import torch
from PIL import Image

from features import extract, SIZE
from model import GSDNet


class GSDPredictor:
    def __init__(self, ckpt_path, device="cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.model = GSDNet(ck["n_in"]).to(device)
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.device = device

    def _tiles(self, gray):
        h, w = gray.shape
        if h < SIZE or w < SIZE:
            gray = np.asarray(Image.fromarray(gray).resize((SIZE, SIZE), Image.BICUBIC), dtype=np.float32)
            return [gray]
        return [gray[y:y + SIZE, x:x + SIZE]
                for y in range(0, h - SIZE + 1, SIZE) for x in range(0, w - SIZE + 1, SIZE)]

    @torch.no_grad()
    def predict(self, path):
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        feats = np.stack([extract(t) for t in self._tiles(gray)])
        x = torch.tensor((feats - self.mu) / self.sd, dtype=torch.float32, device=self.device)
        g = torch.exp(self.model(x)).cpu().numpy()
        return {"gsd_m": float(np.median(g)), "spread": float(g.std()), "n_tiles": len(g)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gsd_model.pt")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    p = GSDPredictor(a.model)
    for im in a.images:
        r = p.predict(im)
        print(f"{im}: {r['gsd_m']:.2f} m/pixel  (spread {r['spread']:.2f}, tiles {r['n_tiles']})")
