r"""Encrypt, embed, extract, decrypt with DEDS.

Hybrid encryption: a random AES-256 key encrypts the message with AES-GCM (12-byte nonce, 16-byte tag); the AES key
is wrapped with RSA-2048 OAEP (SHA-256). The package [RSA-wrapped key 256 B | nonce 12 B | ciphertext + tag] is the
payload. It is embedded in fine-tuned corpus covers with the paper's embedder (v5, margin 0.15, K forced from the
payload), the stego image is written to PNG and read back, the package is extracted, the AES key unwrapped with the
RSA private key, and the message decrypted and compared byte for byte.
Covers: corpus fine-tuned covers 0, 10, ..., 990 (100). Payloads: 0.30, 0.40, 0.60 bpp of package.
Tamper tests on 20 stego images at 0.40 bpp: (a) one pixel changed by +1 grey level, (b) one 8x8 block brightened
by 30 grey levels. Test keys are generated for this run only and are not saved.
Writes only inside F:\DEDS_secure_trials\crypto."""
import os, io, sys, csv, time, contextlib, importlib.util, concurrent.futures as cf
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from pathlib import Path
import numpy as np, cv2
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

HERE = Path(__file__).resolve().parent
COVERS = Path(r"F:\scirep results\09_expansion_1000\corpus1000\finetuned")
EMB = Path(r"F:\DEDS_project\embedder\DCT_Adaptive_v5.py")
TMP = HERE / "stego_png"; TMP.mkdir(exist_ok=True)
OAEP = padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
_D = None


class RawMessage(str):
    def __new__(cls, data):
        o = super().__new__(cls, "x" * len(data)); o.data = data; return o

    def encode(self, *a, **k):
        return self.data


def _init():
    global _D
    spec = importlib.util.spec_from_file_location("emb", EMB)
    _D = importlib.util.module_from_spec(spec); spec.loader.exec_module(_D); _D.PAYLOAD_MARGIN = 0.15


def extract_bytes(img):
    Y = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = _D._to_blocks(Y)
    n_bits, K, _, ok = _D._read_header(blocks)
    if not ok:
        return None
    nr, nc = blocks.shape[:2]
    bits = np.array(_D._parities(blocks, _D._slot_arrays(_D._build_slot_order(K, nr, nc)[:n_bits])), dtype=np.uint8)
    return np.packbits(bits).tobytes()


def embed_extract(task):
    """Embed a package, save PNG, reload, extract; plus the two tamper variants when asked."""
    i, bpp, package, tamper = task
    cover = cv2.imread(str(COVERS / f"{i:03d}.png"))
    n_bits = 8 * len(package)
    K = max(1, min(_D.MAX_K, int(np.ceil(n_bits / 1022)) - 1))
    orig = _D._compute_K; _D._compute_K = lambda b, K=K: K
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            stego = _D._embed_normal(cover, RawMessage(package))
    finally:
        _D._compute_K = orig
    t_embed = time.perf_counter() - t0
    path = TMP / f"{bpp:.2f}_{i:03d}.png"
    cv2.imwrite(str(path), stego)
    back = cv2.imread(str(path))
    yc = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    ys = cv2.cvtColor(back, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(float)
    psnr = 10 * np.log10(255 ** 2 / ((ys - yc) ** 2).mean())
    out = {"i": i, "bpp": bpp, "K": K, "extracted": extract_bytes(back), "psnr_y": psnr, "embed_s": t_embed}
    if tamper:
        a = back.copy(); a[100, 100] = np.clip(a[100, 100].astype(int) + 1, 0, 255).astype(np.uint8)
        b = back.copy(); b[96:104, 96:104] = np.clip(b[96:104, 96:104].astype(int) + 30, 0, 255).astype(np.uint8)
        out["tamper_pixel"] = extract_bytes(a); out["tamper_block"] = extract_bytes(b)
    return out


def main():
    rng = np.random.default_rng(2026)
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048); pub = priv.public_key()
    tasks, truth = [], {}
    for bpp in (0.30, 0.40, 0.60):
        total = int(bpp * 65536) // 8                     # package bytes
        n_plain = total - 256 - 12 - 16
        for i in range(0, 1000, 10):
            msg = rng.integers(0, 256, n_plain, dtype=np.uint8).tobytes()
            key = AESGCM.generate_key(bit_length=256); nonce = os.urandom(12)
            package = pub.encrypt(key, OAEP) + nonce + AESGCM(key).encrypt(nonce, msg, None)
            assert len(package) == total
            truth[(i, bpp)] = msg
            tasks.append((i, bpp, package, bpp == 0.40 and i < 200))

    def decrypt(pkg):
        if pkg is None:
            return None, "header failed"
        try:
            key = priv.decrypt(pkg[:256], OAEP)
            return AESGCM(key).decrypt(pkg[256:268], pkg[268:], None), "ok"
        except (InvalidTag, ValueError) as e:
            return None, type(e).__name__

    rows = []; t0 = time.time()
    with cf.ProcessPoolExecutor(6, initializer=_init) as ex:
        for k, r in enumerate(ex.map(embed_extract, tasks, chunksize=2)):
            msg, status = decrypt(r["extracted"])
            row = {"cover": r["i"], "bpp": r["bpp"], "K": r["K"], "package_bytes": len(tasks[k][2]),
                   "plaintext_bytes": len(truth[(r["i"], r["bpp"])]), "psnr_y": round(r["psnr_y"], 2),
                   "embed_s": round(r["embed_s"], 2), "decrypt": status,
                   "message_identical": msg == truth[(r["i"], r["bpp"])]}
            if "tamper_pixel" in r:
                for t in ("tamper_pixel", "tamper_block"):
                    m2, st2 = decrypt(r[t])
                    row[t] = "identical" if m2 == truth[(r["i"], r["bpp"])] else st2
            rows.append(row)
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
    fields = list(dict.fromkeys(f for r in rows for f in r))
    with open(HERE / "crypto_roundtrip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    print("\npayload  covers  plaintext  decrypted+identical  mean PSNR_Y")
    for bpp in (0.30, 0.40, 0.60):
        g = [r for r in rows if r["bpp"] == bpp]
        print(f"{bpp:.2f} bpp  {len(g)}  {g[0]['plaintext_bytes']} B  {sum(r['message_identical'] for r in g)}/{len(g)}"
              f"  {np.mean([r['psnr_y'] for r in g]):.2f} dB")
    tp = [r for r in rows if "tamper_pixel" in r]
    for t in ("tamper_pixel", "tamper_block"):
        vals = [r[t] for r in tp]
        print(t, {v: vals.count(v) for v in set(vals)})


if __name__ == "__main__":
    main()
