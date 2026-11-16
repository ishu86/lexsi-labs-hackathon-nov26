"""usage: ak_sig.py A.json B.json  (each {item_id: label}, 1=refused). Delta = B - A, paired by id."""
import sys, json, random, math
import auditkit as ak
A, B = (json.load(open(f)) for f in sys.argv[1:3])
ids = sorted(A.keys() & B.keys())
runs = []
for lab in (A, B):
    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="refusal")
    def refusal(sample, output, lab=lab): return float(lab[sample.id])
    samples = [ak.Sample(input="x", id=i, actual_output="x") for i in ids]
    runs.append(ak.evaluate(samples, model="precomputed", scorers=[refusal], cache=False))
r = ak.compare(runs[0], runs[1]).significance("refusal", confidence=0.95)
print("AuditKIT", {k: r[k] for k in ("n", "mean_baseline", "mean_candidate", "delta", "p_value", "ci")})
# cross-check
a, b = [A[i] for i in ids], [B[i] for i in ids]; n = len(ids); d = [y - x for x, y in zip(a, b)]
rng = random.Random(0); bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
ci = (bs[int(.025 * 10000)], bs[int(.975 * 10000) - 1])
b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1); b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
k = min(b01, b10); m = b01 + b10
mc = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
print("manual  ", dict(n=n, mean_a=sum(a) / n, mean_b=sum(b) / n, delta=sum(d) / n, boot_ci=ci, mcnemar_exact_p=mc, discordant=(b01, b10)))
