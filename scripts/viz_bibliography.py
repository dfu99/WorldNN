"""Visualize the bibliography expansion: 13 → 43 entries by category."""
import matplotlib.pyplot as plt
import numpy as np

categories = {
    "POMDP / RL theory":      ["kaelbling1998planning", "hausknecht2015deep", "igl2018deep",
                                "schulman2017ppo", "schulman2016gae"],
    "World models":           ["ha2018world", "hafner2019planet", "hafner2023mastering",
                                "hafner2025dreamer4", "schrittwieser2020muzero",
                                "micheli2023transformers"],
    "Rep. learning for RL":   ["laskin2020curl", "schwarzer2021data", "stooke2021decoupling",
                                "yarats2021image", "bengio2013representation", "anand2019unsupervised",
                                "wang2023quasimetric"],
    "VAE / info-theory":      ["kingma2014autoencoding", "higgins2017beta", "tishby2000information",
                                "cover2006elements", "kraskov2004estimating", "belghazi2018mine",
                                "poole2019variational"],
    "Sensorimotor / embod.":  ["oregan2001sensorimotor", "smith2005development",
                                "pfeifer2006body", "li2006towards", "bajcsy1988active",
                                "aloimonos1988active", "actjepa2025", "vjepa2_2025"],
    "Neuroscience":           ["blakemore1970development", "hubel1970period", "huang1999bdnf",
                                "rossi1999monocular", "achille2019critical"],
    "Asymmetric / curiosity": ["pinto2018asymmetric", "vapnik2009lupi", "pathak2017curiosity",
                                "tobin2017domain"],
    "Free energy":            ["friston2010free"],
}

pre_existing = {
    "blakemore1970development", "hubel1970period", "huang1999bdnf", "rossi1999monocular",
    "friston2010free", "tishby2000information", "pinto2018asymmetric", "hafner2023mastering",
    "hafner2025dreamer4", "wang2023quasimetric", "micheli2023transformers",
    "actjepa2025", "vjepa2_2025",
}

labels = list(categories.keys())
pre_counts  = [sum(1 for k in v if k in pre_existing) for v in categories.values()]
new_counts  = [sum(1 for k in v if k not in pre_existing) for v in categories.values()]
totals      = [p + n for p, n in zip(pre_counts, new_counts)]

assert sum(totals) == 43, f"expected 43 total, got {sum(totals)}"
assert sum(pre_counts) == 13, f"expected 13 pre-existing, got {sum(pre_counts)}"
assert sum(new_counts) == 30, f"expected 30 new, got {sum(new_counts)}"

order = np.argsort(totals)[::-1]
labels      = [labels[i] for i in order]
pre_counts  = [pre_counts[i] for i in order]
new_counts  = [new_counts[i] for i in order]

fig, ax = plt.subplots(figsize=(9, 5.2))
y = np.arange(len(labels))
ax.barh(y, pre_counts, color="#666", label=f"Pre-existing (n={sum(pre_counts)})")
ax.barh(y, new_counts, left=pre_counts, color="#1f77b4",
        label=f"Added in expansion (n={sum(new_counts)})")

for i, (p, n) in enumerate(zip(pre_counts, new_counts)):
    total = p + n
    ax.text(total + 0.15, i, str(total), va="center", fontsize=10)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Bibliography entries", fontsize=11)
ax.set_title("NeurIPS 2026 bibliography expansion: 13 → 43 entries", fontsize=12)
ax.legend(loc="lower right", fontsize=10, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, max(totals) + 2)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
out = "/home2/Documents/code/WorldNN/figures/obj037_bibliography_expansion.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved {out}")
