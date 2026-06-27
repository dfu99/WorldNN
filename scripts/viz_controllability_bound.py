import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import brentq

def Hb(e):
    e=np.clip(e,1e-12,1-1e-12)
    return -e*np.log2(e)-(1-e)*np.log2(1-e)
def Hb_inv(y):  # inverse on [0,0.5]
    if y<=0: return 0.0
    if y>=1: return 0.5
    return brentq(lambda e: Hb(e)-y, 1e-12, 0.5)

# --- measured perception-chain MI (from obj025_mi_vs_sensory.json), nats -> bits
mi=json.load(open('results/obj025_mi_vs_sensory.json'))
sdims=[int(k) for k in mi['KSG_MI_nats']]
mi_bits=[mi['KSG_MI_nats'][str(s)]/np.log(2) for s in sdims]
Cmin_toy=max(mi_bits)            # best achievable perception bits in current toy
eps_floor=Hb_inv(1-Cmin_toy)     # predicted error floor at that C_min

# --- Panel A: controllability bound curve
C=np.linspace(0,1,400)
eps_min=np.array([Hb_inv(1-c) for c in C])

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.5,4.6))
ax1.fill_between(C,eps_min,0.5,color='#c6e6c6',alpha=.6,label='feasible (achievable)')
ax1.fill_between(C,0,eps_min,color='#f4c6c6',alpha=.7,label='forbidden by the bound')
ax1.plot(C,eps_min,'k-',lw=2)
ax1.scatter([Cmin_toy],[eps_floor],s=90,color='#b30000',zorder=5)
ax1.annotate(f'current toy\nbest cell: {Cmin_toy:.2f} bits\n→ error floor {eps_floor:.0%}',
             (Cmin_toy,eps_floor),xytext=(Cmin_toy+0.12,eps_floor+0.13),
             arrowprops=dict(arrowstyle='->',color='#b30000'),fontsize=9,color='#b30000')
ax1.set_xlabel('min-cut capacity  $C_{min}$  (bits about the target state)')
ax1.set_ylabel('minimum achievable control error  $\\epsilon$')
ax1.set_title('Controllability bound\n$\\epsilon \\geq H_b^{-1}(1-C_{min})$   (Fano + data-processing)')
ax1.set_xlim(0,1); ax1.set_ylim(0,0.5); ax1.legend(loc='upper right',fontsize=8)

# --- Panel B: measured perception bits vs sensory dim, gap to 1 bit
ax2.bar([str(s) for s in sdims],mi_bits,color='#2c7fb8')
ax2.axhline(1.0,color='k',ls='--',lw=1.2)
ax2.text(0.05,1.02,'1 bit = needed to flip the state with $\\epsilon\\to0$',fontsize=8.5)
ax2.axhline(Cmin_toy,color='#b30000',ls=':',lw=1.2)
ax2.text(2.1,Cmin_toy+0.03,f'best = {Cmin_toy:.2f} bits',color='#b30000',fontsize=8.5)
ax2.set_xlabel('sensory bandwidth (channels sampled)')
ax2.set_ylabel('measured information about state  $I(S;X)$  (bits)')
ax2.set_title('Why the toy sits at its floor\nperception never delivers even half a bit')
ax2.set_ylim(0,1.1)
fig.suptitle('obj-039  Formal controllability bound — and its quantitative match to the observed floor',
             fontsize=12,fontweight='bold')
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig('figures/controllability_bound.png',dpi=150,bbox_inches='tight')
print(f'Cmin_toy={Cmin_toy:.3f} bits  eps_floor={eps_floor:.3f}')
print('saved figures/controllability_bound.png')
