import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=json.load(open('results/sensory_capacity_tradeoff.json'))['results']
sens=sorted(set(c['sensory_dim'] for c in d)); emb=sorted(set(c['embedding_dim'] for c in d))
SA=np.full((len(sens),len(emb)),np.nan)
for i,s in enumerate(sens):
  for j,e in enumerate(emb):
    v=[abs(c['SA']) for c in d if c['sensory_dim']==s and c['embedding_dim']==e]
    if v: SA[i,j]=np.mean(v)
ii,jj=np.meshgrid(range(len(sens)),range(len(emb)),indexing='ij')
def corr(a,b):
  a=np.array(a,float).flatten(); b=np.array(b,float).flatten(); m=~np.isnan(a)
  return np.corrcoef(a[m],b[m])[0,1]
c_s=corr(SA,ii); c_e=corr(SA,jj); c_m=corr(SA,np.minimum(ii,jj))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.4))
im=ax1.imshow(SA,origin='lower',cmap='viridis',aspect='auto')
ax1.set_xticks(range(len(emb))); ax1.set_xticklabels(emb)
ax1.set_yticks(range(len(sens))); ax1.set_yticklabels(sens)
ax1.set_xlabel('embedding capacity (internal memory)')
ax1.set_ylabel('sensory bandwidth')
ax1.set_title('Control alignment |SA| over the capacity grid\n(peak at the balanced high-capacity cell; over-\nprovisioning one axis past the other adds little)')
# overlay min-cut iso lines (min(rank) constant -> L-shaped)
for k in range(len(sens)):
    xs=[k+0.5]*2; # vertical+horizontal L corner at (k,k)
# draw L-corners for min-cut contours
for k in range(1,len(sens)):
    ax1.plot([k-0.5,k-0.5,len(emb)-0.5],[len(sens)-0.5,k-0.5,k-0.5],color='white',lw=1.2,alpha=0.7,ls='--')
fig.colorbar(im,ax=ax1,fraction=0.046,pad=0.04,label='|SA|  (control alignment)')

bars=ax2.bar(['sensory\nonly','memory\nonly','min(sensory,\nmemory)'],[c_s,c_e,c_m],
             color=['#bbbbbb','#bbbbbb','#2c7fb8'])
ax2.set_ylabel('correlation with control alignment |SA|')
ax2.set_title('The bottleneck is the *tighter* of the two\n(min-cut beats either axis alone)')
ax2.set_ylim(0,0.8)
for b,v in zip(bars,[c_s,c_e,c_m]):
    ax2.text(b.get_x()+b.get_width()/2,v+0.02,f'{v:.2f}',ha='center',fontweight='bold')
ax2.axhline(0,color='k',lw=0.5)
fig.suptitle('Cheap test of the min-cut reframe: existing 100-cell data, no new compute',fontsize=12,fontweight='bold')
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig('figures/mincut_first_test.png',dpi=150,bbox_inches='tight')
print('saved figures/mincut_first_test.png')
print(f'corrs: sensory={c_s:.3f} memory={c_e:.3f} min={c_m:.3f}')
