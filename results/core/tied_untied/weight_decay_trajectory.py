"""AdamW decay trajectory for the five-feature untied model, with biases exempt.

Unlike the norm experiments, selection here uses FINAL trained states, never an
early checkpoint with a better MSE before decay takes effect. AdamW is decoupled
weight shrinkage, not a claim to solve MSE plus an L2 penalty with this coefficient.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import one_sided_norm_experiments as base


HERE = Path(__file__).resolve().parent
OUT = HERE/'weight_decay_results'
DECAYS = [0.,1e-4,3e-4,.001,.003,.01,.03,.1,.2,.3,.5,.7,1.,1.5,2.,3.,5.,10.,30.,100.,300.,1000.]
MEAN = (1-base.S)/2
MEAN_MSE = (1-base.S)/3-MEAN**2
FAMILIES = ('pictured_untied','refined_untied','tied','random','mean_start','free_positive_bias')


def sources():
    saved = json.loads((HERE/'one_sided_norm_results/summary.json').read_text())
    previous = {k:{q:torch.tensor(saved['winners'][k][q]) for q in ('E','D','b')} for k in ('tied','free')}
    pictured=base.saved_models()['untied']
    previous['pictured'] = dict(zip(('E','D','b'),pictured))
    return previous


def make_starts(shard):
    source=sources();starts=[]
    for index,wd in enumerate(DECAYS):
        if index%2!=shard:continue
        for k,family in enumerate(FAMILIES):
            seed=9570000+index*100+k
            g=torch.Generator().manual_seed(seed)
            if family in ('pictured_untied','refined_untied','tied','free_positive_bias'):
                key={'pictured_untied':'pictured','refined_untied':'free','tied':'tied','free_positive_bias':'free'}[family]
                E,D,b=(source[key][q].clone() for q in ('E','D','b'))
                if family=='free_positive_bias':b.fill_(MEAN)
            else:
                scale=.02 if family=='mean_start' else .4
                E=scale*torch.randn(2,5,generator=g);D=scale*torch.randn(2,5,generator=g)
                b=torch.full((5,),MEAN)
            starts.append(dict(id=f'wd{wd:g}_{family}',family=family,seed=seed,weight_decay=wd,E=E,D=D,b=b))
    return starts


def shrink(model,learning_rate,decay):
    # AdamW applies weight shrinkage after computing gradients and before the
    # Adam parameter update. A per-model multiplier exactly implements separate
    # AdamW parameter groups with distinct coefficients, without touching bias.
    with torch.no_grad():
        factor=1-learning_rate*decay[:,None,None]
        if torch.any(factor<0):raise ValueError('Learning rate times decay must not exceed one')
        model.e.mul_(factor);model.d.mul_(factor)


def self_check():
    starts=make_starts(0)[:1]
    actual=base.Models(starts,'free','cartesian')
    reference=base.Models(starts,'free','cartesian')
    optimizer=torch.optim.Adam(actual.parameters(),lr=.001,foreach=False)
    ref_optimizer=torch.optim.AdamW([{'params':[reference.e,reference.d],'weight_decay':.7},
                                    {'params':[reference.b],'weight_decay':0.}],lr=.001,foreach=False)
    g=torch.Generator().manual_seed(1001)
    for _ in range(12):
        x=base.sample(97,g)
        optimizer.zero_grad(set_to_none=True);ref_optimizer.zero_grad(set_to_none=True)
        (actual(x)-x).square().mean().backward();(reference(x)-x).square().mean().backward()
        shrink(actual,.001,torch.tensor([.7]));optimizer.step();ref_optimizer.step()
    for a,b in zip(actual.parameters(),reference.parameters()):torch.testing.assert_close(a,b)
    old_bias=actual.b.detach().clone();shrink(actual,.001,torch.tensor([1000.]))
    torch.testing.assert_close(actual.b,old_bias,rtol=0,atol=0)
    torch.testing.assert_close(actual.e,torch.zeros_like(actual.e),rtol=0,atol=0)
    print('Per-model decay matches torch AdamW; bias is exactly exempt.',flush=True)


def train(starts,steps,batch,seed,path,validation,lr=.001):
    model=base.Models(starts,'free','cartesian')
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,foreach=False)
    decay=torch.tensor([a['weight_decay'] for a in starts])
    g=torch.Generator().manual_seed(seed)
    trace=[];began=time.monotonic()
    for step in range(1,steps+1):
        progress=max(0,(step/steps-.5)/.5)
        rate=lr*(.01+.99*(1+math.cos(math.pi*progress))/2)
        optimizer.param_groups[0]['lr']=rate
        x=base.sample(batch,g)
        optimizer.zero_grad(set_to_none=True)
        (model(x)-x).square().mean(dim=(1,2)).sum().backward()
        shrink(model,rate,decay);optimizer.step()
        if step%5000==0 or step==steps:
            with torch.no_grad():E,D,b=(v.detach().clone() for v in model.weights())
            score=base.evaluate(E,D,b,validation)
            if not torch.isfinite(score).all():raise RuntimeError('Nonfinite validation loss')
            trace.append(dict(step=step,lr=rate,elapsed_seconds=time.monotonic()-began,validation_mse=score.tolist()))
            runs=base.serialize(starts,'free','cartesian',E,D,b,score,torch.full((len(starts),),step))
            for run in runs:
                run['checkpoint_step']=run.pop('best_step')
                run['weight_squared_norm']=float(np.square(run['E']).sum()+np.square(run['D']).sum())
            payload=dict(steps=steps,completed_steps=step,batch_size=batch,training_seed=seed,
                         optimizer='AdamW (verified against torch.optim.AdamW)',learning_rate=lr,
                         schedule='flat first half, cosine cooldown to 1% thereafter',bias_weight_decay=0,
                         checkpoint_selection='Final state only; earlier checkpoints are diagnostics, not candidates',
                         trace=trace,runs=runs)
            base.save_json(path,payload)
            by_wd={wd:float(score[[a['weight_decay']==wd for a in starts]].min()) for wd in sorted(set(a['weight_decay'] for a in starts))}
            print(f'{path.stem}: {step}/{steps}, {time.monotonic()-began:.1f}s; '+
                  ', '.join(f'{k:g}: {v:.5f}' for k,v in by_wd.items()),flush=True)
    return payload


def groups(output,phases):
    result={}
    for phase in phases:
        for p in sorted(output.glob(f'{phase}_*.json')):
            d=json.loads(p.read_text())
            if d['completed_steps']!=d['steps']:raise RuntimeError(f'{p.name} still running')
            result[p.stem]=d
    return result


def chosen_by_decay(data):
    result={}
    for group,d in data.items():
        for a in d['runs']:
            wd=a['weight_decay']
            if wd not in result or a['validation_mse']<result[wd]['validation_mse']:
                result[wd]=dict(a,source_group=group)
    return result


def from_record(a,wd=None,family=None):
    decay=a['weight_decay'] if wd is None else wd
    return dict(id=f'wd{decay:g}_{family or a["id"]}',family=family or a['family'],seed=a['seed'],
                weight_decay=decay,source_decay=a['weight_decay'],source_id=a['id'],
                source_family=a['family'],
                **{q:torch.tensor(a[q]) for q in ('E','D','b')})


def continuation_starts(output):
    initial=groups(output,['explore'])
    if len(initial)!=2:raise RuntimeError('Finish both initial shards')
    best=chosen_by_decay(initial);starts=[]
    keys=sorted(best)
    for i,wd in enumerate(keys):
        for offset,label in ((-1,'lower_decay_warm'),(1,'higher_decay_warm')):
            j=i+offset
            if 0<=j<len(keys):starts.append(from_record(best[keys[j]],wd,label))
    return starts


def refine_starts(output):
    data=groups(output,['explore','continuation'])
    if len(data)!=4:raise RuntimeError('Finish both exploration and continuation shards first')
    best=chosen_by_decay(data);starts=[]
    for wd,a in sorted(best.items()):
        starts.append(from_record(a,family='longer_training'))
        # Preserve the original pictured model's branch for a direct trajectory.
        initial=[a for d in data.values() for a in d['runs'] if a['weight_decay']==wd and a['family']=='pictured_untied']
        pictured=min(initial,key=lambda a:a['validation_mse'])
        if pictured['id']!=a['id']:starts.append(from_record(pictured,family='pictured_longer_training'))
    reference=sources()['tied'];val=base.sample(131072,torch.Generator().manual_seed(957001))
    tied_loss=float(base.evaluate(reference['E'][None],reference['D'][None],reference['b'][None],val)[0])
    keys=sorted(best)
    cross_intervals=[]
    for index,(lo,hi) in enumerate(zip(keys,keys[1:])):
        if best[lo]['validation_mse']<tied_loss<=best[hi]['validation_mse']:
            cross_intervals.append((lo,hi))
            # Longer training and the larger batch may move the transition.
            if index+2<len(keys):cross_intervals.append((hi,keys[index+2]))
    for lo,hi in cross_intervals:
        middle=np.geomspace(lo,hi,4)[1:-1] if lo else np.linspace(lo,hi,4)[1:-1]
        for wd in middle:
            wd=float(f'{wd:.4g}')
            for source_wd in (lo,hi):starts.append(from_record(best[source_wd],wd,'crossover_warm_'+str(source_wd)))
    return starts


def make_report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    data=groups(output,['explore','continuation','refine','crossover'])
    if not all(f'refine_shard{i}' in data for i in range(2)):raise RuntimeError('Finish both longer-training shards before evaluating the test set')
    # Only the final states of the longest runs determine the main curve.
    final={name:d for name,d in data.items() if name.startswith(('refine_','crossover_'))}
    selected=chosen_by_decay(final)
    initial=chosen_by_decay({name:d for name,d in data.items() if name.startswith('explore_')})
    original=sources()
    points=[dict(a,weight_decay=wd) for wd,a in sorted(selected.items())]
    tied=dict(id='unregularized_tied',**{k:original['tied'][k].tolist() for k in ('E','D','b')})
    mean=dict(id='mean_predictor',E=np.zeros((2,5)).tolist(),D=np.zeros((2,5)).tolist(),b=[MEAN]*5)
    records=[tied,mean]+points
    seed,count=957999,1000000
    x=base.sample(count,torch.Generator().manual_seed(seed))
    E,D,b=(torch.stack([torch.tensor(a[q]) for a in records]) for q in ('E','D','b'))
    errors=np.empty((len(records),count),dtype=np.float64)
    with torch.no_grad():
        for start in range(0,count,4096):
            a=x[start:start+4096]
            error=(torch.relu((a@E.transpose(1,2))@D+b[:,None,:])-a).square().double().mean(dim=2)
            errors[:,start:start+len(a)]=error.numpy()
    baseline=errors[0];mean_errors=errors[1]
    for i,a in enumerate(records):
        err=errors[i];ratio=err.mean()/baseline.mean();gain=100*(1-ratio)
        se=100*np.std(err-ratio*baseline,ddof=1)/math.sqrt(count)/baseline.mean()
        a.update(test_mse=float(err.mean()),test_mse_se=float(err.std(ddof=1)/math.sqrt(count)),
                 improvement_percent=float(gain),improvement_ci95_percent=[float(gain-1.96*se),float(gain+1.96*se)],
                 paired_mse_difference_vs_tied=float((err-baseline).mean()),
                 paired_difference_se=float((err-baseline).std(ddof=1)/math.sqrt(count)),
                 mse_difference_vs_mean=float((err-mean_errors).mean()))
    crossing=[]
    for low,high in zip(points,points[1:]):
        if low['test_mse']<tied['test_mse']<=high['test_mse']:
            fraction=(tied['test_mse']-low['test_mse'])/(high['test_mse']-low['test_mse'])
            estimate=math.exp(math.log(low['weight_decay'])*(1-fraction)+math.log(high['weight_decay'])*fraction) if low['weight_decay'] else None
            crossing.append(dict(lower_decay=low['weight_decay'],upper_decay=high['weight_decay'],log_interpolated_estimate=estimate))
    # A convergence diagnostic compares the final two checkpoints of each winner.
    for a in points:
        group=data[a['source_group']];index=next(i for i,r in enumerate(group['runs']) if r['id']==a['id'])
        before=group['trace'][-2]['validation_mse'][index]
        after=group['trace'][-1]['validation_mse'][index]
        a['last_5000_step_validation_change']=after-before
        a['initial_pass_best_validation_mse']=initial.get(a['weight_decay'],{}).get('validation_mse')
    summary=dict(n=base.N,m=base.M,sparsity=base.S,
                 optimizer='AdamW, decay only on encoder/decoder weights; learned output bias exempt',
                 definition='MSE averaged over all examples and all five features; output ReLU; no encoder bias',
                 selection='Across longest-run final states by validation MSE; never select an earlier low-MSE checkpoint before regularization takes effect',
                 validation_seed=957001,validation_samples=131072,test_seed=seed,test_samples=count,
                 initial_decays=DECAYS,counts={name:len(d['runs']) for name,d in data.items()},
                 mean_prediction=MEAN,mean_population_mse=MEAN_MSE,
                 tied_reference='Fixed trained tied model without weight decay; crossing its loss does not imply recovering its weights',
                 optimizer_caveat='AdamW decay is decoupled shrinkage, not the coefficient of a claimed MSE-plus-L2 optimum; numerical coefficients depend on the optimizer and training recipe',
                 uncertainty='Paired test-sampling 95% intervals; not optimization uncertainty',
                 global_optimality_established=False,tied=tied,mean_baseline=mean,
                 tied_crossing_brackets=crossing,points=points)
    base.save_json(output/'summary.json',summary)

    wd=np.array([a['weight_decay'] for a in points]);loss=np.array([a['test_mse'] for a in points])
    err=np.array([1.96*a['test_mse_se'] for a in points])
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.5))
    crossing_upper=crossing[0]['upper_decay'] if crossing else 1.
    zoom_max=max(.03,min(3*crossing_upper,max(wd)))
    for ax in axes:
        ax.errorbar(wd,loss,yerr=err,fmt='o-',color='#2779a7',markersize=3.5,lw=1.6,capsize=2,label='Untied, weights decayed')
        ax.axhline(tied['test_mse'],color='#71569b',lw=1.3,ls='--',label='Tied, no decay')
        ax.axhline(mean['test_mse'],color='#7c7c7c',lw=1.3,ls=':',label='Predict the mean')
        ax.set_xscale('symlog',linthresh=1e-4,linscale=.6)
        ax.set_xlabel('AdamW weight decay (biases exempt)')
        ax.set_ylabel('Reconstruction MSE')
        ax.grid(axis='y',alpha=.15)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
    axes[0].set_title('Full trajectory',fontsize=11)
    axes[0].set_xlim(-1e-5,max(wd)*1.25)
    axes[0].set_xticks([0,1e-4,.001,.01,.1,1,10,100,1000])
    axes[0].tick_params(axis='x',labelsize=8)
    axes[0].set_ylim(0,mean['test_mse']*1.09)
    axes[0].legend(frameon=False,fontsize=8,loc='center left')
    axes[1].set_title('Near the tied baseline',fontsize=11)
    axes[1].set_xlim(-1e-5,zoom_max)
    axes[1].set_xticks([t for t in [0,.001,.01,.1,.3,1,3,10] if t<=zoom_max])
    axes[1].tick_params(axis='x',labelsize=8)
    axes[1].set_ylim(min(loss)*.92,tied['test_mse']*1.22)
    fig.tight_layout(pad=1.3)
    fig.savefig(output/'weight_decay_mse.png',dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)

    lines=['# Weight decay and the untied solution','',
           '![Reconstruction MSE against weight decay](weight_decay_mse.png)','',
           'The same five-feature, two-dimensional model at sparsity 0.9, retrained with AdamW weight decay.',
           'Only encoder and decoder weights are decayed; output biases remain trainable and unpenalized.',
           'MSE averages over examples and all five features; error bars show test-sampling 95% intervals.',
           'The tied reference is an unregularized tied model, not a tied model retrained at each decay strength.','',
           '## Main results','',
           f'- Zero decay: MSE {points[0]["test_mse"]:.6e}, improvement {points[0]["improvement_percent"]:.2f}% over tied.',
           f'- Tied reference: MSE {tied["test_mse"]:.6e}.',
           f'- Predicting the population mean {MEAN:.2f}: population MSE {MEAN_MSE:.8f}, shared-test MSE {mean["test_mse"]:.6e}.']
    for a in points:
        if a['weight_decay']==.01:lines.append(f'- Weight decay 0.01: MSE {a["test_mse"]:.6e}, improvement {a["improvement_percent"]:.2f}% over tied.')
    for cross in crossing:
        lines.append(f'- Tied-level loss is crossed between decay {cross["lower_decay"]:g} and {cross["upper_decay"]:g} (log-axis interpolation approximately {cross["log_interpolated_estimate"]:.4g}; not an exact threshold).')
    final=points[-1]
    lines += [f'- At decay {final["weight_decay"]:g}: MSE {final["test_mse"]:.6e}; biases range from {min(final["b"]):.5f} to {max(final["b"]):.5f}.','',
              'These are numerical training outcomes, not globally proven optima.',
              'AdamW is decoupled weight shrinkage; its coefficient is not interchangeable with an explicit L2-penalty coefficient.',
              'Crossing tied-level loss does not mean that the learned weights become tied.','',
              '## Full table','',
              '| Weight decay | Test MSE | Improvement over tied | Paired 95% interval | Total squared weight norm |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for a in points:
        lo,hi=a['improvement_ci95_percent']
        lines.append(f'| {a["weight_decay"]:g} | {a["test_mse"]:.6e} | {a["improvement_percent"]:+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] | {a["weight_squared_norm"]:.5g} |')
    lines += ['','## Training protocol','',
              'Each initial coefficient has six starts: the pictured untied solution, the refined untied solution, tied weights, random weights, small random weights with mean biases, and untied weights with mean biases.',
              'Positive-bias starts prevent zero-output ReLU states from being mistaken for the mean-prediction limit.',
              'After the first pass, the best final model at each neighboring coefficient supplies warm starts from both lower and higher decay.',
              'The best final result per coefficient and the pictured-model branch receive longer training.',
              'Four additional coefficients near the preliminary tied-level crossing are initialized from both bounding solutions.',
              'Checkpoint traces are diagnostic: the curve selects among final states of the longest runs, not among earlier checkpoints with lower reconstruction loss.',
              'Training uses fresh samples, learning rate 0.001 held constant for half the run and cosine-decayed to 0.00001 thereafter; Adam betas=(0.9,0.999), epsilon=1e-8.',
              'The per-model batched decay implementation is numerically checked against torch.optim.AdamW, including exact bias exemption.',
              'Model selection uses 131,072 validation examples; every final point and both baselines share a separate test draw of one million examples.','']
    for name,d in data.items():lines.append(f'- `{name}`: {len(d["runs"])} models, {d["steps"]:,} steps, batch {d["batch_size"]:,}.')
    lines += ['','Warm starts inherit earlier training; this is a search for stable solutions, not a matched-compute comparison.',
              f'The largest change in validation MSE over the final 5,000 steps among plotted models was {100*max(abs(a["last_5000_step_validation_change"])/a["validation_mse"] for a in points):.2f}%.',
              'The original notebook applied decay to its output bias too; this sweep intentionally excludes biases as requested.','',
              '## Files','',
              '- [Summary, all curve points, and selected weights](summary.json).',
              '- [Independent NumPy verification of the test MSEs](verification.json).',
              '- `explore_*.json`, `continuation_*.json`, and `refine_*.json` retain all final models and convergence traces.',
              '- Included reproduction scripts: `results/core/tied_untied/weight_decay_trajectory.py` and its helper `one_sided_norm_experiments.py`.','',
              '```sh',
              'python results/core/tied_untied/weight_decay_trajectory.py --phase explore --shard 0',
              'python results/core/tied_untied/weight_decay_trajectory.py --phase explore --shard 1',
              'python results/core/tied_untied/weight_decay_trajectory.py --phase continuation --shard 0',
              'python results/core/tied_untied/weight_decay_trajectory.py --phase continuation --shard 1',
              'python results/core/tied_untied/weight_decay_trajectory.py --phase refine --shard 0 --steps 60000 --batch 4096',
              'python results/core/tied_untied/weight_decay_trajectory.py --phase refine --shard 1 --steps 60000 --batch 4096',
              'MPLCONFIGDIR=/tmp/weight-decay-mpl python results/core/tied_untied/weight_decay_trajectory.py --phase report',
              '```','']
    (output/'README.md').write_text('\n'.join(lines))
    print(f'Tied={tied["test_mse"]:.8f}, mean={mean["test_mse"]:.8f}; crossings={crossing}',flush=True)
    for a in points:print(f'wd={a["weight_decay"]:g}: MSE={a["test_mse"]:.8f}, gain={a["improvement_percent"]:+.2f}%',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('explore','continuation','refine','report','self-check'),default='explore')
    parser.add_argument('--shard',type=int,choices=(0,1),default=0)
    parser.add_argument('--steps',type=int,default=30000)
    parser.add_argument('--batch',type=int,default=2048)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();torch.set_num_threads(1);self_check()
    if args.phase=='self-check':return
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=='report':make_report(args.output);return
    validation=base.sample(131072,torch.Generator().manual_seed(957001))
    if args.phase=='explore':
        starts=make_starts(args.shard);name=f'explore_shard{args.shard}';seed=957100+args.shard
    elif args.phase=='continuation':
        starts=continuation_starts(args.output)
        starts=[a for a in starts if DECAYS.index(a['weight_decay'])%2==args.shard]
        name=f'continuation_shard{args.shard}';seed=957200+args.shard
    else:
        starts=refine_starts(args.output)
        keys=sorted(set(a['weight_decay'] for a in starts))
        starts=[a for a in starts if keys.index(a['weight_decay'])%2==args.shard]
        name=f'refine_shard{args.shard}';seed=957300+args.shard
    path=args.output/f'{name}.json'
    if path.exists() and json.loads(path.read_text())['completed_steps']==args.steps:
        print(f'Skipping completed {name}',flush=True);return
    train(starts,args.steps,args.batch,seed,path,validation)


if __name__=='__main__':main()
