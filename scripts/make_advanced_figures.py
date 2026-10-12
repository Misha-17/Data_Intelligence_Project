"""Create exportable statistical figures from the frozen and exploratory results."""
from pathlib import Path
import json
import os
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from armd.visual_style import (setup,header,footer,xgrid,save,NAVY,TEAL,ORANGE,GRAY,GRID,PALE,INK,
                               DRUG_LABELS,MODEL_LABELS,FEATURE_LABELS)


def interval(ax,x,y,lo,hi,color=TEAL,marker='o',size=7):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.hlines(y,lo,hi,color=color,lw=2,alpha=.75)
        ax.vlines([lo,hi],y-.065,y+.065,color=color,lw=1)
    ax.plot(x,y,marker=marker,color=color,ms=size,markeredgecolor='white',markeredgewidth=.8,zorder=4)


def register(index,stem,title,question,sources,note):
    index.append({'stem':stem,'title':title,'question':question,'source_tables':sources,'note':note,
                  'files':{ext:f'figures/{stem}.{ext}' for ext in ['png','svg','pdf']}})


def comparison(root,out,index):
    ci=pd.read_csv(root/'results/models/test_confidence_intervals.csv')
    order=['lightgbm_balanced','lightgbm_unweighted','logistic_balanced','logistic_unweighted','antibiotic_baseline','prevalence_baseline']
    fig,axes=plt.subplots(1,3,figsize=(13.6,6.6),sharey=True)
    header(fig,'Model differences, with uncertainty','Original test results | six comparisons | the selected model stays fixed')
    definitions=[('antibiotic_macro_auroc','Drug-macro AUROC',(.45,.79)),
                 ('antibiotic_macro_auprc_r','Drug-macro resistance AP',(.12,.49)),
                 ('brier','Probability error (Brier)',(.095,.17))]
    for ax,(metric,title,limits) in zip(axes,definitions):
        for y,model in enumerate(order):
            row=ci[(ci.model==model)&(ci.metric==metric)].iloc[0]
            color=TEAL if model=='lightgbm_balanced' else (NAVY if 'lightgbm' in model else GRAY)
            if y==0: ax.axhspan(-.38,.38,color=PALE,zorder=0)
            interval(ax,row.estimate,y,row.ci_low,row.ci_high,color)
            ax.text(limits[1],y,f'{row.estimate:.3f}',ha='right',va='center',fontsize=10,color=color)
        ax.set_title(title,loc='left',pad=17)
        ax.set_xlim(limits); xgrid(ax)
        ax.set_xlabel('Lower is better' if metric=='brier' else 'Higher is better')
        ax.set_yticks(range(len(order)),[MODEL_LABELS[x] for x in order]); ax.set_ylim(len(order)-.5,-.5)
    footer(fig,'Dots show estimates; lines show 95% patient-bootstrap intervals from the primary run (200 draws). AP is average precision for resistance. Weighted and unweighted LightGBM are very close; this is not evidence of a clear benefit from class weighting.')
    fig.subplots_adjust(left=.235,right=.96,bottom=.2,top=.75,wspace=.28)
    stem='01_model_comparison'; save(fig,out,stem)
    register(index,stem,'Model differences, with uncertainty','How large are the differences between the original models?',
             ['../models/test_confidence_intervals.csv'],'Original test results; fixed fitted models; pointwise intervals.')


def ablation(root,out,index):
    f=pd.read_csv(root/'results/advanced/ablation_metrics.csv')
    delta=pd.read_csv(root/'results/advanced/ablation_paired_differences.csv')
    order=['core_only','core_labs_vitals','core_availability_only','core_prior_history','full']
    labels=['Core only','Core + labs and vital signs','Core + availability patterns','Core + prior history','All features (original model)']
    fig,axes=plt.subplots(1,2,figsize=(12.5,6.8),sharey=True)
    header(fig,'What each feature group adds on validation','Feature-group comparisons on validation only | identical patient partitions and fixed model settings')
    for ax,metric,title,limits in zip(axes,['antibiotic_macro_auprc_r','brier'],
                                     ['Resistance average precision','Probability error (Brier)'],[(.18,.45),(.1,.14)]):
        for y,model in enumerate(order):
            row=f[(f.model==model)&(f.metric==metric)].iloc[0]
            color=TEAL if model in ['core_prior_history','full'] else (ORANGE if model=='core_availability_only' else GRAY)
            if model=='full': ax.axhspan(y-.36,y+.36,color=PALE)
            interval(ax,row.estimate,y,row.ci_low,row.ci_high,color)
            ax.text(limits[1],y,f'{row.estimate:.3f}',ha='right',va='center',fontsize=11,color=color)
        ax.set(xlim=limits,ylim=(4.6,-.6),yticks=range(5),yticklabels=labels)
        ax.set_title(title,loc='left',pad=17); ax.set_xlabel('Higher is better' if metric!='brier' else 'Lower is better'); xgrid(ax)
    footer(fig,'Core = drug, age band, gender code, culture type and ward flags. The availability control keeps missing-value patterns but removes recorded numeric values. Lines are 95% patient-bootstrap intervals (500 shared draws). This experiment is exploratory and does not measure causal effects.',145)
    fig.subplots_adjust(left=.26,right=.96,top=.75,bottom=.23,wspace=.32)
    stem='02_feature_group_ablation';save(fig,out,stem)
    register(index,stem,'What each feature group adds on validation','Do measured values add information beyond basic fields and recording patterns?',
             ['ablation_metrics.csv','ablation_paired_differences.csv'],'Validation only. Similar scores do not establish equivalence or transportability.')


def per_drug(root,out,index,drugs):
    f=pd.read_csv(root/'results/advanced/drug_performance.csv').set_index('candidate_antibiotic').loc[drugs]
    fig,axes=plt.subplots(1,3,figsize=(14.2,8.3),sharey=True)
    header(fig,'Performance varies substantially across antibiotics','Frozen selected model | test results | confidence intervals keep each patient’s records together')
    defs=[('auroc','Ranking (AUROC)',(.3,1.05)),('ap_r','Resistance average precision',(0,1.02)),('recall_r','Resistance found (recall)',(-.03,1.03))]
    for ax,(metric,title,limits) in zip(axes,defs):
        for y,(drug,row) in enumerate(f.iterrows()):
            if y%2==0: ax.axhspan(y-.42,y+.42,color='#F6F8F9')
            interval(ax,row[metric],y,row[metric+'_ci_low'],row[metric+'_ci_high'],TEAL)
            if metric=='ap_r': ax.plot(row.prevalence_r,y,'|',color=ORANGE,ms=16,mew=2)
        ax.set(xlim=limits,ylim=(8.6,-.65),yticks=range(9),yticklabels=[DRUG_LABELS[x]+f'  |  R = {int(f.loc[x,"n_r"]):,}' for x in drugs])
        ax.set_title(title,loc='left',pad=18); xgrid(ax)
        if metric=='recall_r':
            ax.set_xticks([0,.2,.4,.6,.8,1])
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].axvline(.5,ls='--',color=GRAY,lw=1)
    axes[1].legend(handles=[Line2D([0],[0],color=TEAL,marker='o',label='Model AP'),
                           Line2D([0],[0],color=ORANGE,marker='|',ls='',markersize=12,label='Resistance prevalence')],
                   loc='lower right',bbox_to_anchor=(1,-.22),fontsize=9)
    footer(fig,'R counts resistant test rows. Lines are 95% patient-bootstrap intervals (500 draws). Orange ticks mark resistance prevalence, a no-skill AP baseline. Recall uses the original cutoff. Boundary intervals at 0% or 100% do not imply certainty for future patients. AP also depends on prevalence and case mix.',160)
    fig.subplots_adjust(left=.30,right=.97,top=.78,bottom=.22,wspace=.24)
    stem='03_drug_performance';save(fig,out,stem)
    register(index,stem,'Performance varies substantially across antibiotics','Which drugs have stronger ranking and resistance detection, and how certain are those estimates?',
             ['drug_performance.csv'],'Drug order follows the configured panel, not test performance.')


def calibration_overall(root,out,index):
    f=pd.read_csv(root/'results/advanced/calibration_bins.csv')
    hist=pd.read_csv(root/'results/advanced/probability_histogram.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.5,7),gridspec_kw={'width_ratios':[1.12,1]})
    header(fig,'Probability checks before and after calibration','Resistance probability = 1 minus predicted susceptibility | calibration was fitted on training patients only')
    ax=axes[0]
    ax.plot([0,1],[0,1],ls='--',color=GRAY,lw=1,label='Exact agreement')
    for state,color,label in [('raw',ORANGE,'Before calibration'),('calibrated',TEAL,'After calibration')]:
        sub=f[(f.scope=='overall')&(f.probability==state)].sort_values('bin')
        sub=sub[sub.n>0]
        ax.plot(sub.mean_predicted_r,sub.observed_r,color=color,lw=1.6,alpha=.85)
        ax.vlines(sub.mean_predicted_r,sub.observed_r_ci_low,sub.observed_r_ci_high,color=color,lw=1.5,alpha=.55)
        ax.scatter(sub.mean_predicted_r,sub.observed_r,color=color,s=42,label=label,zorder=4,edgecolors='white')
        h=hist[(hist.scope=='overall')&(hist.probability==state)]
        mid=(h.edge_left+h.edge_right)/2
        axes[1].plot(mid,h.n/h.n.sum(),color=color,lw=2,label=label)
        axes[1].fill_between(mid,h.n/h.n.sum(),color=color,alpha=.08)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability of resistance',ylabel='Observed resistant share')
    ax.set_title('Agreement with observed outcomes',loc='left',pad=16)
    ax.grid(color=GRID,lw=.6); ax.set_axisbelow(True); ax.legend(loc='upper left',fontsize=9)
    axes[1].set(xlim=(0,1),ylim=(0,None),xlabel='Predicted probability of resistance',ylabel='Share of test rows')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1)); axes[1].grid(axis='y',color=GRID,lw=.6)
    axes[1].set_title('Where the predictions fall',loc='left',pad=16);axes[1].legend(fontsize=9)
    footer(fig,'Reliability-bin edges were fixed from validation predictions. Vertical lines show 95% patient-bootstrap intervals for the observed resistant share. Distribution lines summarize fixed-width probability bins. These are exploratory checks on the existing test set, not a new calibration fit.',150)
    fig.subplots_adjust(left=.085,right=.965,bottom=.20,top=.75,wspace=.35)
    stem='04_overall_calibration';save(fig,out,stem)
    register(index,stem,'Probability checks before and after calibration','Are the probabilities close to observed resistance rates, and how concentrated are they?',
             ['calibration_bins.csv','probability_histogram.csv','calibration_summary.csv'],'A lower Brier score alone does not prove better calibration; use the reliability display too.')


def calibration_drugs(root,out,index,drugs):
    f=pd.read_csv(root/'results/advanced/calibration_bins.csv')
    fig,axes=plt.subplots(3,3,figsize=(13.6,11.5),sharex=True,sharey=True)
    fig.text(.05,.975,'ARMD | EXPLORATORY ANALYSIS',fontsize=9,color=TEAL,weight='bold',va='top')
    fig.text(.05,.944,'A good pooled calibration curve can hide drug differences',fontsize=18,color=NAVY,weight='bold',va='top')
    fig.text(.05,.912,'Selected model | test outcomes in validation-defined bins | open circles mark sparse bins',fontsize=11,color=GRAY,va='top')
    for ax,drug in zip(axes.flat,drugs):
        sub=f[(f.scope=='drug')&(f.candidate_antibiotic==drug)&(f.probability=='calibrated')].sort_values('bin')
        sub=sub[sub.n>0]
        ax.plot([0,1],[0,1],ls='--',color=GRAY,lw=.9)
        ax.plot(sub.mean_predicted_r,sub.observed_r,color=TEAL,lw=1,alpha=.6)
        for _,row in sub.iterrows():
            ax.vlines(row.mean_predicted_r,row.observed_r_ci_low,row.observed_r_ci_high,color=TEAL,lw=1.25,alpha=.65)
            ax.scatter(row.mean_predicted_r,row.observed_r,s=25+65*np.sqrt(row.n/sub.n.max()),
                       facecolor='white' if bool(row['sparse']) else TEAL,edgecolor=TEAL,lw=1.2,zorder=4)
        ax.set_title(DRUG_LABELS[drug],loc='left',fontsize=10.6,pad=9)
        ax.text(.98,.04,f'n = {int(sub.n.sum()):,}',transform=ax.transAxes,ha='right',fontsize=9,color=GRAY)
        ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.25,.5,.75,1],yticks=[0,.25,.5,.75,1])
        ax.grid(color=GRID,lw=.6);ax.set_axisbelow(True)
    fig.supxlabel('Mean predicted probability of resistance',x=.53,y=.09,fontsize=12)
    fig.supylabel('Observed resistant share',x=.015,y=.5,fontsize=12)
    footer(fig,'The dashed line marks exact agreement. Larger points mark larger bins; lines show 95% patient-bootstrap intervals (500 draws). Sparse means fewer than 100 rows or fewer than 20 resistant results in a bin. Intervals are pointwise and do not prove calibration across all probabilities.',160)
    fig.subplots_adjust(left=.075,right=.98,top=.856,bottom=.14,wspace=.19,hspace=.28)
    stem='05_calibration_by_drug';save(fig,out,stem)
    register(index,stem,'A good pooled calibration curve can hide drug differences','Does probability agreement hold within each candidate drug?',
             ['calibration_bins.csv','calibration_summary.csv'],'Open markers flag sparse bins; their estimates need more caution.')


def threshold_tradeoff(root,out,index):
    f=pd.read_csv(root/'results/advanced/threshold_operating_points.csv')
    test=f[f['split']=='test'].copy()
    order=['original','recall_70','recall_80','recall_90']
    field='operating_point' if 'operating_point' in test else 'policy'
    test=test.set_index(field).loc[order]
    labels=['Original cutoff','70% validation recall target','80% validation recall target','90% validation recall target']
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.6),sharey=True)
    header(fig,'Finding more resistance brings more false alerts','Cutoffs chosen on validation, then applied unchanged to test | counts per 1,000 candidate test rows')
    for ax,metric,title,color in zip(axes,['false_alerts_per_1000','missed_r_per_1000'],
                                   ['False resistance alerts','Resistant results missed'],[ORANGE,TEAL]):
        upper=test[metric+'_ci_high'].max()*1.30
        for y,(_,row) in enumerate(test.iterrows()):
            ax.barh(y,row[metric],height=.42,color=color,alpha=.18)
            interval(ax,row[metric],y,row[metric+'_ci_low'],row[metric+'_ci_high'],color,size=7)
            ax.text(upper,y,f'{row[metric]:.0f}',ha='right',va='center',fontsize=12,weight='bold',color=color)
        ax.set(xlim=(0,upper),ylim=(3.6,-.65),yticks=range(4),yticklabels=labels)
        ax.set_title(title,loc='left',pad=17);ax.set_xlabel('Rows per 1,000 test rows');xgrid(ax)
    for y,(_,row) in enumerate(test.iterrows()):
        axes[1].text(.0,y+.3,f'Actual test recall: {row.recall_r:.1%}  |  alerts correct: {row.precision_r:.1%}',
                     transform=axes[1].get_yaxis_transform(),fontsize=9,color=GRAY)
    footer(fig,'False alerts are susceptible rows flagged as resistant. Misses are resistant rows flagged as susceptible. Lines are 95% patient-bootstrap intervals. Targets describe validation recall; actual test recall may differ. These retrospective operating points are not clinical treatment rules.',155)
    fig.subplots_adjust(left=.26,right=.965,top=.75,bottom=.23,wspace=.35)
    stem='06_threshold_tradeoffs';save(fig,out,stem)
    register(index,stem,'Finding more resistance brings more false alerts','What is the error burden when a cutoff aims to find more resistance?',
             ['threshold_operating_points.csv','fixed_thresholds_validation.json'],'Counts refer to candidate test rows, not patients. No treatment benefit or cost assumption is made.')


def coverage(root,out,index,drugs):
    f=pd.read_csv(root/'results/advanced/organism_drug_slices.csv')
    organisms=list(dict.fromkeys(f.organism))
    matrix=f.pivot(index='organism',columns='candidate_antibiotic',values='observed_panel_coverage').reindex(index=organisms,columns=drugs)
    fig,ax=plt.subplots(figsize=(14,7.5))
    header(fig,'The available labels are uneven across organisms and drugs','Six most common training organisms | test-set retained label coverage | empty outcomes remain unknown')
    cmap=LinearSegmentedColormap.from_list('coverage',['#F5F8F8','#B5D8D9',TEAL,NAVY])
    image=ax.imshow(matrix.to_numpy(float),cmap=cmap,vmin=0,vmax=1,aspect='auto')
    for i in range(len(organisms)):
        for j in range(len(drugs)):
            v=matrix.iloc[i,j]
            if np.isfinite(v):
                label='<1%' if 0<v<.005 else ('>99%' if .995<=v<1 else f'{v:.0%}')
                ax.text(j,i,label,ha='center',va='center',fontsize=11,color='white' if v>.6 else INK)
    ax.set_xticks(range(9),[DRUG_LABELS[d].replace(' / ',' /\n') for d in drugs],rotation=35,ha='right')
    ax.set_yticks(range(len(organisms)),[x.capitalize() for x in organisms])
    ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,len(organisms),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=3);ax.tick_params(which='minor',bottom=False,left=False)
    bar=fig.colorbar(image,ax=ax,fraction=.025,pad=.025);bar.ax.yaxis.set_major_formatter(PercentFormatter(1));bar.set_label('Retained label coverage')
    footer(fig,'Each percentage is the share of included organism-events with a clear S/R label for that drug. The denominator includes events with at least one retained panel result. This is not the raw testing rate. A zero means no retained label, not susceptibility or resistance.',160)
    fig.subplots_adjust(left=.23,right=.94,top=.77,bottom=.31)
    stem='07_label_coverage';save(fig,out,stem)
    register(index,stem,'The available labels are uneven across organisms and drugs','Which organism-drug combinations are well represented in the retained data?',
             ['organism_drug_slices.csv'],'The analysis cannot establish why a label is absent or estimate performance for untested combinations.')


def missingness(root,out,index):
    df=pd.read_parquet(root/'results/analytical.parquet')
    features=list(FEATURE_LABELS)
    rows=[]
    for name in features:
        keys=['patient_id','culture_id']+(['candidate_antibiotic'] if name.startswith('prior_candidate') else [])
        values=df[keys+['split',name]]
        if values.groupby(keys,dropna=False)[name].nunique(dropna=False).max()>1:
            raise ValueError(f'Conflicting values at missingness unit: {name}')
        frame=values.drop_duplicates(keys)
        for split,group in frame.groupby('split'):
            rows.append({'feature':name,'split':split,'units':len(group),'missing_n':int(group[name].isna().sum()),
                         'missing_fraction':float(group[name].isna().mean()),
                         'unit':'culture-candidate' if name.startswith('prior_candidate') else 'culture'})
    f=pd.DataFrame(rows);f.to_csv(root/'results/advanced/missingness_by_split.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,7))
    header(fig,'Missing information is common in every data split','Source information before imputation | each culture counted once for the shared clinical fields')
    for split,offset,color in [('train',-.15,NAVY),('validation',0,ORANGE),('test',.15,TEAL)]:
        sub=f[f['split']==split].set_index('feature').loc[features]
        ax.scatter(sub.missing_fraction,np.arange(7)+offset,color=color,s=55,label=split.capitalize(),edgecolors='white',zorder=4)
    ax.set(xlim=(0,1),ylim=(6.65,-.65),yticks=range(7),yticklabels=[FEATURE_LABELS[n] for n in features],xlabel='Share with no usable value')
    ax.xaxis.set_major_formatter(PercentFormatter(1));xgrid(ax)
    ax.legend(loc='lower center',bbox_to_anchor=(.55,-.20),ncol=3)
    footer(fig,'Laboratory, vital-sign and antibiotic-exposure fields use one record per culture. Candidate resistance history uses one record per culture and candidate drug. No record means unknown; it does not establish no prior exposure or no prior resistance. Similar missing rates alone do not establish comparable patients.',145)
    fig.subplots_adjust(left=.30,right=.965,top=.77,bottom=.26)
    stem='08_missingness';save(fig,out,stem)
    register(index,stem,'Missing information is common in every data split','How much clinical information is unavailable, and does that change across splits?',
             ['missingness_by_split.csv'],'Counts avoid repeatedly counting shared culture features once per organism and drug.')


def report_calibration(root,out):
    """A compact version of figure 04 sized for the five-page report."""
    f=pd.read_csv(root/'results/advanced/calibration_bins.csv')
    fig,ax=plt.subplots(figsize=(6.4,3.45))
    ax.plot([0,1],[0,1],ls='--',color=GRAY,lw=1,label='Exact agreement')
    for state,color,label in [('raw',ORANGE,'Before calibration'),('calibrated',TEAL,'After calibration')]:
        sub=f[(f.scope=='overall')&(f.probability==state)&(f.n>0)].sort_values('bin')
        ax.plot(sub.mean_predicted_r,sub.observed_r,color=color,lw=1.6,marker='o',ms=4,label=label)
        ax.vlines(sub.mean_predicted_r,sub.observed_r_ci_low,sub.observed_r_ci_high,color=color,lw=1,alpha=.75)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted resistance probability',ylabel='Observed resistant share')
    ax.legend(loc='upper left',fontsize=9)
    ax.grid(color=GRID,lw=.6);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.12,right=.985,top=.97,bottom=.16)
    save(fig,out,'report_calibration')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    root=args.root;out=root/'results/advanced/figures'; setup()
    drugs=json.loads((root/'config.json').read_text())['antibiotics']
    index=[]
    comparison(root,out,index);ablation(root,out,index);per_drug(root,out,index,drugs)
    calibration_overall(root,out,index);calibration_drugs(root,out,index,drugs)
    threshold_tradeoff(root,out,index);coverage(root,out,index,drugs);missingness(root,out,index)
    report_calibration(root,out)
    (out.parent/'figure_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    print(f'Created {len(index)} figures in PNG, SVG and PDF formats.')


if __name__=='__main__':
    main()
