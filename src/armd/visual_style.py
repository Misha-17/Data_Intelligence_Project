"""A small shared plotting style for clear statistical figures."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAVY = '#183B4E'
TEAL = '#087E8B'
ORANGE = '#C66B3D'
GOLD = '#B88726'
GRAY = '#788A96'
PALE = '#EEF4F5'
GRID = '#DDE5E8'
INK = '#203743'
DRUG_LABELS = {
    'ampicillin':'Ampicillin', 'amoxicillin/clavulanate':'Amoxicillin / clavulanate',
    'cefazolin':'Cefazolin','cefepime':'Cefepime','ceftriaxone':'Ceftriaxone',
    'ciprofloxacin':'Ciprofloxacin','gentamicin':'Gentamicin',
    'piperacillin/tazobactam':'Piperacillin / tazobactam',
    'trimethoprim/sulfamethoxazole':'Trimethoprim / sulfamethoxazole',
}
MODEL_LABELS = {
    'prevalence_baseline':'Overall baseline','antibiotic_baseline':'Drug baseline',
    'logistic_unweighted':'Logistic regression','logistic_balanced':'Logistic, weighted',
    'lightgbm_unweighted':'LightGBM','lightgbm_balanced':'LightGBM, weighted (selected)',
}
FEATURE_LABELS = {
    'lab_median_wbc_14d':'White blood cell count',
    'lab_median_creatinine_14d':'Creatinine',
    'vital_median_heart_rate_48h':'Heart rate',
    'vital_median_temperature_48h':'Temperature',
    'prior_abx_days_ago':'Days since antibiotic exposure',
    'prior_abx_class_count_90d':'Antibiotic classes in 90 days',
    'prior_candidate_resistance_days_ago':'Days since resistance to candidate',
}


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.labelcolor':INK,'text.color':INK,'xtick.color':GRAY,'ytick.color':INK,
        'axes.edgecolor':GRID,'axes.spines.top':False,'axes.spines.right':False,
        'axes.spines.left':False,'axes.spines.bottom':False,
        'axes.labelsize':11,'axes.titlesize':13,'axes.titleweight':'bold',
        'xtick.labelsize':10,'ytick.labelsize':10,'xtick.major.size':0,'ytick.major.size':0,
        'legend.frameon':False,'legend.fontsize':10,'savefig.facecolor':'white',
        'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})


def header(fig, title, subtitle):
    fig.text(.04,.975,'ARMD | EXPLORATORY ANALYSIS',ha='left',va='top',fontsize=9,color=TEAL,weight='bold')
    fig.text(.04,.93,title,ha='left',va='top',fontsize=19,weight='bold',color=NAVY)
    fig.text(.04,.872,subtitle,ha='left',va='top',fontsize=11,color=GRAY)


def footer(fig, note, width=155):
    fig.text(.04,.025,textwrap.fill(note,width),ha='left',va='bottom',fontsize=9,color=GRAY,linespacing=1.45)


def xgrid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='x',color=GRID,linewidth=.7)
    ax.tick_params(axis='y',pad=8)


def save(fig, directory, stem):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg','pdf']:
        fig.savefig(directory/f'{stem}.{extension}',dpi=260,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)
