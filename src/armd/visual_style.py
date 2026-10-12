"""A small shared plotting style for clear statistical figures."""
from pathlib import Path
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
        'figure.facecolor':'white'})


def clean_figure(fig):
    """Keep axis names and tick labels, with no other displayed text."""
    axis_labels = [getattr(fig, name, None) for name in ['_supxlabel', '_supylabel']]
    for label in fig.texts:
        if label not in axis_labels:
            label.set_visible(False)
    for legend in fig.legends:
        legend.set_visible(False)
    for ax in fig.axes:
        for location in ['left', 'center', 'right']:
            ax.set_title('', loc=location)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for label in ax.texts:
            label.set_visible(False)


def xgrid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='x',color=GRID,linewidth=.7)
    ax.tick_params(axis='y',pad=8)


def save(fig, directory, stem):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    clean_figure(fig)
    fig.tight_layout(pad=1.2)
    fig.savefig(directory/f'{stem}.png',dpi=260,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)
