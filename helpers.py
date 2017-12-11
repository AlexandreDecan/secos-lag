import subprocess
import pandas
import scipy
import os


RE_SEMVER = r'^(?:v|V)?(?P<v_major>\d+)\.(?P<v_minor>\d+)\.(?P<v_patch>\d+)(?P<v_misc>.*)$'

FIG_SIZE = (7, 4)
DATE_RANGE = (pandas.to_datetime('2011-01-01'), pandas.to_datetime('2018-01-01'))
FIGURE_PATH = '../figures'


def savefig(fig, name):
    fig.savefig(
        os.path.join(FIGURE_PATH, '{}.pdf'.format(name)),
        bbox_inches='tight'
    )


def semver(constraint, versions):
    """
    Return the versions in ``versions'' that satisfy given constraint.
    Semantic is provided by the semver tool of nodejs.
    """
    args = ['semver', '-r', constraint] + list(versions)
    
    completed = subprocess.run(args, stdout=subprocess.PIPE)
    if completed.returncode == 0:
        return completed.stdout.decode().strip().split('\n')
    else:
        return []
    

def comply_semver(series_of_releases):
    extracted = series_of_releases.str.extract(RE_SEMVER, expand=True)
    return (
        extracted
        [['v_major', 'v_minor', 'v_patch']]
        .astype('str')
        .assign(Semver=lambda d: d['v_major'] + '.' + d['v_minor'] + '.' + d['v_patch'])
        ['Semver']
    )


def CohenEffectSize(group1, group2):
    """Compute Cohen's d.

    group1: Series or NumPy array
    group2: Series or NumPy array

    returns: float
    
    effect sizes as "small, d = .2," "medium, d = .5," and "large, d = .8"
    """
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / pandas.np.sqrt(pooled_var)
    return d
    

def cliffsDelta(lst1, lst2):
    """
    0.147 / 0.33 / 0.474 (negligible/small/medium/large).
    """
    def runs(lst):
        "Iterator, chunks repeated values"
        for j, two in enumerate(lst):
            if j == 0:
                one, i = two, 0
            if one != two:
                yield j - i, one
                i = j
            one = two
        yield j - i + 1, two
        
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
    d = abs(d)
    if d < 0.147:
        label = 'negligible'
    elif d < 0.33: 
        label = 'small'
    elif d < 0.474:
        label = 'medium'
    else:
        label = 'large'
    
    return d, label


def compare_distributions(a, b):
    """
    Test for a < b using Mann Whitney U and Cliff's delta. 
    Return score, p-value, Cliff's delta, label.
    """
    score, pvalue = scipy.stats.mannwhitneyu(a, b, alternative='less')
    d, label = cliffsDelta(a, b)
    return score, pvalue, d, label
