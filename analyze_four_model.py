"""Four-model test analysis (deep vs R vs C vs C_trunc). Run after the key hash was verified.
Primary: deep vs R pooled over P1+P2, two-sided Fisher. Everything else exploratory."""
import csv, json, hashlib, itertools, math, datetime, sys
from scipy.stats import fisher_exact
OUT = 'four_model_analysis'
KEYF, HASHF = 'protocol/four_model_key.json', 'protocol/four_model_key.sha256'
h = hashlib.sha256(open(KEYF, 'rb').read()).hexdigest()
exp = open(HASHF).read().split()[0]
assert h == exp, 'key hash mismatch'
key = {c: v['model'] for c, v in json.load(open(KEYF))['key'].items()}
order = {r['trial_id']: r for r in csv.DictReader(open('protocol/four_model_trial_order.csv'))}
outs = list(csv.DictReader(open('results/four_model_outcomes.csv')))
final = {}
for r in outs:                       # last non-interrupted attempt per trial
    if r['outcome'] != 'interrupted':
        final[r['trial_id']] = r
assert len(final) == 64, len(final)
MODELS = ['deep', 'R', 'C', 'C_trunc']
rows = []
for t, r in sorted(final.items()):
    meta = json.load(open(f"results/four_model_raw/{t}_a{r['attempt']}/meta.json"))
    rows.append(dict(trial=t, pos=order[t]['position_id'], model=key[order[t]['model_code']],
                     ok=int(r['outcome'] == 'success'), ftype=r['failure_type'], note=r['note'],
                     block=(int(t[1:]) - 1) // 4 + 1, ts=float(r['timestamp_unix']),
                     early=meta['early_end_key'], dur=meta['ended_unix'] - meta['started_unix'],
                     gray=meta.get('first_frame_table_gray'), meta=meta))

def newcombe(k1, n1, k2, n2, z=1.96):
    def wil(k, n):
        p = k / n; d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d; w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return c - w, c + w
    l1, u1 = wil(k1, n1); l2, u2 = wil(k2, n2); p1, p2 = k1 / n1, k2 / n2; d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)

def cnt(sel): s = [r for r in rows if sel(r)]; return sum(r['ok'] for r in s), len(s)
lines = []
def P(s=''): print(s); lines.append(s)
P('# Four-model test results'); P(f'key hash verified: {h}'); P()
P('## Success by model x position')
P('| model | P1 | P2 | total |'); P('|---|---|---|---|')
tab = {}
for m in MODELS:
    a = cnt(lambda r: r['model'] == m and r['pos'] == 'P1'); b = cnt(lambda r: r['model'] == m and r['pos'] == 'P2')
    t = cnt(lambda r: r['model'] == m); tab[m] = (a, b, t)
    P(f'| {m} | {a[0]}/{a[1]} | {b[0]}/{b[1]} | {t[0]}/{t[1]} ({100*t[0]/t[1]:.0f}%) |')
P(); P('## Pairwise (two-sided Fisher, pooled P1+P2; Newcombe 95% CI of difference)')
P('| comparison | rates | diff (pp) | CI (pp) | p | role |'); P('|---|---|---|---|---|---|')
for a, b in itertools.combinations(MODELS, 2):
    (k1, n1), (k2, n2) = tab[a][2], tab[b][2]
    p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]; lo, hi = newcombe(k1, n1, k2, n2)
    role = 'PRIMARY' if (a, b) == ('deep', 'R') else ('secondary' if (a, b) in [('deep', 'C'), ('R', 'C')] else 'exploratory')
    P(f'| {a} vs {b} | {k1}/{n1} vs {k2}/{n2} | {100*(k1/n1-k2/n2):+.0f} | [{100*lo:+.0f}, {100*hi:+.0f}] | {p:.4f} | {role} |')
P(); P('## Per position pairwise (exploratory; 8 vs 8)')
for pos in ['P1', 'P2']:
    for a, b in itertools.combinations(MODELS, 2):
        k1, n1 = cnt(lambda r: r['model'] == a and r['pos'] == pos); k2, n2 = cnt(lambda r: r['model'] == b and r['pos'] == pos)
        p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
        P(f'- {pos} {a} {k1}/{n1} vs {b} {k2}/{n2}: p={p:.3f}')
P(); P('## Failure types (counts)')
ft = sorted({r['ftype'] for r in rows if not r['ok']})
P('| model | ' + ' | '.join(ft) + ' |'); P('|---|' + '---|' * len(ft))
for m in MODELS:
    P(f'| {m} | ' + ' | '.join(str(sum(1 for r in rows if r['model'] == m and not r['ok'] and r['ftype'] == f)) for f in ft) + ' |')
P(); P('## Failure notes by model')
for m in MODELS:
    P(f'- {m}: ' + '; '.join(f"{r['trial']}({r['pos']},{r['ftype']},{r['note']})" for r in rows if r['model'] == m and not r['ok']))
P(); P('## Time order: successes per 8-block half / block')
for m in MODELS:
    P(f'- {m}: ' + ' '.join(f"b{r['block']}{'o' if r['ok'] else 'x'}" for r in sorted([r for r in rows if r['model']==m], key=lambda r: r['trial'])))
P(); P('## Date factor'); 
days = sorted({datetime.datetime.fromtimestamp(r['ts']).strftime('%Y-%m-%d') for r in rows}); P(f'days: {days}')
P(); P('## Early end / outside window')
P(f"outside_time_window trials: {[r['trial'] for r in rows if r['meta'].get('outside_time_window')]}")
open(f'{OUT}/summary.md', 'w').write('\n'.join(lines) + '\n')
json.dump([{k: v for k, v in r.items() if k != 'meta'} for r in rows], open(f'{OUT}/trials_decoded.json', 'w'), indent=1)
