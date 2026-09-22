import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import statistics, market_maker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS
def run(days=25, sessions=16):
    pnls=[]; bank=win=0
    for p in SCENARIOS.values():
        for inf in (0.2,0.35,0.5):
            for i in range(sessions):
                r=FieldSession(p,seed=1000+i*37,num_days=days,history_days=[60,150,400][i%3],
                               informed_fraction=inf,competitors=FIELD).run()
                pnls.append(float(r["all"]["SUBJECT"])); bank+=r["bankrupt"]; win+=r["rank"]==1
    return pnls,bank,win
base_v=market_maker._BASE_HALF_SPREAD
base,_,_=run()
print(f"\n{len(base)} paired sessions/cell; incumbent _BASE_HALF_SPREAD={base_v}\n")
print(f"{'spread':>8}{'mean':>9}{'median':>9}{'p10':>8}{'worst':>9}{'>0':>7}{'win%':>7}{'bank':>6}{'d':>9}{'t':>7}")
print("-"*79)
for s in (0.008,0.012,0.018,0.025,0.035):
    market_maker._BASE_HALF_SPREAD=s
    pnls,bank,win=run()
    d=[a-b for a,b in zip(pnls,base)]; md=statistics.fmean(d)
    e=statistics.stdev(d)/len(d)**0.5 if len(d)>1 else 0
    o=sorted(pnls); pos=sum(1 for v in pnls if v>0)/len(pnls)
    m="  <-- incumbent" if s==base_v else ""
    print(f"{s:>8.3f}{statistics.fmean(pnls):>9.2f}{statistics.median(pnls):>9.2f}"
          f"{o[int(0.1*(len(o)-1))]:>8.1f}{min(pnls):>9.1f}{pos:>7.0%}{win/len(pnls):>7.0%}{bank:>6}"
          f"{md:>+9.2f}{(md/e if e else 0):>7.2f}{m}",flush=True)
market_maker._BASE_HALF_SPREAD=base_v
