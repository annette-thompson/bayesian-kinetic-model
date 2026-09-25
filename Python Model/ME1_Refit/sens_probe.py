import sys, numpy as np  # run with python -u to see output live
sys.path.insert(0,'.')
import me1_config as cfg, me1_model as mm, me1_objective as mo
M=mm.ME1Model(); data=mo.load_data(); p0=dict(cfg.PUBLISHED)
base=mo.evaluate(M,p0,data=data)
print(f"baseline  obj1={base.obj1:8.4f} obj2={base.obj2:8.3f} obj3={base.obj3:8.3f} total={base.total:.6g}\n")
print(f"{'param':<6}{'factor':>8}{'obj1':>10}{'obj2':>10}{'obj3':>10}{'total':>12}{'% change':>11}")
for name in cfg.MATLAB_FITTED:
    for fac in (0.5, 2.0):
        p=dict(p0); p[name]=p0[name]*fac
        r=mo.evaluate(M,p,data=data)
        d=(r.total-base.total)/base.total*100
        print(f"{name:<6}{fac:>8.1f}{r.obj1:>10.4f}{r.obj2:>10.3f}{r.obj3:>10.3f}{r.total:>12.6g}{d:>10.2f}%")
