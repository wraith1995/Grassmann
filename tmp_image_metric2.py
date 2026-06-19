import numpy as np
import grassmann as G
from grassmann.core import space as S

def key(v):
    return f"{v.node.name}_{v.node.uid}"

# -----------------------------------------------------------------
# ROUTE 1: literal image basis via WithInner-free? Try dot on bare Img var,
# feeding projected coords (rank-sized). See what metric comes out.
# -----------------------------------------------------------------
print("=== ROUTE 1: bare Img var dot, dynamic path (feed rank-sized coords) ===")
tdim, edim = 2, 2
R_t = G.space("Rt", tdim); R_e = G.space("Re", edim)
J = G.var("J", R_t >> R_e, props=G.inj)
T = G.img(J, source=R_t, target=R_e)
a = G.var("a", T); b = G.var("b", T)
dt = G.dot(a, b)
try:
    prog = G.compile(dt)
    print("compiled OK; statements:", len(prog.statements))
    Jnum = np.array([[2.0,0.0],[1.0,3.0]])
    for orient,Jin in [("Jnum",Jnum),("Jnum.T",Jnum.T)]:
        K = np.zeros((tdim,tdim))
        for m in range(tdim):
            for n in range(tdim):
                em=np.zeros(tdim);em[m]=1; en=np.zeros(tdim);en[n]=1
                out=prog.run({key(a):em,key(b):en,key(J):Jin})
                K[m,n]=float(np.asarray(out["result"]).reshape(()))
        print(f"  orient={orient} K=\n{K}\n  JtJ=\n{Jin.T@Jin if orient=='Jnum' else (Jin.T@Jin)}")
except Exception as e:
    import traceback; traceback.print_exc()
