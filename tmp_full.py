import numpy as np
import grassmann as G
from grassmann.core import space as S

def key(v): return f"{v.node.name}_{v.node.uid}"

def gram(dot_term, a, b, J, Jstored, tdim):
    prog = G.compile(dot_term)
    K = np.zeros((tdim, tdim))
    for m in range(tdim):
        for n in range(tdim):
            em=np.zeros(tdim); em[m]=1.0; en=np.zeros(tdim); en[n]=1.0
            out=prog.run({key(a):em, key(b):en, key(J):Jstored})["result"]
            K[m,n]=float(np.asarray(out).reshape(()))
    return K

def tangent_metric(tdim, edim, Jmath):
    """T = WithInner(R^tdim, J^T J) with J a runtime map var. Returns Gram K."""
    Rt=G.space("Rt",tdim); Re=G.space("Re",edim)
    J=G.var("J", Rt>>Re)                  # map var, runtime input
    g=G.compose(G.dagger(J), J)           # metric TERM = J^T J : Rt* ⊗ Rt
    T=G.with_inner(Rt, g)                  # immersed-metric tangent space
    a=G.var("a", T); b=G.var("b", T)
    dt=G.dot(a,b)
    Jstored = Jmath.T                      # Grassmann stores (dom,cod) = transpose of math Jacobian
    return gram(dt, a, b, J, Jstored, tdim)

print("### TANGENT (immersed metric J^T J) ###")
cases = [
    ("identity 2x2", 2,2, np.eye(2)),
    ("skew 2x2",     2,2, np.array([[2.,0.],[1.,3.]])),
    ("sub-ambient 2->3", 2,3, np.array([[1.,0.],[0.,1.],[1.,1.]])),
]
for name,tdim,edim,Jmath in cases:
    K=tangent_metric(tdim,edim,Jmath)
    exp=Jmath.T@Jmath
    ok=np.allclose(K,exp)
    print(f"\n[{name}] J(math)=\n{Jmath}\n K=\n{K}\n expected J^T J=\n{exp}\n MATCH={ok}")
