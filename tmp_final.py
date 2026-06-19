import numpy as np
import grassmann as G
from grassmann.lower._dims import dim_of

def key(v): return f"{v.node.name}_{v.node.uid}"

def gram(dt, a, b, J, Jstored, tdim):
    prog=G.compile(dt)
    K=np.zeros((tdim,tdim))
    for m in range(tdim):
        for n in range(tdim):
            em=np.zeros(tdim);em[m]=1.;en=np.zeros(tdim);en[n]=1.
            K[m,n]=float(np.asarray(prog.run({key(a):em,key(b):en,key(J):Jstored})["result"]).reshape(()))
    return K

def make(tdim, edim):
    Rt=G.space("Rt",tdim); Re=G.space("Re",edim)
    J=G.var("J", Rt>>Re, props=G.inj)          # runtime Jacobian, asserted injective
    g    = G.compose(G.dagger(J), J)           # immersed metric term  J^T J  : Rt ⊸ Rt
    T    = G.with_inner(Rt,        g)           # TANGENT  space (metric J^T J)
    Tstar= G.with_inner(G.dual(Rt), G.pinv(g)) # COTANGENT space (metric (J^T J)^-1)
    return Rt,Re,J,g,T,Tstar

cases=[("identity 2x2",2,2,np.eye(2)),
       ("skew 2x2",2,2,np.array([[2.,0.],[1.,3.]])),
       ("sub-ambient 2->3",2,3,np.array([[1.,0.],[0.,1.],[1.,1.]]))]

print("=== dims ===")
Rt,Re,J,g,T,Tstar=make(2,3)
print("dim_of(T.node)     =", dim_of(T.node),     " (expect tdim=2)")
print("dim_of(Tstar.node) =", dim_of(Tstar.node), " (expect tdim=2)")
print("J space:", repr(J.node.space))

allok=True
print("\n=== TANGENT  <a,b>_T  -> J^T J ===")
for name,td,ed,Jm in cases:
    Rt,Re,J,g,T,Tstar=make(td,ed)
    a=G.var("a",T); b=G.var("b",T)
    K=gram(G.dot(a,b),a,b,J,Jm.T,td); exp=Jm.T@Jm
    ok=np.allclose(K,exp); allok&=ok
    print(f"[{name}] MATCH={ok}\n{K}")

print("\n=== COTANGENT <a,b>_T*  -> (J^T J)^-1 ===")
for name,td,ed,Jm in cases:
    Rt,Re,J,g,T,Tstar=make(td,ed)
    a=G.var("a",Tstar); b=G.var("b",Tstar)
    K=gram(G.dot(a,b),a,b,J,Jm.T,td); exp=np.linalg.inv(Jm.T@Jm)
    ok=np.allclose(K,exp); allok&=ok
    print(f"[{name}] MATCH={ok}\n{K}")

print("\nALL MATCH:", allok)
