import numpy as np
import grassmann as G

def key(v):
    return f"{v.node.name}_{v.node.uid}"

def dim_of(space_node):
    from grassmann.infer.dims import eval_dim  # may not exist; fallback below
    return eval_dim(space_node)

# ---- helper to extract a gram kernel by running on basis inputs ----
def gram_kernel(dot_term, a, b, jvar, Jnum, tdim):
    K = np.zeros((tdim, tdim))
    for m in range(tdim):
        for n in range(tdim):
            em = np.zeros(tdim); em[m] = 1.0
            en = np.zeros(tdim); en[n] = 1.0
            out = G.run(dot_term, {key(a): em, key(b): en, key(jvar): Jnum})
            K[m, n] = float(np.asarray(out["result"]).reshape(()))
    return K

def build(tdim, edim, inj=True):
    R_t = G.space("Rt", tdim)
    R_e = G.space("Re", edim)
    J = G.var("J", R_t >> R_e, props=G.inj if inj else None)
    T = G.img(J, source=R_t, target=R_e)
    return R_t, R_e, J, T

print("=== probe dim_of on Img ===")
try:
    R_t, R_e, J, T = build(2, 3)
    print("J space repr:", repr(J.node.space))
    print("T node:", repr(T.node))
    # try to find dim
    from grassmann.core import space as S
    print("T is Img:", isinstance(T.node, S.Img))
except Exception as e:
    import traceback; traceback.print_exc()

print("\n=== CASE A: tdim=edim=2, identity J ===")
for orient in ["Jnum", "Jnum.T"]:
    R_t, R_e, J, T = build(2, 2)
    a = G.var("a", T); b = G.var("b", T)
    dt = G.dot(a, b)
    Jnum = np.array([[1.0, 0.0],[0.0,1.0]])
    Jin = Jnum if orient == "Jnum" else Jnum.T
    try:
        K = gram_kernel(dt, a, b, J, Jin, 2)
        print(f"  orient={orient}\n{K}")
    except Exception as e:
        print(f"  orient={orient} FAILED: {type(e).__name__}: {e}")
