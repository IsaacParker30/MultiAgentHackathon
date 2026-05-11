# PySCF geometry optimization for H2O; if geomeTRIC/geomopt is unavailable, this falls back to a single-point energy.
from pyscf import gto, dft

mol = gto.M(
    atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
    basis="def2-svp",
    charge=0,
    spin=0,
)
mf = dft.RKS(mol)
mf.xc = "b3lyp"

try:
    from pyscf.geomopt.geometric_solver import optimize
    mol_eq = optimize(mf)
    mf_eq = dft.RKS(mol_eq)
    mf_eq.xc = "b3lyp"
    energy = mf_eq.kernel()
    print(f"Optimized DFT energy: {energy:.8f} Ha")
except Exception as exc:
    print(f"Geometry optimization unavailable ({exc}); running single-point energy instead.")
    energy = mf.kernel()
    print(f"DFT energy: {energy:.8f} Ha")