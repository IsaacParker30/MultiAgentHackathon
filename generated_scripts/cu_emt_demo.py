from ase.build import bulk
from ase.calculators.emt import EMT

atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
atoms.calc = EMT()
energy = atoms.get_potential_energy()
print(f"Total energy: {energy:.6f} eV")
