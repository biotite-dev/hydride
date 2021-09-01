# This source code is part of the Hydride package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "hydride"
__author__ = "Patrick Kunzmann"
__all__ = ["estimate_amino_acid_charges"]

import numpy as np
import biotite.structure as struc


PK_NH3 = {
    "ALA" :  9.71,
    "ARG" :  9.00,
    "ASN" :  8.73,
    "ASP" :  9.66,
    "CYS" : 10.28,
    "GLU" :  9.58,
    "GLN" :  9.00,
    "GLY" :  9.58,
    "HIS" :  9.09,
    "ILE" :  9.60,
    "LEU" :  9.58,
    "LYS" :  9.16,
    "MET" :  9.08,
    "PHE" :  9.09,
    "PRO" : 10.47,
    "SER" :  9.05,
    "THR" :  8.96,
    "TRP" :  9.34,
    "TYR" :  9.04,
    "VAL" :  9.52,
}

PK_COOH = {
    "ALA" : 2.33,
    "ARG" : 2.03,
    "ASN" : 2.16,
    "ASP" : 1.95,
    "CYS" : 1.91,
    "GLU" : 2.16,
    "GLN" : 2.18,
    "GLY" : 2.34,
    "HIS" : 1.70,
    "ILE" : 2.26,
    "LEU" : 2.32,
    "LYS" : 2.15,
    "MET" : 2.16,
    "PHE" : 2.18,
    "PRO" : 1.95,
    "SER" : 2.13,
    "THR" : 2.20,
    "TRP" : 2.38,
    "TYR" : 2.24,
    "VAL" : 2.27,
}

# pK values for side chains
# The second value gives the expected charge if pH is less than pK
PK_SIDE_CHAIN = {
    ("ARG", "NH2") : (12.10, 1),
    ("ASP", "OD2") : ( 3.71, 0),
    ("CYS",  "SG") : ( 8.14, 0),
    ("GLU", "OE2") : ( 4.15, 0),
    ("HIS", "ND1") : ( 6.04, 1),
    ("LYS",  "NZ") : (10.67, 1),
    ("TYR",  "OH") : (10.10, 0),
}


def estimate_amino_acid_charges(atoms, ph):
    """
    Estimate the charge of heavy atoms in peptides based on the
    protonation state of the free amino acid [1]_.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The atoms to calculate the charges for.
    ph : float
        The charges are estimated based on this *pH* value.
    
    Returns
    -------
    charges : ndarray, shape=(n,), dtype=int
        The estimated charges.
        0 for all atoms that are not part of an amino acid.
    
    References
    ----------
    
    .. [1] DR Lide,
       "CRC Handbook of Chemistry and Physics."
       CRC Press, (2003).
    """
    charges_nh3 = {
        res_name : 1 if ph < pk else 0
        for res_name, pk in PK_NH3.items()
    }
    charges_cooh = {
        res_name : 0 if ph < pk else -1
        for res_name, pk in PK_COOH.items()
    }
    charges_side_chain = {
        key : charge if ph < pk else charge - 1
        for key, (pk, charge) in PK_SIDE_CHAIN.items()
    }

    atom_charges = np.zeros(atoms.array_length(), dtype=int)
    
    # Charges for termini
    amino_acid_mask = struc.filter_amino_acids(atoms)
    amino_indices   = np.where((atoms.atom_name ==   "N") & amino_acid_mask)[0]
    carboxy_indices = np.where((atoms.atom_name == "OXT") & amino_acid_mask)[0]
    chain_starts = struc.get_chain_starts(atoms, add_exclusive_stop=True)
    for i in range(len(chain_starts) -1):
        start = chain_starts[i]
        stop  = chain_starts[i+1]
        chain_amino_indices = amino_indices[
            (amino_indices >= start) & (amino_indices < stop)
        ]
        chain_carboxy_indices = carboxy_indices[
            (carboxy_indices >= start) & (carboxy_indices < stop)
        ]
        if len(chain_amino_indices) > 0:
            amino_i = np.min(chain_amino_indices)
            atom_charges[amino_i] = charges_nh3[atoms.res_name[amino_i]]
        if len(chain_carboxy_indices) > 0:
            carboxy_i = np.max(chain_carboxy_indices)
            atom_charges[carboxy_i] = charges_cooh[atoms.res_name[carboxy_i]]
    
    # Charges for other heavy atoms
    for (res_name, atom_name), charge in charges_side_chain.items():
        mask  = (atoms.res_name == res_name) & (atoms.atom_name == atom_name)
        atom_charges[mask] = charge

    return atom_charges