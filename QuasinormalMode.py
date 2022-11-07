import qnm 
import jax.numpy as jnp

class mode:
    def __init__(self, lmnx, M, a, retro = False, s = -2):
        self.omegar = 0
        self.omegai = 0
        if lmnx != "constant":
            if isinstance(lmnx, str):
                lmnx = str_to_lmnx(lmnx)
            if retro == True:
                retrofac = -1
            else:
                retrofac = 1
            for lmn in lmnx:
                l, m, n = tuple(lmn)
                spinseq = qnm.modes_cache(s=s,l=l,m=m,n=n)
                omega, _, _ = spinseq(a=a)
                self.omegar += jnp.sign(m)*retrofac*jnp.real(omega)/M
                self.omegai += jnp.imag(omega)/M
        self.lmnx = lmnx
        self.omega = self.omegar + 1.j*self.omegai
        self.M = M
        self.a = a
        
    def string(self):
        if self.lmnx == "constant":
            return "constant"
        _lmnstrings = []
        for _lmn in self.lmnx:
            _l, _m, _n = tuple(_lmn)
            _lmnstrings.append(f"{_l}.{_m}.{_n}")
        return 'x'.join(_lmnstrings)
    
    def tex_string(self):
        if self.lmnx == "constant":
            return r"constant"
        _string = '$' + self.string() + '$'
        _tex_string = _string.replace('x', r" \! \times \! ")
        _tex_string = _tex_string.replace('-', r" \! - \! ")
        return _tex_string
    
def str_to_lmnx(lmnxstring):
    if lmnxstring == "constant":
        return "constant"
    lmnx = []
    lmnstrings = lmnxstring.split('x')
    for lmnstring in lmnstrings:
        lmnchars = lmnstring.split('.')
        lmn = list(map(int, lmnchars))
        lmnx.append(lmn)
    return lmnx

def long_str_to_lmnxs(longstring):
    lmnxs = []
    lmnxstrings = longstring.split('_')
    for lmnxstring in lmnxstrings:
        lmnxs.append(str_to_lmnx(lmnxstring))
    return lmnxs

def long_str_to_strs(longstring):
    return longstring.split('_')

def str_list_sort(str_list):
    str_list.sort()
    return str_list

def long_str_sort(longstring):
    lmnxstrings = longstring.split('_')
    lmnxstrings.sort()
    return '_'.join(lmnxstrings)
    
def lmnxs_to_qnms(lmnxs, M, a):
    qnms = []
    for lmnx in lmnxs:
        qnms.append(mode(lmnx, M, a))
    return qnms

def long_str_to_qnms(longstring, M, a):
    lmnxs = long_str_to_lmnxs(longstring)
    return lmnxs_to_qnms(lmnxs, M, a)

def qnms_to_string(qnms):
    string_list = []
    for qnm in qnms:
        string_list.append(qnm.string())
    return string_list

def qnms_to_tex_string(qnms):
    string_list = []
    for qnm in qnms:
        string_list.append(qnm.tex_string())
    return string_list

def qnms_to_lmnxs(qnms):
    lmnxs_list = []
    for qnm in qnms:
        lmnxs_list.append(qnm.lmnx())
    return lmnxs_list

def lmnxs_to_string(lmnxs):
    string_list = []
    for lmnx in lmnxs:
        lmnstrings = []
        for lmn in lmnx:
            l, m, n = tuple(lmn)
            lmnstrings.append(f"{l}.{m}.{n}")
        string_list.append('x'.join(lmnstrings))
    return string_list

def potential_modes(l, m, M, a, relevant_lm_list):
    potential_lmnx_list = []
    potential_lmnx_list.extend(overtone_modes(l,m))
    potential_lmnx_list.extend(spheroidal_mixing_modes(l,m))
    potential_lmnx_list.extend(recoil_modes(relevant_lm_list))
    potential_lmnx_list.extend(retrograde_modes(relevant_lm_list))
    potential_lmnx_list.extend(quadratic_modes(relevant_lm_list))
    potential_mode_strings = lmnxs_to_string(potential_lmnx_list)
    potential_mode_strings.append("constant")
    potential_mode_list = lmnxs_to_qnms(list(set(potential_mode_strings)), M, a)
    return potential_mode_list
    
def overtone_modes(l, m, overtone_n_max = 7):
    overtone_mode_list = []
    for n in range(overtone_n_max + 1):
        overtone_mode_list.append([[l,m,n]])
    return overtone_mode_list

def spheroidal_mixing_modes(l, m,l_max = 10, spheroidal_n_max = 4):
    spheroidal_mode_list = []
    for n in range(spheroidal_n_max):
        for l in range(max(2, m), l_max):
            spheroidal_mode_list.append([[l,m,n]])
    return spheroidal_mode_list
    
def recoil_modes(relevant_lm_list, recoil_n_max = 3):
    recoil_mode_list = []
    for lm in relevant_lm_list:
        l, m = lm
        for n in range(recoil_n_max + 1):
            recoil_mode_list.append([[l, m, n]])
    return recoil_mode_list
            
def quadratic_modes(relevant_lm_list_unsorted, quadratic_n_max = 1):
    relevant_lm_list = relevant_lm_list_unsorted
    relevant_lm_list.sort()
    quad_mode_list = []
    relevant_length = len(relevant_lm_list)
    for i in range(relevant_length):
        l1, m1 = relevant_lm_list[i]
        for j in range(i+1):
            l2, m2 = relevant_lm_list[j]
            for n1 in range(quadratic_n_max+1):
                if i == j:
                    quadratic_n_max2 = n1
                else:
                    quadratic_n_max2 = quadratic_n_max
                for n2 in range(quadratic_n_max2+1):
                    lmnx = [[l2,m2,n2],[l1,m1,n1]]
                    lmnx.sort()
                    quad_mode_list.append(lmnx)
    return quad_mode_list

def retrograde_modes(relevant_lm_list, retrograde_n_max = 3):
    retrograde_mode_list = []
    for lm in relevant_lm_list:
        l, m = lm
        for n in range(retrograde_n_max + 1):
            retrograde_mode_list.append([[l, -m, n]])
    return retrograde_mode_list

def lower_overtone_present(test_mode, found_modes):
    lmnx = test_mode.lmnx
    lower_overtone_modes = []
    if lmnx == "constant":
        return True
    for i, lmn in enumerate(lmnx):
        l, m, n = tuple(lmn)
        if n > 0:
            lower_overtone_lmnx = list(filter(None, lmnx[:i] + [[l, m, n-1]] + lmnx[i+1:]))
            lower_overtone_lmnx.sort()
            lower_overtone_modes.append(lower_overtone_lmnx)
    if len(lower_overtone_modes) == 0:
        return True
    else:
        for mode in lmnxs_to_string(lower_overtone_modes):
            if mode not in qnms_to_string(found_modes):
                return False
    return True
            
def sort_lmnx(lmnx_in):
    lmnx = lmnx_in
    lmnx.sort()
    return lmnx
    