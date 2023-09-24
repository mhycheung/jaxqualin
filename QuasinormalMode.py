import qnm
import jax.numpy as jnp
import numpy as np
import pykerr
from utils import *
import itertools

class mode_free:
    
    def __init__(self, lmnx, s = -2):
        self.spinseq_list = []
        self.spinseq_list_neg_a = []
        lmnx_actual = []
        self.retro_mode_fac_x = []
        self.lmnx_retro = lmnx
        if lmnx != "constant":
            if isinstance(lmnx, str):
                lmnx = str_to_lmnx(lmnx)
                self.lmnx_retro = str_to_lmnx(self.lmnx_retro)
            for lmn in lmnx:
                l, m, n = tuple(lmn)
                if l < 0:
                    l = -l
                    self.retro_mode_fac_x.append(-1)
                    self.spinseq_list.append(qnm.modes_cache(s=s, l=l, m=-m, n=n))
                    self.spinseq_list_neg_a.append(qnm.modes_cache(s=s, l=l, m=m, n=n))
                else:
                    self.retro_mode_fac_x.append(1)
                    self.spinseq_list.append(qnm.modes_cache(s=s, l=l, m=m, n=n))
                    self.spinseq_list_neg_a.append(qnm.modes_cache(s=s, l=l, m=-m, n=n))
                lmnx_actual.append([l, m, n])
            self.lmnx = lmnx_actual
        else:
            self.lmnx = "constant"

    def init_lmnx_retro(self):
        lmnx = self.lmnx
        self.lmnx_retro = lmnx
        self.retro_mode_fac_x = []
        lmnx_actual = []
        if lmnx != "constant":
            if isinstance(lmnx, str):
                lmnx = str_to_lmnx(lmnx)
                self.lmnx_retro = str_to_lmnx(self.lmnx_retro)
            for lmn in lmnx:
                l, m, n = tuple(lmn)
                if l < 0:
                    l = -l
                    self.retro_mode_fac_x.append(-1)
                else:
                    self.retro_mode_fac_x.append(1)
                lmnx_actual.append([l, m, n])
            self.lmnx = lmnx_actual
        
    def fix_mode(self, M, a, retro = False):
        if a > 0.99:
            a = 0.99
        elif a < -0.99:
            a = -0.99
        self.omegar = 0
        self.omegai = 0
        if self.lmnx != "constant":
            if retro:
                retrofac = -1
            else:
                retrofac = 1
            if a < 0:
                spinseq_list = self.spinseq_list_neg_a
            else:
                spinseq_list = self.spinseq_list
            for i, (lmn, spinseq) in enumerate(zip(self.lmnx, spinseq_list)):
                l, m, n = tuple(lmn)
                omega, _, _ = spinseq(a=np.abs(a))
                self.omegar += self.retro_mode_fac_x[i]*jnpsign0(m) * retrofac * jnp.real(omega) / M
                self.omegai += jnp.imag(omega) / M
        self.omega = self.omegar + 1.j * self.omegai
        self.M = M
        self.a = a
        
    def string(self):
        if self.lmnx == "constant":
            return "constant"
        lmnstrings = []
        if not hasattr(self, 'lmnx_retro'):
            self.init_lmnx_retro()
        for lmn in self.lmnx_retro:
            l, m, n = tuple(lmn)
            lmnstrings.append(f"{l}.{m}.{n}")
        return 'x'.join(lmnstrings)

    def tex_string(self):
        if self.lmnx == "constant":
            return r"constant"
        lmnstrings = []
        for lmn in self.lmnx_retro:
            l, m, n = tuple(lmn)
            if l < 0:
                lmnstrings.append(f"r{-l}.{m}.{n}")
            else:
                lmnstrings.append(f"{l}.{m}.{n}")
        lmnx_string = 'x'.join(lmnstrings)
        _string = '$' + lmnx_string + '$'
        _tex_string = _string.replace('x', r" \! \times \! ")
        _tex_string = _tex_string.replace('-', r" \! - \! ")
        _tex_string = _tex_string.replace('.', r"{,}")
        return _tex_string

    def is_overtone(self):
        if self.lmnx == "constant":
            return False
        else:
            for lmn in self.lmnx:
                l, m, n = tuple(lmn)
                if n > 0:
                    return True

    def sum_lm(self):
        l_sum = 0
        m_sum = 0
        if self.lmnx != "constant":
            for lmn in self.lmnx:
                l, m, n = tuple(lmn)
                l_sum += l
                m_sum += m
        return l_sum, m_sum

class mode(mode_free):
    
    def __init__(self, lmnx, M, a, retro = False, s = -2):
        super().__init__(lmnx, s=s)
        super().fix_mode(M, a, retro = retro)
        self.M = M
        self.a = a
        self.retro = retro

def tex_string_physical_notation(mode):
    if mode.lmnx == "constant":
        return r"constant"
    lmnstrings = []
    if hasattr(mode, 'lmnx_retro'):
        lmnx = mode.lmnx_retro
    else:
        lmnx = mode.lmnx
    for lmn in lmnx:
        l, m, n = tuple(lmn)
        if l < 0:
            lmnstrings.append(f"{-l}.{-m}.{n}")
        elif m < 0:
            lmnstrings.append(f"r{l}.{-m}.{n}")
        else:
            lmnstrings.append(f"{l}.{m}.{n}")
    lmnx_string = 'x'.join(lmnstrings)
    _string = '$' + lmnx_string + '$'
    _tex_string = _string.replace('x', r" \! \times \! ")
    _tex_string = _tex_string.replace('-', r" \! - \! ")
    _tex_string = _tex_string.replace('.', r"{,}")
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
    lmnxstrings = sorted(longstring.split('_'))
    return '_'.join(lmnxstrings)


def lmnxs_to_qnms(lmnxs, M, a, **kwargs):
    qnms = []
    for lmnx in lmnxs:
        qnms.append(mode(lmnx, M, a, **kwargs))
    return qnms


def lmnxs_to_qnms_free(lmnxs, **kwargs):
    qnms = []
    for lmnx in lmnxs:
        qnms.append(mode_free(lmnx, **kwargs))
    return qnms

def long_str_to_qnms(longstring, M, a, **kwargs):
    if longstring == '':
        return []
    lmnxs = long_str_to_lmnxs(longstring)
    return lmnxs_to_qnms(lmnxs, M, a, **kwargs)


def mode_list(mode_list, M, a, **kwargs):
    long_str = '_'.join(mode_list)
    return long_str_to_qnms(long_str, M, a, **kwargs)


def long_str_to_qnms_free(longstring, **kwargs):
    lmnxs = long_str_to_lmnxs(longstring)
    return lmnxs_to_qnms_free(lmnxs, **kwargs)


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


def qnms_to_tex_string_physical_notation(qnms):
    string_list = []
    for qnm in qnms:
        string_list.append(tex_string_physical_notation(qnm))
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


def lmnx_to_string(lmnx):
    lmnstrings = []
    for lmn in lmnx:
        l, m, n = tuple(lmn)
        lmnstrings.append(f"{l}.{m}.{n}")
    return 'x'.join(lmnstrings)


def lmnx_sum_lm(lmnx):
    l_sum = 0
    m_sum = 0
    if lmnx != "constant":
        for lmn in lmnx:
            l, m, n = tuple(lmn)
            l_sum += l
            m_sum += m
    return l_sum, m_sum

def fix_modes(qnms_free_list, M, a):
    for qnm in qnms_free_list:
        qnm.fix_mode(M, a)


def potential_modes(l, m, M, a, relevant_lm_list, retro = False, recoil_n_max = 0,
                    return_lmnx = False, include_r220 = True, include_constant = True):
    potential_lmnx_list = []
    potential_lmnx_list.extend(overtone_modes(l, m, retro = retro))
    potential_lmnx_list.extend(spheroidal_mixing_modes(l, m, retro = retro))
    potential_lmnx_list.extend(recoil_modes(relevant_lm_list, retro = retro, recoil_n_max=recoil_n_max))
    potential_lmnx_list.extend(retrograde_modes_spheroidal(l, m, retro = retro))
    potential_lmnx_list.extend(quadratic_modes_matching_m(m, relevant_lm_list, retro = retro))
    
    for lmnx in potential_lmnx_list:
        lmn_pos = []
        for lmn in lmnx:
            l, m, n = tuple(lmn)
            if m == 0:
                lmn_pos.append([[l, 0, n], [-l, 0, n]])
            else:
                lmn_pos.append([[l, m, n]])
        lmnx_comb = [p for p in itertools.product(*lmn_pos)]
        for lmntup in lmnx_comb:
            lmnx_pos = sorted(list(lmntup))
            if lmnx_pos not in potential_lmnx_list:
                potential_lmnx_list.append(lmnx_pos)
    
    potential_mode_strings = lmnxs_to_string(potential_lmnx_list)
    if include_constant:
        potential_mode_strings.append("constant")
    if include_r220:
        potential_mode_strings.append("-2.2.0")

    if return_lmnx:
        return potential_mode_strings
    else:
        potential_mode_list = lmnxs_to_qnms(
        list(set(potential_mode_strings)), M, a, retro = retro)
        return potential_mode_list


def overtone_modes(l, m, overtone_n_max=7, retro = False):
    if retro:
        retrofac = -1
    else:
        retrofac = 1
    overtone_mode_list = []
    for n in range(overtone_n_max + 1):
        overtone_mode_list.append([[retrofac*l, m, n]])
    return overtone_mode_list


def spheroidal_mixing_modes(l, m, l_max=10, spheroidal_n_max=4, retro = False):
    if retro:
        retrofac = -1
    else:
        retrofac = 1
    spheroidal_mode_list = []
    for n in range(spheroidal_n_max):
        for l in range(max(2, m), l_max):
            spheroidal_mode_list.append([[retrofac*l, m, n]])
    return spheroidal_mode_list


def recoil_modes(relevant_lm_list, recoil_n_max = 0, retro = False):
    if retro:
        retrofac = -1
    else:
        retrofac = 1
    recoil_mode_list = []
    for lm in relevant_lm_list:
        l, m = lm
        for n in range(recoil_n_max + 1):
            recoil_mode_list.append([[retrofac*l, m, n]])
    return recoil_mode_list

def quadratic_modes_matching_m(m, relevant_lm_list_unsorted, quadratic_n_max=1, retro = False):
    if retro:
        retrofac = -1
    else:
        retrofac = 1
    relevant_lm_list = sorted(relevant_lm_list_unsorted)
    quad_mode_list = []
    relevant_length = len(relevant_lm_list)
    for i in range(relevant_length):
        l1, m1 = relevant_lm_list[i]
        for j in range(i + 1):
            l2, m2 = relevant_lm_list[j]
            for n1 in range(quadratic_n_max + 1):
                if i == j:
                    quadratic_n_max2 = n1
                else:
                    quadratic_n_max2 = quadratic_n_max
                for n2 in range(quadratic_n_max2 + 1):
                    if m1 + m2 == m: 
                        lmnx = sorted([[retrofac*l2, m2, n2], [retrofac*l1, m1, n1]])
                        quad_mode_list.append(lmnx)
    return quad_mode_list

# def quadratic_modes(relevant_lm_list_unsorted, quadratic_n_max=1, retro = False):
#     if retro:
#         retrofac = -1
#     else:
#         retrofac = 1
#     relevant_lm_list = sorted(relevant_lm_list_unsorted)
#     quad_mode_list = []
#     relevant_length = len(relevant_lm_list)
#     for i in range(relevant_length):
#         l1, m1 = relevant_lm_list[i]
#         for j in range(i + 1):
#             l2, m2 = relevant_lm_list[j]
#             for n1 in range(quadratic_n_max + 1):
#                 if i == j:
#                     quadratic_n_max2 = n1
#                 else:
#                     quadratic_n_max2 = quadratic_n_max
#                 for n2 in range(quadratic_n_max2 + 1):
#                     lmnx = sorted([[l2, retrofac*m2, n2], [l1, retrofac*m1, n1]])
#                     quad_mode_list.append(lmnx)
#     return quad_mode_list


def retrograde_modes_relevant(relevant_lm_list, retrograde_n_max=3, retro = False):
    if retro:
        retrofac = -1
    else:
        retrofac = 1
    retrograde_mode_list = []
    for lm in relevant_lm_list:
        l, m = lm
        for n in range(retrograde_n_max + 1):
            retrograde_mode_list.append([[-retrofac*l, m, n]])
    return retrograde_mode_list

def retrograde_modes_spheroidal(l, m, retrograde_n_max=3, retrograde_l_max = 10, retro = False):
    if retro:
        retrofac = -1
    else:
        retrofac = 1
    retrograde_mode_list = []
    for l in range(max(2, m), retrograde_l_max + 1):
        for n in range(retrograde_n_max + 1):
            retrograde_mode_list.append([[-retrofac*l, m, n]])
    return retrograde_mode_list


def lower_overtone_present(test_mode, found_modes):
    lmnx = test_mode.lmnx
    lower_overtone_modes = []
    if lmnx == "constant":
        return True
    for i, lmn in enumerate(lmnx):
        l, m, n = tuple(lmn)
        if n > 0:
            lower_overtone_lmnx = sorted(
                filter(None, lmnx[:i] + [[l, m, n - 1]] + lmnx[i + 1:]))
            lower_overtone_modes.append(lower_overtone_lmnx)
    if len(lower_overtone_modes) == 0:
        return True
    else:
        for mode in lmnxs_to_string(lower_overtone_modes):
            if mode not in qnms_to_string(found_modes):
                return False
    return True

def lower_l_mode_present(l, m, relevant_lm_list, test_mode, found_modes):
    lmnx = test_mode.lmnx
    if lmnx == "constant":
        return True
    if len(lmnx) >= 2:
        return True
    l_test, m_test, n_test = tuple(lmnx[0])
    if l_test == l and m_test == m:
        return True
    if m_test != m or l_test == m_test or l_test == 2:
        return True
    if (l_test, m_test) in relevant_lm_list:
        return True
    if l_test > l:
        lower_l_mode_lmnx = [[l_test-1, m_test, n_test]]
    else:
        lower_l_mode_lmnx = [[l_test+1, m_test, n_test]]
    if lmnx_to_string(lower_l_mode_lmnx) not in qnms_to_string(found_modes):
        return False
    return True

def sort_lmnx(lmnx_in):
    lmnx = sorted(lmnx_in)
    return lmnx

def first_n_overtones_string(l, m, n):
    strings = [f"{l}.{m}.{i}" for i in range(n+1)]
    return "_".join(strings)
               
def qnm_string_m_reverse(str):
    if str == 'constant':
        return 'constant'
    lmnx = str_to_lmnx(str)
    for lmn in lmnx:
        if lmn[1] == -99:
            lmn[1] = 0
        elif lmn[1] == 0:
            lmn[1] = -99
        else:
            lmn[1] *= -1
    str_out = lmnx_to_string(lmnx)
    return str_out

def qnm_string_l_reverse(str):
    if str == 'constant':
        return 'constant'
    lmnx = str_to_lmnx(str)
    for lmn in lmnx:
        lmn[0] *= -1
    str_out = lmnx_to_string(lmnx)
    return str_out

def S_retro_fac(iota, af, l, m, n, phi = 0.):
    S = pykerr.spheroidal(iota, af, l, m, n, phi=phi)
    S_n = pykerr.spheroidal(iota, af, l, -m, n, phi=phi)
    return np.abs(S_n)/np.abs(S)

def S_retro_fac_complex(iota, af, l, m, n, phi = 0.):
    S = pykerr.spheroidal(iota, af, l, m, n, phi=phi)
    S_n = pykerr.spheroidal(iota, af, l, -m, n, phi=phi)
    return S_n/S

def S_retro_phase_diff(iota, af, l, m, n, phi = 0.):
    S = pykerr.spheroidal(iota, af, l, m, n, phi=phi)
    S_n = pykerr.spheroidal(iota, af, l, -m, n, phi=phi)
    if l % 2 != 0:
        return np.angle(S_n) - np.angle(S) + np.pi
    return np.angle(S_n) - np.angle(S)