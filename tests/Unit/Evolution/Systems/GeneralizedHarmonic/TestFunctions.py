# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Test functions for normal dot fluxes


def spacetime_metric_normal_dot_flux(spacetime_metric, pi, phi, gamma1, gamma2,
                                     lapse, shift, inverse_spatial_metric,
                                     unit_normal):
    return - (1. + gamma1) * np.dot(shift, unit_normal) * spacetime_metric


def pi_normal_dot_flux(spacetime_metric, pi, phi, gamma1, gamma2, lapse, shift,
                       inverse_spatial_metric, unit_normal):
    return - np.dot(shift, unit_normal) * pi  \
        + lapse * np.einsum("ki,k,iab->ab",
                            inverse_spatial_metric, unit_normal, phi) \
           - gamma1 * gamma2 * np.dot(shift, unit_normal) * spacetime_metric


def phi_dot_flux(spacetime_metric, pi, phi, gamma1, gamma2, lapse, shift,
                 inverse_spatial_metric, unit_normal):
    return - np.dot(shift, unit_normal) * phi \
        + lapse * np.einsum("i,ab->iab", unit_normal, pi) \
           - gamma2 * lapse * np.einsum("i,ab->iab", unit_normal,
                                        spacetime_metric)

# End test functions for normal dot fluxes

# Test functions for gauge constraint


def gauge_constraint(gauge_function, spacetime_normal_one_form,
                     spacetime_normal_vector, inverse_spatial_metric,
                     inverse_spacetime_metric, pi, phi):
    # Sums only go over spatial indices of second index of phi
    phi_ija = phi[:, 1:, :]
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    constraint = gauge_function \
        + np.einsum("ij,ija->a", inverse_spatial_metric, phi_ija) \
        + np.einsum("b,ba->a", spacetime_normal_vector, pi) \
        - 0.5 * np.insert(np.einsum("bc,ibc->i", inverse_spacetime_metric, phi),
                          0, np.zeros(phi[0][0][0].shape)) \
        - 0.5 * np.einsum("a,i,bc,ibc->a", spacetime_normal_one_form,
                          spacetime_normal_vector_I,
                          inverse_spacetime_metric, phi) \
        - 0.5 * np.einsum("a,bc,bc->a", spacetime_normal_one_form,
                          inverse_spacetime_metric, pi)
    return constraint

# End test functions for gauge constraint

# Test functions for two-index constraint


def two_index_constraint_term_1_of_11(inverse_spatial_metric, d_phi):
    d_phi_jika = d_phi[:, :, 1:, :]
    return np.einsum("jk,jika->ia", inverse_spatial_metric, d_phi_jika)


def two_index_constraint_term_2_of_11(spacetime_normal_vector,
                                      spacetime_normal_one_form,
                                      inverse_spacetime_metric, d_phi):
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    d_phi_aicd = np.pad(d_phi, ((1, 0), (0, 0), (0, 0), (0, 0)), 'constant')
    term = np.einsum("cd,aicd->ia", inverse_spacetime_metric, d_phi_aicd)
    term += np.einsum("j,a,cd,jicd->ia", spacetime_normal_vector_I,
                      spacetime_normal_one_form, inverse_spacetime_metric,
                      d_phi)
    return -0.5 * term


def two_index_constraint_term_3_of_11(spacetime_normal_vector, d_pi):
    return np.einsum("b,iba->ia", spacetime_normal_vector, d_pi)


def two_index_constraint_term_4_of_11(spacetime_normal_one_form,
                                      inverse_spacetime_metric, d_pi):
    return -0.5 * np.einsum("cd,icd,a->ia", inverse_spacetime_metric, d_pi,
                            spacetime_normal_one_form)


def two_index_constraint_term_5_of_11(d_gauge_function):
    return d_gauge_function


def two_index_constraint_term_6_of_11(spacetime_normal_vector,
                                      spacetime_normal_one_form, phi,
                                      inverse_spacetime_metric):
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    phi_acd = np.pad(phi, ((1, 0), (0, 0), (0, 0)), 'constant')
    term = np.einsum("acd,ief,ce,df->ia", phi_acd, phi,
                     inverse_spacetime_metric,
                     inverse_spacetime_metric)
    term += np.einsum("j,a,jcd,ief,ce,df->ia", spacetime_normal_vector_I,
                      spacetime_normal_one_form, phi, phi,
                      inverse_spacetime_metric,
                      inverse_spacetime_metric)
    return 0.5 * term


def two_index_constraint_term_7_of_11(inverse_spatial_metric, phi,
                                      inverse_spacetime_metric,
                                      spacetime_normal_vector,
                                      spacetime_normal_one_form):
    phi_ike = phi[:, 1:, :]
    return 0.5 * np.einsum("jk,jcd,ike,cd,e,a->ia", inverse_spatial_metric, phi,
                           phi_ike, inverse_spacetime_metric,
                           spacetime_normal_vector,
                           spacetime_normal_one_form)


def two_index_constraint_term_8_of_11(inverse_spatial_metric, phi):
    phi_jma = phi[:, 1:, :]
    phi_ikn = phi[:, 1:, 1:]
    return -1.0 * np.einsum("jk,mn,jma,ikn->ia", inverse_spatial_metric,
                            inverse_spatial_metric, phi_jma, phi_ikn)


def two_index_constraint_term_9_of_11(phi, pi, spacetime_normal_one_form,
                                      inverse_spacetime_metric,
                                      spacetime_normal_vector):
    term = 0.5 * np.einsum("icd,be,a,cb,de->ia", phi, pi,
                           spacetime_normal_one_form, inverse_spacetime_metric,
                           inverse_spacetime_metric)
    term += 0.25 * np.einsum("icd,be,a,be,c,d->ia", phi, pi,
                             spacetime_normal_one_form,
                             inverse_spacetime_metric,
                             spacetime_normal_vector, spacetime_normal_vector)
    return term


def two_index_constraint_term_10_of_11(phi, pi, spacetime_normal_vector,
                                       inverse_spacetime_metric):
    term = -1.0 * np.einsum("icd,ba,c,bd->ia", phi, pi,
                            spacetime_normal_vector, inverse_spacetime_metric)
    term -= 0.5 * np.einsum("icd,ba,c,b,d->ia", phi, pi,
                            spacetime_normal_vector, spacetime_normal_vector,
                            spacetime_normal_vector)
    return term


def two_index_constraint_term_11_of_11(gamma2, spacetime_normal_one_form,
                                       inverse_spacetime_metric,
                                       spacetime_normal_vector,
                                       three_index_constraint):
    term = 0.5 * gamma2 * np.einsum("a,cd,icd->ia",
                                    spacetime_normal_one_form,
                                    inverse_spacetime_metric,
                                    three_index_constraint)
    term -= gamma2 * np.einsum("d,iad->ia",
                               spacetime_normal_vector, three_index_constraint)
    return term


def two_index_constraint(d_gauge_function, spacetime_normal_one_form,
                         spacetime_normal_vector, inverse_spatial_metric,
                         inverse_spacetime_metric, pi, phi, d_pi, d_phi, gamma2,
                         three_index_constraint):
    constraint = two_index_constraint_term_1_of_11(
        inverse_spatial_metric, d_phi)
    constraint += two_index_constraint_term_2_of_11(spacetime_normal_vector,
                                                    spacetime_normal_one_form,
                                                    inverse_spacetime_metric,
                                                    d_phi)
    constraint += two_index_constraint_term_3_of_11(
        spacetime_normal_vector, d_pi)
    constraint += two_index_constraint_term_4_of_11(spacetime_normal_one_form,
                                                    inverse_spacetime_metric,
                                                    d_pi)
    constraint += two_index_constraint_term_5_of_11(d_gauge_function)
    constraint += two_index_constraint_term_6_of_11(spacetime_normal_vector,
                                                    spacetime_normal_one_form,
                                                    phi,
                                                    inverse_spacetime_metric)
    constraint += two_index_constraint_term_7_of_11(inverse_spatial_metric, phi,
                                                    inverse_spacetime_metric,
                                                    spacetime_normal_vector,
                                                    spacetime_normal_one_form)
    constraint += two_index_constraint_term_8_of_11(
        inverse_spatial_metric, phi)
    constraint += two_index_constraint_term_9_of_11(phi, pi,
                                                    spacetime_normal_one_form,
                                                    inverse_spacetime_metric,
                                                    spacetime_normal_vector)
    constraint += two_index_constraint_term_10_of_11(
        phi, pi, spacetime_normal_vector, inverse_spacetime_metric)
    constraint += two_index_constraint_term_11_of_11(gamma2,
                                                     spacetime_normal_one_form,
                                                     inverse_spacetime_metric,
                                                     spacetime_normal_vector,
                                                     three_index_constraint)
    return constraint

# End test functions for two-index constraint

# Test functions for four-index constraint

def four_index_constraint(d_phi):
    e_ijk = np.zeros((3, 3, 3))
    e_ijk[0, 1, 2] = e_ijk[1, 2, 0] = e_ijk[2, 0, 1] = 1.0
    e_ijk[0, 2, 1] = e_ijk[2, 1, 0] = e_ijk[1, 0, 2] = -1.0
    constraint = np.einsum("ijk,jkab->iab", e_ijk, d_phi)
    return constraint

# End test functions for four-index constraint

# Test functions for characteristic speeds
def char_speed_upsi(gamma1, lapse, shift, unit_normal):
    return - (1. + gamma1) * np.dot(shift, unit_normal)


def char_speed_uzero(gamma1, lapse, shift, unit_normal):
    return - np.dot(shift, unit_normal)


def char_speed_uplus(gamma1, lapse, shift, unit_normal):
    return - np.dot(shift, unit_normal) + lapse


def char_speed_uminus(gamma1, lapse, shift, unit_normal):
    return - np.dot(shift, unit_normal) - lapse

# End test functions for characteristic speeds


# Test functions for characteristic fields
def char_field_upsi(gamma2, inverse_spatial_metric, spacetime_metric,
                    pi, phi, normal_one_form):
    return spacetime_metric


def char_field_uzero(gamma2, inverse_spatial_metric, spacetime_metric,
                     pi, phi, normal_one_form):
    normal_vector =\
        np.einsum('ij,j', inverse_spatial_metric, normal_one_form)
    projection_tensor = np.identity(len(normal_vector)) -\
        np.einsum('i,j', normal_one_form, normal_vector)
    return np.einsum('ij,jab->iab', projection_tensor, phi)


def char_field_uplus(gamma2, inverse_spatial_metric, spacetime_metric,
                     pi, phi, normal_one_form):
    normal_vector =\
        np.einsum('ij,j', inverse_spatial_metric, normal_one_form)
    phi_dot_normal = np.einsum('i,iab->ab', normal_vector, phi)
    return pi + 1*phi_dot_normal - (gamma2 * spacetime_metric)


def char_field_uminus(gamma2, inverse_spatial_metric, spacetime_metric,
                      pi, phi, normal_one_form):
    normal_vector =\
        np.einsum('ij,j', inverse_spatial_metric, normal_one_form)
    phi_dot_normal = np.einsum('i,iab->ab', normal_vector, phi)
    return pi - phi_dot_normal - (gamma2 * spacetime_metric)

# Test functions for evolved fields


def evol_field_psi(gamma2, upsi, uzero, uplus, uminus, normal_one_form):
    return upsi


def evol_field_pi(gamma2, upsi, uzero, uplus, uminus, normal_one_form):
    return 0.5 * (uplus + uminus) + gamma2 * upsi


def evol_field_phi(gamma2, upsi, uzero, uplus, uminus, normal_one_form):
    udiff = 0.5 * (uplus - uminus)
    return np.einsum('i,ab->iab', normal_one_form, udiff) + uzero

# End test functions for characteristic fields

# Test functions for F constraint


def f_constraint_term_1_of_25(spacetime_normal_one_form,
                              spacetime_normal_vector,
                              inverse_spacetime_metric, d_pi):
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    d_pi_abc = np.pad(d_pi, ((1, 0), (0, 0), (0, 0)), 'constant')
    term = np.einsum("bc,abc", inverse_spacetime_metric, d_pi_abc)
    term += np.einsum("a,i,bc,ibc", spacetime_normal_one_form,
                      spacetime_normal_vector_I, inverse_spacetime_metric, d_pi)
    return 0.5 * term


def f_constraint_term_2_of_25(inverse_spatial_metric, d_pi):
    d_pi_ija = d_pi[:, 1:, :]
    return -1.0 * np.einsum("ij,ija", inverse_spatial_metric, d_pi_ija)


def f_constraint_term_3_of_25(inverse_spatial_metric, spacetime_normal_vector,
                              d_phi):
    return -1.0 * np.einsum("ij,b,ijba", inverse_spatial_metric,
                            spacetime_normal_vector, d_phi)


def f_constraint_term_4_of_25(spacetime_normal_one_form,
                              inverse_spacetime_metric,
                              inverse_spatial_metric, d_phi):
    return 0.5 * np.einsum("a,bc,ij,ijbc", spacetime_normal_one_form,
                           inverse_spacetime_metric,
                           inverse_spatial_metric, d_phi)


def f_constraint_term_5_of_25(spacetime_normal_one_form, inverse_spatial_metric,
                              d_gauge_function):
    d_gauge_function_ij = d_gauge_function[:, 1:]
    return np.einsum("a,ij,ij", spacetime_normal_one_form,
                     inverse_spatial_metric,
                     d_gauge_function_ij)


def f_constraint_term_6_of_25(spacetime_normal_one_form,
                              spacetime_normal_vector,
                              phi, inverse_spatial_metric,
                              inverse_spacetime_metric):
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    phi_ijb = phi[:, 1:, :]
    phi_ajb = np.pad(phi, ((1, 0), (0, 0), (0, 0)), 'constant')[:, 1:, :]
    term = np.einsum("ajb,jk,kcd,bd,c", phi_ajb, inverse_spatial_metric, phi,
                     inverse_spacetime_metric,
                     spacetime_normal_vector)
    return term + np.einsum("a,i,ijb,jk,kcd,bd,c", spacetime_normal_one_form,
                            spacetime_normal_vector_I, phi_ijb,
                            inverse_spatial_metric, phi,
                            inverse_spacetime_metric, spacetime_normal_vector)


def f_constraint_term_7_of_25(spacetime_normal_one_form,
                              spacetime_normal_vector,
                              phi, inverse_spatial_metric,
                              inverse_spacetime_metric):
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    phi_ijb = phi[:, 1:, :]
    phi_ajb = np.pad(phi, ((1, 0), (0, 0), (0, 0)), 'constant')[:, 1:, :]
    term = np.einsum("ajb,jk,kcd,cd,b", phi_ajb, inverse_spatial_metric, phi,
                     inverse_spacetime_metric,
                     spacetime_normal_vector)
    return -0.5 * (term + np.einsum("a,i,ijb,jk,kcd,cd,b",
                                    spacetime_normal_one_form,
                                    spacetime_normal_vector_I, phi_ijb,
                                    inverse_spatial_metric, phi,
                                    inverse_spacetime_metric,
                                    spacetime_normal_vector))


def f_constraint_term_8_of_25(spacetime_normal_one_form,
                              spacetime_normal_vector,
                              d_gauge_function):
    d_gauge_function_ab = np.pad(
        d_gauge_function, ((1, 0), (0, 0)), 'constant')
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    term = -1.0 * np.einsum("b,ab", spacetime_normal_vector,
                            d_gauge_function_ab)
    return term - np.einsum("a,i,b,ib", spacetime_normal_one_form,
                            spacetime_normal_vector_I,
                            spacetime_normal_vector, d_gauge_function)


def f_constraint_term_9_of_25(inverse_spatial_metric, phi,
                              inverse_spacetime_metric,
                              spacetime_normal_vector):
    return np.einsum("ij,icd,jba,bc,d", inverse_spatial_metric, phi, phi,
                     inverse_spacetime_metric, spacetime_normal_vector)


def f_constraint_term_10_of_25(spacetime_normal_one_form,
                               inverse_spatial_metric, phi,
                               inverse_spacetime_metric):
    phi_imc = phi[:, 1:, :]
    return -0.5 * np.einsum("a,ij,mn,imc,njd,cd", spacetime_normal_one_form,
                            inverse_spatial_metric, inverse_spatial_metric,
                            phi_imc, phi_imc, inverse_spacetime_metric)


def f_constraint_term_11_of_25(spacetime_normal_one_form,
                               inverse_spatial_metric, phi,
                               inverse_spacetime_metric):
    return -0.25 * np.einsum("a,ij,icd,jbe,cb,de", spacetime_normal_one_form,
                             inverse_spatial_metric, phi, phi,
                             inverse_spacetime_metric, inverse_spacetime_metric)


def f_constraint_term_12_of_25(spacetime_normal_one_form, pi,
                               inverse_spacetime_metric):
    return 0.25 * np.einsum("a,cd,be,cb,de", spacetime_normal_one_form, pi, pi,
                            inverse_spacetime_metric, inverse_spacetime_metric)


def f_constraint_term_13_of_25(inverse_spatial_metric, gauge_function, pi):
    gauge_function_i = gauge_function[1:]
    pi_ja = pi[1:, :]
    return -1.0 * np.einsum("ij,i,ja", inverse_spatial_metric,
                            gauge_function_i, pi_ja)


def f_constraint_term_14_of_25(spacetime_normal_vector, inverse_spatial_metric,
                               pi):
    pi_bi = pi[:, 1:]
    pi_ja = pi[1:, :]
    return -1.0 * np.einsum("b,ij,bi,ja", spacetime_normal_vector,
                            inverse_spatial_metric, pi_bi, pi_ja)


def f_constraint_term_15_of_25(inverse_spacetime_metric,
                               spacetime_normal_one_form,
                               spacetime_normal_vector, phi, pi):
    phi_acd = np.pad(phi, ((1, 0), (0, 0), (0, 0)), 'constant')
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    term = np.einsum("acd,c,d,be,be", phi_acd, spacetime_normal_vector,
                     spacetime_normal_vector, pi, inverse_spacetime_metric)
    term += np.einsum("a,i,icd,c,d,be,be", spacetime_normal_one_form,
                      spacetime_normal_vector_I, phi, spacetime_normal_vector,
                      spacetime_normal_vector, pi, inverse_spacetime_metric)
    return -0.25 * term


def f_constraint_term_16_of_25(spacetime_normal_one_form, pi,
                               inverse_spacetime_metric,
                               spacetime_normal_vector):
    return 0.5 * np.einsum("a,cd,be,ce,d,b", spacetime_normal_one_form,
                           pi, pi, inverse_spacetime_metric,
                           spacetime_normal_vector, spacetime_normal_vector)


def f_constraint_term_17_of_25(inverse_spacetime_metric,
                               spacetime_normal_one_form,
                               spacetime_normal_vector, phi, pi):
    phi_acd = np.pad(phi, ((1, 0), (0, 0), (0, 0)), 'constant')
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    term = np.einsum("acd,be,c,b,de", phi_acd, pi, spacetime_normal_vector,
                     spacetime_normal_vector, inverse_spacetime_metric)
    return term + np.einsum("a,i,icd,be,c,b,de", spacetime_normal_one_form,
                            spacetime_normal_vector_I, phi, pi,
                            spacetime_normal_vector, spacetime_normal_vector,
                            inverse_spacetime_metric)


def f_constraint_term_18_of_25(inverse_spatial_metric, phi,
                               spacetime_normal_vector, pi):
    pi_je = pi[1:, :]
    return -1.0 * np.einsum("ij,iba,b,je,e", inverse_spatial_metric, phi,
                            spacetime_normal_vector, pi_je,
                            spacetime_normal_vector)


def f_constraint_term_19_of_25(inverse_spatial_metric, phi,
                               spacetime_normal_vector, pi):
    pi_ja = pi[1:, :]
    return -0.5 * np.einsum("ij,icd,c,d,ja", inverse_spatial_metric, phi,
                            spacetime_normal_vector, spacetime_normal_vector,
                            pi_ja)


def f_constraint_term_20_of_25(inverse_spatial_metric, gauge_function, phi,
                               spacetime_normal_vector):
    gauge_function_i = gauge_function[1:]
    return -1.0 * np.einsum("ij,i,jab,b", inverse_spatial_metric,
                            gauge_function_i, phi, spacetime_normal_vector)


def f_constraint_term_21_of_25(spacetime_normal_one_form,
                               spacetime_normal_vector, phi, gauge_function,
                               inverse_spacetime_metric):
    phi_acd = np.pad(phi, ((1, 0), (0, 0), (0, 0)), 'constant')
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    term = np.einsum("acd,b,bc,d", phi_acd, gauge_function,
                     inverse_spacetime_metric, spacetime_normal_vector)
    return term + np.einsum("a,i,icd,b,bc,d", spacetime_normal_one_form,
                            spacetime_normal_vector_I, phi, gauge_function,
                            inverse_spacetime_metric, spacetime_normal_vector)


def f_constraint_term_22_of_25(gamma2, inverse_spacetime_metric,
                               spacetime_normal_vector, three_index_constraint,
                               spacetime_normal_one_form):
    inverse_spacetime_metric_ID = inverse_spacetime_metric[1:, :]
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    three_index_constraint_acd = np.pad(three_index_constraint,
                                        ((1, 0), (0, 0), (0, 0)), 'constant')
    term = np.einsum("id,ida", inverse_spacetime_metric_ID,
                     three_index_constraint)
    term += np.einsum("i,d,ida", spacetime_normal_vector_I,
                      spacetime_normal_vector, three_index_constraint)
    term -= 0.5 * np.einsum("cd,acd", inverse_spacetime_metric,
                            three_index_constraint_acd)
    term -= 0.5 * np.einsum("a,i,cd,icd", spacetime_normal_one_form,
                            spacetime_normal_vector_I, inverse_spacetime_metric,
                            three_index_constraint)
    return gamma2 * term


def f_constraint_term_23_of_25(spacetime_normal_one_form, pi,
                               inverse_spacetime_metric, gauge_function,
                               spacetime_normal_vector):
    return 0.5 * np.einsum("a,cd,cd,b,b", spacetime_normal_one_form, pi,
                           inverse_spacetime_metric, gauge_function,
                           spacetime_normal_vector)


def f_constraint_term_24_of_25(spacetime_normal_one_form,
                               inverse_spatial_metric, phi, gauge_function,
                               inverse_spacetime_metric):
    phi_ijc = phi[:, 1:, :]
    return -1.0 * np.einsum("a,ij,ijc,d,cd", spacetime_normal_one_form,
                            inverse_spatial_metric, phi_ijc, gauge_function,
                            inverse_spacetime_metric)


def f_constraint_term_25_of_25(spacetime_normal_one_form,
                               inverse_spatial_metric, gauge_function,
                               phi, inverse_spacetime_metric):
    gauge_function_i = gauge_function[1:]
    return 0.5 * np.einsum("a,ij,i,jcd,cd", spacetime_normal_one_form,
                           inverse_spatial_metric, gauge_function_i, phi,
                           inverse_spacetime_metric)


def f_constraint(gauge_function, d_gauge_function, spacetime_normal_one_form,
                 spacetime_normal_vector, inverse_spatial_metric,
                 inverse_spacetime_metric, pi, phi, d_pi, d_phi, gamma2,
                 three_index_constraint):
    constraint = f_constraint_term_1_of_25(spacetime_normal_one_form,
                                           spacetime_normal_vector,
                                           inverse_spacetime_metric, d_pi)
    constraint += f_constraint_term_2_of_25(inverse_spatial_metric, d_pi)
    constraint += f_constraint_term_3_of_25(inverse_spatial_metric,
                                            spacetime_normal_vector, d_phi)
    constraint += f_constraint_term_4_of_25(spacetime_normal_one_form,
                                            inverse_spacetime_metric,
                                            inverse_spatial_metric, d_phi)
    constraint += f_constraint_term_5_of_25(spacetime_normal_one_form,
                                            inverse_spatial_metric,
                                            d_gauge_function)
    constraint += f_constraint_term_6_of_25(spacetime_normal_one_form,
                                            spacetime_normal_vector,
                                            phi, inverse_spatial_metric,
                                            inverse_spacetime_metric)
    constraint += f_constraint_term_7_of_25(spacetime_normal_one_form,
                                            spacetime_normal_vector,
                                            phi, inverse_spatial_metric,
                                            inverse_spacetime_metric)
    constraint += f_constraint_term_8_of_25(spacetime_normal_one_form,
                                            spacetime_normal_vector,
                                            d_gauge_function)
    constraint += f_constraint_term_9_of_25(inverse_spatial_metric, phi,
                                            inverse_spacetime_metric,
                                            spacetime_normal_vector)
    constraint += f_constraint_term_10_of_25(spacetime_normal_one_form,
                                             inverse_spatial_metric, phi,
                                             inverse_spacetime_metric)
    constraint += f_constraint_term_11_of_25(spacetime_normal_one_form,
                                             inverse_spatial_metric, phi,
                                             inverse_spacetime_metric)
    constraint += f_constraint_term_12_of_25(spacetime_normal_one_form, pi,
                                             inverse_spacetime_metric)
    constraint += f_constraint_term_13_of_25(inverse_spatial_metric,
                                             gauge_function, pi)
    constraint += f_constraint_term_14_of_25(spacetime_normal_vector,
                                             inverse_spatial_metric, pi)
    constraint += f_constraint_term_15_of_25(inverse_spacetime_metric,
                                             spacetime_normal_one_form,
                                             spacetime_normal_vector, phi, pi)
    constraint += f_constraint_term_16_of_25(spacetime_normal_one_form, pi,
                                             inverse_spacetime_metric,
                                             spacetime_normal_vector)
    constraint += f_constraint_term_17_of_25(inverse_spacetime_metric,
                                             spacetime_normal_one_form,
                                             spacetime_normal_vector, phi, pi)
    constraint += f_constraint_term_18_of_25(inverse_spatial_metric, phi,
                                             spacetime_normal_vector, pi)
    constraint += f_constraint_term_19_of_25(inverse_spatial_metric, phi,
                                             spacetime_normal_vector, pi)
    constraint += f_constraint_term_20_of_25(inverse_spatial_metric,
                                             gauge_function, phi,
                                             spacetime_normal_vector)
    constraint += f_constraint_term_21_of_25(spacetime_normal_one_form,
                                             spacetime_normal_vector, phi,
                                             gauge_function,
                                             inverse_spacetime_metric)
    constraint += f_constraint_term_22_of_25(gamma2, inverse_spacetime_metric,
                                             spacetime_normal_vector,
                                             three_index_constraint,
                                             spacetime_normal_one_form)
    constraint += f_constraint_term_23_of_25(spacetime_normal_one_form, pi,
                                             inverse_spacetime_metric,
                                             gauge_function,
                                             spacetime_normal_vector)
    constraint += f_constraint_term_24_of_25(spacetime_normal_one_form,
                                             inverse_spatial_metric,
                                             phi, gauge_function,
                                             inverse_spacetime_metric)
    constraint += f_constraint_term_25_of_25(spacetime_normal_one_form,
                                             inverse_spatial_metric,
                                             gauge_function, phi,
                                             inverse_spacetime_metric)
    return constraint

# End test functions for F constraint

# Begin test functions for constraint energy


def constraint_energy(gauge_constraint, f_constraint, two_index_constraint,
                      three_index_constraint, four_index_constraint,
                      inverse_spatial_metric, spatial_metric_determinant,
                      gauge_constraint_multiplier,
                      two_index_constraint_multiplier,
                      three_index_constraint_multiplier,
                      four_index_constraint_multiplier):
    energy = 0.0
    energy += gauge_constraint_multiplier \
        * np.einsum("a,a", gauge_constraint, gauge_constraint)
    energy += two_index_constraint_multiplier \
        * np.einsum("a,a", f_constraint, f_constraint)
    energy += two_index_constraint_multiplier \
        * np.einsum("ia,ja,ij", two_index_constraint, two_index_constraint,
                    inverse_spatial_metric)
    energy += three_index_constraint_multiplier \
        * np.einsum("iab,jab,ij", three_index_constraint,
                    three_index_constraint, inverse_spatial_metric)
    energy += 2.0 * four_index_constraint_multiplier \
        * spatial_metric_determinant \
        * np.einsum("iab,jab,ij", four_index_constraint,
                    four_index_constraint, inverse_spatial_metric)
    return energy

# End test functions for constraint energy
