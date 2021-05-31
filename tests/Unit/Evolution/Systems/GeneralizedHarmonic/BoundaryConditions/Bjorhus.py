# Distributed under the MIT License.
# See LICENSE.txt for details.

import itertools as it
import numpy as np

import PointwiseFunctions.GeneralRelativity.Christoffel as ch
import PointwiseFunctions.GeneralRelativity.ComputeGhQuantities as gh
import PointwiseFunctions.GeneralRelativity.ProjectionOperators as proj
import PointwiseFunctions.GeneralRelativity.WeylPropagating as wp


def constraint_preserving_bjorhus_corrections_dt_v_psi(
    unit_interface_normal_vector, three_index_constraint, char_speeds):
    return (char_speeds[0] * np.einsum(
        'i,iab->ab', unit_interface_normal_vector, three_index_constraint))


def constraint_preserving_bjorhus_corrections_dt_v_zero(
    unit_interface_normal_vector, four_index_constraint, char_speeds):
    spatial_dim = len(unit_interface_normal_vector)
    result = np.zeros([spatial_dim, 1 + spatial_dim, 1 + spatial_dim])

    if spatial_dim == 2:
        result[0, :, :] += char_speeds[1] * unit_interface_normal_vector[
            1] * four_index_constraint[1, :, :]
        result[1, :, :] += char_speeds[1] * unit_interface_normal_vector[
            0] * four_index_constraint[0, :, :]
    elif spatial_dim == 3:

        def is_even(sequence):
            count = 0
            for i, n in enumerate(sequence, start=1):
                count += sum(n > num for num in sequence[i:])
            return not count % 2

        for p in it.permutations(np.arange(len(unit_interface_normal_vector))):
            sgn = 1 if is_even(p) else -1
            result[p[0], :, :] += (sgn * char_speeds[1] *
                                   unit_interface_normal_vector[p[2]] *
                                   four_index_constraint[p[1], :, :])
    return result


def add_gauge_sommerfeld_terms_to_dt_v_minus(
    gamma2, inertial_coords, incoming_null_one_form, outgoing_null_one_form,
    incoming_null_vector, outgoing_null_vector, projection_Ab,
    char_projected_rhs_dt_v_psi):
    gauge_bc_coeff = 1.
    inertial_radius = np.sum(inertial_coords**2)**0.5
    prefac = (gamma2 - gauge_bc_coeff / inertial_radius)

    t1_ = np.einsum('a,cb,d,cd->ab', incoming_null_one_form, projection_Ab,
                    outgoing_null_vector, char_projected_rhs_dt_v_psi)
    t2_ = np.einsum('b,ca,d,cd->ab', incoming_null_one_form, projection_Ab,
                    outgoing_null_vector, char_projected_rhs_dt_v_psi)
    t3_ = np.einsum('a,b,c,d,cd->ab', incoming_null_one_form,
                    outgoing_null_one_form, incoming_null_vector,
                    outgoing_null_vector, char_projected_rhs_dt_v_psi)
    t4_ = np.einsum('b,a,c,d,cd->ab', incoming_null_one_form,
                    outgoing_null_one_form, incoming_null_vector,
                    outgoing_null_vector, char_projected_rhs_dt_v_psi)
    t5_ = np.einsum('a,b,c,d,cd->ab', incoming_null_one_form,
                    incoming_null_one_form, outgoing_null_vector,
                    outgoing_null_vector, char_projected_rhs_dt_v_psi)
    return prefac * (t1_ + t2_ - t3_ - t4_ - t5_)


def add_constraint_dependent_terms_to_dt_v_minus(
    incoming_null_one_form, outgoing_null_one_form, incoming_null_vector,
    outgoing_null_vector, projection_ab, projection_Ab, projection_AB,
    constraint_char_zero_plus, constraint_char_zero_minus,
    char_projected_rhs_dt_v_minus, char_speeds):
    mu = 0.0  # hard-coded value from SpEC Bbh input file Mu = 0

    t1_ = np.einsum('c,d,a,b,cd->ab', incoming_null_vector,
                    incoming_null_vector, outgoing_null_one_form,
                    outgoing_null_one_form, char_projected_rhs_dt_v_minus)
    t2_ = np.einsum('c,da,b,cd->ab', incoming_null_vector, projection_Ab,
                    outgoing_null_one_form, char_projected_rhs_dt_v_minus)
    t3_ = np.einsum('c,db,a,cd->ab', incoming_null_vector, projection_Ab,
                    outgoing_null_one_form, char_projected_rhs_dt_v_minus)
    t4_ = np.einsum('d,ca,b,cd->ab', incoming_null_vector, projection_Ab,
                    outgoing_null_one_form, char_projected_rhs_dt_v_minus)
    t5_ = np.einsum('d,cb,a,cd->ab', incoming_null_vector, projection_Ab,
                    outgoing_null_one_form, char_projected_rhs_dt_v_minus)
    t6_ = np.einsum('cd,ab,cd->ab', projection_AB, projection_ab,
                    char_projected_rhs_dt_v_minus)

    common_term = np.sqrt(0.5) * char_speeds[3] * (
        constraint_char_zero_minus - mu * constraint_char_zero_plus)
    t7_ = np.einsum('a,b,c,c->ab', outgoing_null_one_form,
                    outgoing_null_one_form, incoming_null_vector, common_term)
    t8_ = np.einsum('ab,c,c->ab', projection_ab, outgoing_null_vector,
                    common_term)
    t9_ = np.einsum('cb,a,c->ab', projection_Ab, outgoing_null_one_form,
                    common_term)
    t10_ = np.einsum('ca,b,c->ab', projection_Ab, outgoing_null_one_form,
                     common_term)
    return (0.5 * (2.0 * t1_ - t2_ - t3_ - t4_ - t5_ + t6_) +
            (t7_ + t8_ - t9_ - t10_))


def add_physical_dof_terms_to_dt_v_minus(
    gamma2, unit_interface_normal_one_form, unit_interface_normal_vector,
    spacetime_unit_normal_vector, projection_ab, projection_Ab, projection_AB,
    inverse_spatial_metric, extrinsic_curvature, spacetime_metric,
    inverse_spacetime_metric, three_index_constraint,
    char_projected_rhs_dt_v_minus, phi, d_phi, d_pi, char_speeds):
    mu_phys = 0
    adjust_phys_using_c4 = True
    gamma2_in_phys = True
    # calculate weyl propagating modes
    #       cov deriv of Kij
    #       calculate ricci3
    #       adjust_phys_using_c4
    #       calculate projection operators
    spatial_christoffel_1st_kind = ch.christoffel_first_kind(phi[:, 1:, 1:])
    spatial_christoffel_second_kind = np.einsum('ij,jkl->ikl',
                                                inverse_spatial_metric,
                                                spatial_christoffel_1st_kind)
    cov_d_Kij = gh.covariant_deriv_extrinsic_curvture(
        extrinsic_curvature, spacetime_unit_normal_vector,
        spatial_christoffel_second_kind, inverse_spacetime_metric, phi, d_pi,
        d_phi)
    ricci3 = gh.gh_spatial_ricci_tensor(phi, d_phi, inverse_spatial_metric)
    if adjust_phys_using_c4:
        ricci3 = ricci3 + 0.25 * (
            np.einsum('kl,iklj->ij', inverse_spatial_metric, d_phi[:, :, 1:,
                                                                   1:]) -
            np.einsum('kl,kilj->ij', inverse_spatial_metric, d_phi[:, :, 1:,
                                                                   1:]) +
            np.einsum('kl,jkli->ij', inverse_spatial_metric, d_phi[:, :, 1:,
                                                                   1:]) - np.
            einsum('kl,kjli->ij', inverse_spatial_metric, d_phi[:, :, 1:, 1:]))
        ricci3 = ricci3 + 0.5 * (
            np.einsum('k,a,ikja->ij', unit_interface_normal_vector,
                      spacetime_unit_normal_vector, d_phi[:, :, 1:, :]) -
            np.einsum('k,a,kija->ij', unit_interface_normal_vector,
                      spacetime_unit_normal_vector, d_phi[:, :, 1:, :]) +
            np.einsum('k,a,jkia->ij', unit_interface_normal_vector,
                      spacetime_unit_normal_vector, d_phi[:, :, 1:, :]) -
            np.einsum('k,a,kjia->ij', unit_interface_normal_vector,
                      spacetime_unit_normal_vector, d_phi[:, :, 1:, :]))
    spatial_proj_IJ = proj.transverse_projection_operator(
        inverse_spatial_metric, unit_interface_normal_vector)
    spatial_proj_ij = proj.transverse_projection_operator(
        spacetime_metric[1:, 1:], unit_interface_normal_one_form)
    spatial_proj_Ij =\
        proj.transverse_projection_operator_mixed_from_spatial_input(
            unit_interface_normal_vector, unit_interface_normal_one_form)
    weyl_prop_plus = wp.weyl_propagating_mode_plus(
        ricci3, extrinsic_curvature, inverse_spatial_metric, cov_d_Kij,
        unit_interface_normal_vector, spatial_proj_IJ, spatial_proj_ij,
        spatial_proj_Ij)
    weyl_prop_minus = wp.weyl_propagating_mode_minus(
        ricci3, extrinsic_curvature, inverse_spatial_metric, cov_d_Kij,
        unit_interface_normal_vector, spatial_proj_IJ, spatial_proj_ij,
        spatial_proj_Ij)
    # calculate U3+ U3-
    U3_plus = 2 * np.einsum('ia,jb,ij->ab', projection_Ab[1:, :],
                            projection_Ab[1:, :], weyl_prop_plus)
    U3_minus = 2 * np.einsum('ia,jb,ij->ab', projection_Ab[1:, :],
                             projection_Ab[1:, :], weyl_prop_minus)
    # calculate corrections
    tmp_ = char_speeds[3] * (U3_minus - mu_phys * U3_plus)
    if gamma2_in_phys:
        tmp_ = tmp_ - char_speeds[3] * gamma2 * np.einsum(
            'i,iab->ab', unit_interface_normal_vector, three_index_constraint)

    t1_ = np.einsum('ac,bd,ab->cd', projection_Ab, projection_Ab,
                    char_projected_rhs_dt_v_minus + tmp_)
    t2_ = -0.5 * np.einsum('ab,cd,ab->cd', projection_AB, projection_ab,
                           char_projected_rhs_dt_v_minus + tmp_)
    return t1_ + t2_


def constraint_preserving_bjorhus_corrections_dt_v_minus(
    gamma2, inertial_coords, incoming_null_one_form, outgoing_null_one_form,
    incoming_null_vector, outgoing_null_vector, projection_ab, projection_Ab,
    projection_AB, char_projected_rhs_dt_v_psi, char_projected_rhs_dt_v_minus,
    constraint_char_zero_plus, constraint_char_zero_minus, char_speeds):
    return add_constraint_dependent_terms_to_dt_v_minus(
        incoming_null_one_form, outgoing_null_one_form, incoming_null_vector,
        outgoing_null_vector, projection_ab, projection_Ab, projection_AB,
        constraint_char_zero_plus, constraint_char_zero_minus,
        char_projected_rhs_dt_v_minus,
        char_speeds) + add_gauge_sommerfeld_terms_to_dt_v_minus(
            gamma2, inertial_coords, incoming_null_one_form,
            outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
            projection_Ab,
            char_projected_rhs_dt_v_psi) - char_projected_rhs_dt_v_minus


def constraint_preserving_physical_bjorhus_corrections_dt_v_minus(
    gamma2, inertial_coords, unit_interface_normal_one_form,
    unit_interface_normal_vector, spacetime_unit_normal_vector,
    incoming_null_one_form, outgoing_null_one_form, incoming_null_vector,
    outgoing_null_vector, projection_ab, projection_Ab, projection_AB,
    inverse_spatial_metric, extrinsic_curvature, spacetime_metric,
    inverse_spacetime_metric, three_index_constraint,
    char_projected_rhs_dt_v_psi, char_projected_rhs_dt_v_minus,
    constraint_char_zero_plus, constraint_char_zero_minus, phi, d_phi, d_pi,
    char_speeds):
    return add_constraint_dependent_terms_to_dt_v_minus(
        incoming_null_one_form, outgoing_null_one_form, incoming_null_vector,
        outgoing_null_vector, projection_ab, projection_Ab, projection_AB,
        constraint_char_zero_plus, constraint_char_zero_minus,
        char_projected_rhs_dt_v_minus,
        char_speeds) + add_physical_dof_terms_to_dt_v_minus(
            gamma2, unit_interface_normal_one_form,
            unit_interface_normal_vector, spacetime_unit_normal_vector,
            projection_ab, projection_Ab, projection_AB,
            inverse_spatial_metric, extrinsic_curvature, spacetime_metric,
            inverse_spacetime_metric, three_index_constraint,
            char_projected_rhs_dt_v_minus, phi, d_phi, d_pi,
            char_speeds) + add_gauge_sommerfeld_terms_to_dt_v_minus(
                gamma2, inertial_coords, incoming_null_one_form,
                outgoing_null_one_form, incoming_null_vector,
                outgoing_null_vector, projection_Ab,
                char_projected_rhs_dt_v_psi) - char_projected_rhs_dt_v_minus
