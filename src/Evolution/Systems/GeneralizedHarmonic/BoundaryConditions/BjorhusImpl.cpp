// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/CovariantDerivOfExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace GeneralizedHarmonic::BoundaryConditions::Bjorhus {
template <size_t VolumeDim, typename DataType>
void constraint_preserving_bjorhus_corrections_dt_v_psi(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_psi,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const std::array<DataType, 4>& char_speeds) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*bc_dt_v_psi)) !=
               get_size(get<0>(unit_interface_normal_vector)))) {
    *bc_dt_v_psi = tnsr::aa<DataType, VolumeDim, Frame::Inertial>{
        get_size(get<0>(unit_interface_normal_vector))};
  }
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_v_psi->get(a, b) = char_speeds[0] *
                               unit_interface_normal_vector.get(0) *
                               three_index_constraint.get(0, a, b);
      for (size_t i = 1; i < VolumeDim; ++i) {
        bc_dt_v_psi->get(a, b) += char_speeds[0] *
                                  unit_interface_normal_vector.get(i) *
                                  three_index_constraint.get(i, a, b);
      }
    }
  }
}

template <size_t VolumeDim, typename DataType>
void constraint_preserving_bjorhus_corrections_dt_v_zero(
    const gsl::not_null<tnsr::iaa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_zero,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        four_index_constraint,
    const std::array<DataType, 4>& char_speeds) noexcept {
  if (UNLIKELY(get_size(get<0, 0, 0>(*bc_dt_v_zero)) !=
               get_size(get<0>(unit_interface_normal_vector)))) {
    *bc_dt_v_zero = tnsr::iaa<DataType, VolumeDim, Frame::Inertial>{
        get_size(get<0>(unit_interface_normal_vector))};
  }
  std::fill(bc_dt_v_zero->begin(), bc_dt_v_zero->end(), 0.);

  if (LIKELY(VolumeDim == 3)) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = a; b <= VolumeDim; ++b) {
        // Lets say this term is T2_{iab} := - n_l \beta^l n^j C_{jiab}.
        // But we store D_{iab} = LeviCivita^{ijk} dphi_{jkab},
        // and C_{ijab} = LeviCivita^{kij} D_{kab}
        // where D is `four_index_constraint`.
        // therefore, T2_{iab} =  char_speed<VZero> n^j C_{jiab}
        // (since char_speed<VZero> = - n_l \beta^l), and therefore:
        // T2_{iab} = char_speed<VZero> n^j LeviCivita^{ikj} D_{kab}.
        // Let LeviCivitaIterator be indexed by
        // it[0] <--> i,
        // it[1] <--> j,
        // it[2] <--> k, then
        // T2_{it[0], ab} += char_speed<VZero> n^it[2] it.sign() D_{it[1], ab};
        for (LeviCivitaIterator<VolumeDim> it; it; ++it) {
          bc_dt_v_zero->get(it[0], a, b) +=
              it.sign() * char_speeds[1] *
              unit_interface_normal_vector.get(it[2]) *
              four_index_constraint.get(it[1], a, b);
        }
      }
    }
  } else if (LIKELY(VolumeDim == 2)) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = a; b <= VolumeDim; ++b) {
        // Lets say this term is T2_{kab} := - n_l \beta^l n^j C_{jkab}.
        // In 2+1 spacetime, we store the four index constraint to
        // be D_{1ab} = C_{12ab}, C_{2ab} = C_{21ab}. Therefore,
        // T_{kab} = -n_l \beta^l (n^1 C_{1kab} + n^2 C_{2kab}), i.e.
        // T_{1ab} = -n_l \beta^l n^2 D_{2ab}, T_{2ab} = -n_l \beta^l n^1
        // D_{1ab}.
        bc_dt_v_zero->get(0, a, b) +=
            char_speeds[1] * (unit_interface_normal_vector.get(1) *
                              four_index_constraint.get(1, a, b));
        bc_dt_v_zero->get(1, a, b) +=
            char_speeds[1] * (unit_interface_normal_vector.get(0) *
                              four_index_constraint.get(0, a, b));
      }
    }
  }
}

namespace detail {
template <size_t VolumeDim, typename DataType>
void add_gauge_sommerfeld_terms_to_dt_v_minus(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const Scalar<DataType>& gamma2,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>& inertial_coords,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& incoming_null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& outgoing_null_one_form,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::Ab<DataType, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi) noexcept {
  // gauge_bc_coeff below is hard-coded here to its default value in SpEC
  constexpr double gauge_bc_coeff = 1.;

  DataType inertial_radius_or_scalar_factor(get_size(get<0>(inertial_coords)),
                                            0.);
  for (size_t i = 0; i < VolumeDim; ++i) {
    inertial_radius_or_scalar_factor += square(inertial_coords.get(i));
  }
  inertial_radius_or_scalar_factor =
      get(gamma2) - (gauge_bc_coeff / sqrt(inertial_radius_or_scalar_factor));

  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t c = 0; c <= VolumeDim; ++c) {
        for (size_t d = 0; d <= VolumeDim; ++d) {
          bc_dt_v_minus->get(a, b) +=
              (incoming_null_one_form.get(a) * projection_Ab.get(c, b) *
                   outgoing_null_vector.get(d) +
               incoming_null_one_form.get(b) * projection_Ab.get(c, a) *
                   outgoing_null_vector.get(d) -
               (incoming_null_one_form.get(a) * outgoing_null_one_form.get(b) *
                    incoming_null_vector.get(c) * outgoing_null_vector.get(d) +
                incoming_null_one_form.get(b) * outgoing_null_one_form.get(a) *
                    incoming_null_vector.get(c) * outgoing_null_vector.get(d) +
                incoming_null_one_form.get(a) * incoming_null_one_form.get(b) *
                    outgoing_null_vector.get(c) *
                    outgoing_null_vector.get(d))) *
              inertial_radius_or_scalar_factor *
              char_projected_rhs_dt_v_psi.get(c, d);
        }
      }
    }
  }
}

template <size_t VolumeDim, typename DataType>
void add_constraint_dependent_terms_to_dt_v_minus(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& outgoing_null_one_form,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataType, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_minus,
    const std::array<DataType, 4>& char_speeds) noexcept {
  constexpr double mu = 0.;  // hard-coded value from SpEC Bbh input file Mu = 0
  const double one_by_sqrt_2 = 1. / sqrt(2.);

  // Add corrections c.f. Eq (64) of gr-qc/0512093
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t c = 0; c <= VolumeDim; ++c) {
        for (size_t d = 0; d <= VolumeDim; ++d) {
          bc_dt_v_minus->get(a, b) +=
              0.5 *
              (2. * incoming_null_vector.get(c) * incoming_null_vector.get(d) *
                   outgoing_null_one_form.get(a) *
                   outgoing_null_one_form.get(b) -
               incoming_null_vector.get(c) * projection_Ab.get(d, a) *
                   outgoing_null_one_form.get(b) -
               incoming_null_vector.get(c) * projection_Ab.get(d, b) *
                   outgoing_null_one_form.get(a) -
               incoming_null_vector.get(d) * projection_Ab.get(c, a) *
                   outgoing_null_one_form.get(b) -
               incoming_null_vector.get(d) * projection_Ab.get(c, b) *
                   outgoing_null_one_form.get(a) +
               projection_AB.get(c, d) * projection_ab.get(a, b)) *
              char_projected_rhs_dt_v_minus.get(c, d);
        }
      }
      if constexpr (mu == 0.) {
        for (size_t c = 0; c <= VolumeDim; ++c) {
          bc_dt_v_minus->get(a, b) +=
              one_by_sqrt_2 * char_speeds[3] *
              constraint_char_zero_minus.get(c) *
              (outgoing_null_one_form.get(a) * outgoing_null_one_form.get(b) *
                   incoming_null_vector.get(c) +
               projection_ab.get(a, b) * outgoing_null_vector.get(c) -
               projection_Ab.get(c, b) * outgoing_null_one_form.get(a) -
               projection_Ab.get(c, a) * outgoing_null_one_form.get(b));
        }
      } else {
        for (size_t c = 0; c <= VolumeDim; ++c) {
          bc_dt_v_minus->get(a, b) +=
              one_by_sqrt_2 * char_speeds[3] *
              (constraint_char_zero_minus.get(c) -
               mu * constraint_char_zero_plus.get(c)) *
              (outgoing_null_one_form.get(a) * outgoing_null_one_form.get(b) *
                   incoming_null_vector.get(c) +
               projection_ab.get(a, b) * outgoing_null_vector.get(c) -
               projection_Ab.get(c, b) * outgoing_null_one_form.get(a) -
               projection_Ab.get(c, a) * outgoing_null_one_form.get(b));
        }
      }
    }
  }
}

template <size_t VolumeDim, typename DataType>
void add_physical_terms_to_dt_v_minus(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const Scalar<DataType>& gamma2,
    const tnsr::i<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_one_form,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataType, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::II<DataType, VolumeDim, Frame::Inertial>&
        inverse_spatial_metric,
    const tnsr::ii<DataType, VolumeDim, Frame::Inertial>& extrinsic_curvature,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& spacetime_metric,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>&
        inverse_spacetime_metric,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_minus,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame::Inertial>& d_phi,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>& d_pi,
    const std::array<DataType, 4>& char_speeds) noexcept {
  // hard-coded value from SpEC Bbh input file Mu = MuPhys = 0
  constexpr double mu_phys = 0.;
  constexpr bool adjust_phys_using_c4 = true;
  constexpr bool gamma2_in_phys = true;

  // In what follows, we follow Kidder, Scheel & Teukolsky (2001)
  // https://arxiv.org/pdf/gr-qc/0105031.pdf.
  TempBuffer<tmpl::list<::Tags::Tempaa<0, VolumeDim, Frame::Inertial, DataType>,
                        ::Tags::Tempaa<1, VolumeDim, Frame::Inertial, DataType>,
                        ::Tags::TempScalar<0, DataType>>>
      u3_buffer(get_size(get<0>(unit_interface_normal_vector)), 0.);
  auto& U3p =
      get<::Tags::Tempaa<0, VolumeDim, Frame::Inertial, DataType>>(u3_buffer);
  auto& U3m =
      get<::Tags::Tempaa<1, VolumeDim, Frame::Inertial, DataType>>(u3_buffer);

  {
    TempBuffer<
        tmpl::list<::Tags::Tempijj<0, VolumeDim, Frame::Inertial, DataType>,
                   // cov deriv of Kij
                   ::Tags::Tempijj<1, VolumeDim, Frame::Inertial, DataType>,
                   // spatial Ricci
                   ::Tags::Tempii<0, VolumeDim, Frame::Inertial, DataType>,
                   // spatial projection operators P_ij, P^ij, and P^i_j
                   ::Tags::TempII<0, VolumeDim, Frame::Inertial, DataType>,
                   ::Tags::Tempii<1, VolumeDim, Frame::Inertial, DataType>,
                   ::Tags::TempIj<0, VolumeDim, Frame::Inertial, DataType>,
                   // weyl propagating modes
                   ::Tags::Tempii<2, VolumeDim, Frame::Inertial, DataType>,
                   ::Tags::Tempii<3, VolumeDim, Frame::Inertial, DataType>,
                   ::Tags::Tempii<4, VolumeDim, Frame::Inertial, DataType>>>
        local_buffer(get_size(get<0>(unit_interface_normal_vector)), 0.);

    auto& spatial_phi =
        get<::Tags::Tempijj<0, VolumeDim, Frame::Inertial, DataType>>(
            local_buffer);
    auto& cov_deriv_ex_curv =
        get<::Tags::Tempijj<1, VolumeDim, Frame::Inertial, DataType>>(
            local_buffer);
    auto& ricci_3 =
        get<::Tags::Tempii<0, VolumeDim, Frame::Inertial, DataType>>(
            local_buffer);
    auto& spatial_projection_IJ =
        get<::Tags::TempII<0, VolumeDim, Frame::Inertial, DataType>>(
            local_buffer);
    auto& spatial_projection_ij =
        get<::Tags::Tempii<1, VolumeDim, Frame::Inertial, DataType>>(
            local_buffer);
    auto& spatial_projection_Ij =
        get<::Tags::TempIj<0, VolumeDim, Frame::Inertial, DataType>>(
            local_buffer);
    auto weyl_prop_minus =
        get<::Tags::Tempii<2, VolumeDim, Frame::Inertial, DataType>>(
            local_buffer);
    auto& spatial_metric =
        get<::Tags::Tempii<3, VolumeDim, Frame::Inertial, DataType>>(
            local_buffer);

    // D_(k,i,j) = (1/2) \partial_k g_(ij) and its derivative
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = i; j < VolumeDim; ++j) {
        for (size_t k = 0; k < VolumeDim; ++k) {
          spatial_phi.get(k, i, j) = phi.get(k, i + 1, j + 1);
        }
      }
    }

    // Compute covariant deriv of extrinsic curvature
    GeneralizedHarmonic::covariant_deriv_of_extrinsic_curvature(
        make_not_null(&cov_deriv_ex_curv), extrinsic_curvature,
        spacetime_unit_normal_vector,
        raise_or_lower_first_index(gr::christoffel_first_kind(spatial_phi),
                                   inverse_spatial_metric),
        inverse_spacetime_metric, phi, d_pi, d_phi);

    // Compute spatial Ricci tensor
    GeneralizedHarmonic::spatial_ricci_tensor(make_not_null(&ricci_3), phi,
                                              d_phi, inverse_spatial_metric);

    if (adjust_phys_using_c4) {
      // This adds 4-index constraint terms to 3Ricci so as to cancel
      // out normal derivatives from the final expression for U8.
      // It is much easier to add them here than to recalculate U8
      // from scratch.

      // Add some 4-index constraint terms to 3Ricci.
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            for (size_t l = 0; l < VolumeDim; ++l) {
              ricci_3.get(i, j) += 0.25 * inverse_spatial_metric.get(k, l) *
                                   (d_phi.get(i, k, 1 + l, 1 + j) -
                                    d_phi.get(k, i, 1 + l, 1 + j) +
                                    d_phi.get(j, k, 1 + l, 1 + i) -
                                    d_phi.get(k, j, 1 + l, 1 + i));
            }
          }
        }
      }

      // Add more 4-index constraint terms to 3Ricci
      // These compensate for some of the cov_deriv_ex_curv terms.
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t a = 0; a <= VolumeDim; ++a) {
            for (size_t k = 0; k < VolumeDim; ++k) {
              ricci_3.get(i, j) +=
                  0.5 * unit_interface_normal_vector.get(k) *
                  spacetime_unit_normal_vector.get(a) *
                  (d_phi.get(i, k, j + 1, a) - d_phi.get(k, i, j + 1, a) +
                   d_phi.get(j, k, i + 1, a) - d_phi.get(k, j, i + 1, a));
            }
          }
        }
      }
    }

    // Make spatial projection operators
    for (size_t j = 0; j < VolumeDim; ++j) {
      for (size_t k = j; k < VolumeDim; ++k) {
        spatial_metric.get(j, k) = spacetime_metric.get(1 + j, 1 + k);
      }
    }
    gr::transverse_projection_operator(make_not_null(&spatial_projection_IJ),
                                       inverse_spatial_metric,
                                       unit_interface_normal_vector);
    gr::transverse_projection_operator(make_not_null(&spatial_projection_ij),
                                       spatial_metric,
                                       unit_interface_normal_one_form);
    gr::transverse_projection_operator(make_not_null(&spatial_projection_Ij),
                                       unit_interface_normal_vector,
                                       unit_interface_normal_one_form);

    // Weyl propagating mode
    gr::weyl_propagating(make_not_null(&weyl_prop_minus), ricci_3,
                         extrinsic_curvature, inverse_spatial_metric,
                         cov_deriv_ex_curv, unit_interface_normal_vector,
                         spatial_projection_IJ, spatial_projection_ij,
                         spatial_projection_Ij, -1);

    if constexpr (mu_phys == 0.) {
      // No need to compute U3p or weyl_prop_plus in this case
      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          for (size_t i = 0; i < VolumeDim; ++i) {
            for (size_t j = 0; j < VolumeDim; ++j) {
              U3m.get(a, b) += 2. * projection_Ab.get(i + 1, a) *
                               projection_Ab.get(j + 1, b) *
                               weyl_prop_minus.get(i, j);
            }
          }
        }
      }
    } else {
      auto& weyl_prop_plus =
          get<::Tags::Tempii<3, VolumeDim, Frame::Inertial, DataType>>(
              local_buffer);
      gr::weyl_propagating(make_not_null(&weyl_prop_plus), ricci_3,
                           extrinsic_curvature, inverse_spatial_metric,
                           cov_deriv_ex_curv, unit_interface_normal_vector,
                           spatial_projection_IJ, spatial_projection_ij,
                           spatial_projection_Ij, 1);

      for (size_t a = 0; a <= VolumeDim; ++a) {
        for (size_t b = a; b <= VolumeDim; ++b) {
          for (size_t i = 0; i < VolumeDim; ++i) {
            for (size_t j = 0; j < VolumeDim; ++j) {
              U3p.get(a, b) += 2. * projection_Ab.get(i + 1, a) *
                               projection_Ab.get(j + 1, b) *
                               weyl_prop_plus.get(i, j);
              U3m.get(a, b) += 2. * projection_Ab.get(i + 1, a) *
                               projection_Ab.get(j + 1, b) *
                               weyl_prop_minus.get(i, j);
            }
          }
        }
      }
    }
  }

  // Add physical boundary corrections
  if (gamma2_in_phys) {
    auto& normal_dot_three_index_constraint_gamma2 =
        get(get<::Tags::TempScalar<0, DataType>>(u3_buffer));

    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = a; b <= VolumeDim; ++b) {
        for (size_t c = 0; c <= VolumeDim; ++c) {
          for (size_t d = 0; d <= VolumeDim; ++d) {
            normal_dot_three_index_constraint_gamma2 =
                get<0>(unit_interface_normal_vector) *
                three_index_constraint.get(0, c, d);
            for (size_t i = 1; i < VolumeDim; ++i) {
              normal_dot_three_index_constraint_gamma2 +=
                  unit_interface_normal_vector.get(i) *
                  three_index_constraint.get(i, c, d);
            }
            normal_dot_three_index_constraint_gamma2 *= get(gamma2);

            if constexpr (mu_phys == 0.) {
              bc_dt_v_minus->get(a, b) +=
                  (projection_Ab.get(c, a) * projection_Ab.get(d, b) -
                   0.5 * projection_ab.get(a, b) * projection_AB.get(c, d)) *
                  (char_projected_rhs_dt_v_minus.get(c, d) +
                   char_speeds[3] * (U3m.get(c, d) -
                                     normal_dot_three_index_constraint_gamma2));
            } else {
              bc_dt_v_minus->get(a, b) +=
                  (projection_Ab.get(c, a) * projection_Ab.get(d, b) -
                   0.5 * projection_ab.get(a, b) * projection_AB.get(c, d)) *
                  (char_projected_rhs_dt_v_minus.get(c, d) +
                   char_speeds[3] * (U3m.get(c, d) -
                                     normal_dot_three_index_constraint_gamma2 -
                                     mu_phys * U3p.get(c, d)));
            }
          }
        }
      }
    }
  } else {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = a; b <= VolumeDim; ++b) {
        for (size_t c = 0; c <= VolumeDim; ++c) {
          for (size_t d = 0; d <= VolumeDim; ++d) {
            if constexpr (mu_phys == 0.) {
              (projection_Ab.get(c, a) * projection_Ab.get(d, b) -
               0.5 * projection_ab.get(a, b) * projection_AB.get(c, d)) *
                  (char_projected_rhs_dt_v_minus.get(c, d) +
                   char_speeds[3] * (U3m.get(c, d)));
            } else {
              bc_dt_v_minus->get(a, b) +=
                  (projection_Ab.get(c, a) * projection_Ab.get(d, b) -
                   0.5 * projection_ab.get(a, b) * projection_AB.get(c, d)) *
                  (char_projected_rhs_dt_v_minus.get(c, d) +
                   char_speeds[3] * (U3m.get(c, d) - mu_phys * U3p.get(c, d)));
            }
          }
        }
      }
    }
  }
}
}  // namespace detail

template <size_t VolumeDim, typename DataType>
void constraint_preserving_bjorhus_corrections_dt_v_minus(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const Scalar<DataType>& gamma2,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>& inertial_coords,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& incoming_null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& outgoing_null_one_form,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataType, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_minus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const std::array<DataType, 4>& char_speeds) noexcept {
  destructive_resize_components(bc_dt_v_minus, get_size(get(gamma2)));
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_v_minus->get(a, b) = -char_projected_rhs_dt_v_minus.get(a, b);
    }
  }
  detail::add_constraint_dependent_terms_to_dt_v_minus(
      bc_dt_v_minus, outgoing_null_one_form, incoming_null_vector,
      outgoing_null_vector, projection_ab, projection_Ab, projection_AB,
      constraint_char_zero_plus, constraint_char_zero_minus,
      char_projected_rhs_dt_v_minus, char_speeds);
  detail::add_gauge_sommerfeld_terms_to_dt_v_minus(
      bc_dt_v_minus, gamma2, inertial_coords, incoming_null_one_form,
      outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
      projection_Ab, char_projected_rhs_dt_v_psi);
}

template <size_t VolumeDim, typename DataType>
void constraint_preserving_physical_bjorhus_corrections_dt_v_minus(
    const gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame::Inertial>*>
        bc_dt_v_minus,
    const Scalar<DataType>& gamma2,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>& inertial_coords,
    const tnsr::i<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_one_form,
    const tnsr::I<DataType, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& incoming_null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>& outgoing_null_one_form,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataType, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataType, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::II<DataType, VolumeDim, Frame::Inertial>&
        inverse_spatial_metric,
    const tnsr::ii<DataType, VolumeDim, Frame::Inertial>& extrinsic_curvature,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>& spacetime_metric,
    const tnsr::AA<DataType, VolumeDim, Frame::Inertial>&
        inverse_spacetime_metric,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_psi,
    const tnsr::aa<DataType, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_v_minus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataType, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame::Inertial>& d_phi,
    const tnsr::iaa<DataType, VolumeDim, Frame::Inertial>& d_pi,
    const std::array<DataType, 4>& char_speeds) noexcept {
  destructive_resize_components(bc_dt_v_minus, get_size(get(gamma2)));
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_v_minus->get(a, b) = -char_projected_rhs_dt_v_minus.get(a, b);
    }
  }
  detail::add_constraint_dependent_terms_to_dt_v_minus(
      bc_dt_v_minus, outgoing_null_one_form, incoming_null_vector,
      outgoing_null_vector, projection_ab, projection_Ab, projection_AB,
      constraint_char_zero_plus, constraint_char_zero_minus,
      char_projected_rhs_dt_v_minus, char_speeds);
  detail::add_physical_terms_to_dt_v_minus(
      bc_dt_v_minus, gamma2, unit_interface_normal_one_form,
      unit_interface_normal_vector, spacetime_unit_normal_vector, projection_ab,
      projection_Ab, projection_AB, inverse_spatial_metric, extrinsic_curvature,
      spacetime_metric, inverse_spacetime_metric, three_index_constraint,
      char_projected_rhs_dt_v_minus, phi, d_phi, d_pi, char_speeds);
  detail::add_gauge_sommerfeld_terms_to_dt_v_minus(
      bc_dt_v_minus, gamma2, inertial_coords, incoming_null_one_form,
      outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
      projection_Ab, char_projected_rhs_dt_v_psi);
}
}  // namespace GeneralizedHarmonic::BoundaryConditions::Bjorhus

// Explicit Instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void GeneralizedHarmonic::BoundaryConditions::Bjorhus::          \
      constraint_preserving_bjorhus_corrections_dt_v_psi(                   \
          const gsl::not_null<                                              \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>           \
              bc_dt_v_psi,                                                  \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_vector,                                 \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>&         \
              three_index_constraint,                                       \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;          \
  template void GeneralizedHarmonic::BoundaryConditions::Bjorhus::          \
      constraint_preserving_bjorhus_corrections_dt_v_zero(                  \
          const gsl::not_null<                                              \
              tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>*>          \
              bc_dt_v_zero,                                                 \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_vector,                                 \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>&         \
              four_index_constraint,                                        \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;          \
  template void GeneralizedHarmonic::BoundaryConditions::Bjorhus::detail::  \
      add_gauge_sommerfeld_terms_to_dt_v_minus(                             \
          const gsl::not_null<                                              \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>           \
              bc_dt_v_minus,                                                \
          const Scalar<DTYPE(data)>& gamma2,                                \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              inertial_coords,                                              \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_one_form,                                       \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_one_form,                                       \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_vector,                                         \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_vector,                                         \
          const tnsr::Ab<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_Ab,                                                \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_psi) noexcept;                        \
  template void GeneralizedHarmonic::BoundaryConditions::Bjorhus::detail::  \
      add_constraint_dependent_terms_to_dt_v_minus(                         \
          const gsl::not_null<                                              \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>           \
              bc_dt_v_minus,                                                \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_one_form,                                       \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_vector,                                         \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_vector,                                         \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_ab,                                                \
          const tnsr::Ab<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_Ab,                                                \
          const tnsr::AA<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_AB,                                                \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_plus,                                    \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_minus,                                   \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_minus,                                \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;          \
  template void GeneralizedHarmonic::BoundaryConditions::Bjorhus::detail::  \
      add_physical_terms_to_dt_v_minus(                                     \
          const gsl::not_null<                                              \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>           \
              bc_dt_v_minus,                                                \
          const Scalar<DTYPE(data)>& gamma2,                                \
          const tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_one_form,                               \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_vector,                                 \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              spacetime_unit_normal_vector,                                 \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_ab,                                                \
          const tnsr::Ab<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_Ab,                                                \
          const tnsr::AA<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_AB,                                                \
          const tnsr::II<DTYPE(data), DIM(data), Frame::Inertial>&          \
              inverse_spatial_metric,                                       \
          const tnsr::ii<DTYPE(data), DIM(data), Frame::Inertial>&          \
              extrinsic_curvature,                                          \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              spacetime_metric,                                             \
          const tnsr::AA<DTYPE(data), DIM(data), Frame::Inertial>&          \
              inverse_spacetime_metric,                                     \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>&         \
              three_index_constraint,                                       \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_minus,                                \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>& phi,    \
          const tnsr::ijaa<DTYPE(data), DIM(data), Frame::Inertial>& d_phi, \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>& d_pi,   \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;          \
  template void GeneralizedHarmonic::BoundaryConditions::Bjorhus::          \
      constraint_preserving_bjorhus_corrections_dt_v_minus(                 \
          const gsl::not_null<                                              \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>           \
              bc_dt_v_minus,                                                \
          const Scalar<DTYPE(data)>& gamma2,                                \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              inertial_coords,                                              \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_one_form,                                       \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_one_form,                                       \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_vector,                                         \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_vector,                                         \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_ab,                                                \
          const tnsr::Ab<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_Ab,                                                \
          const tnsr::AA<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_AB,                                                \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_psi,                                  \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_minus,                                \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_plus,                                    \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_minus,                                   \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;          \
  template void GeneralizedHarmonic::BoundaryConditions::Bjorhus::          \
      constraint_preserving_physical_bjorhus_corrections_dt_v_minus(        \
          const gsl::not_null<                                              \
              tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>*>           \
              bc_dt_v_minus,                                                \
          const Scalar<DTYPE(data)>& gamma2,                                \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              inertial_coords,                                              \
          const tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_one_form,                               \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>&           \
              unit_interface_normal_vector,                                 \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              spacetime_unit_normal_vector,                                 \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_one_form,                                       \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_one_form,                                       \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              incoming_null_vector,                                         \
          const tnsr::A<DTYPE(data), DIM(data), Frame::Inertial>&           \
              outgoing_null_vector,                                         \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_ab,                                                \
          const tnsr::Ab<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_Ab,                                                \
          const tnsr::AA<DTYPE(data), DIM(data), Frame::Inertial>&          \
              projection_AB,                                                \
          const tnsr::II<DTYPE(data), DIM(data), Frame::Inertial>&          \
              inverse_spatial_metric,                                       \
          const tnsr::ii<DTYPE(data), DIM(data), Frame::Inertial>&          \
              extrinsic_curvature,                                          \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              spacetime_metric,                                             \
          const tnsr::AA<DTYPE(data), DIM(data), Frame::Inertial>&          \
              inverse_spacetime_metric,                                     \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>&         \
              three_index_constraint,                                       \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_psi,                                  \
          const tnsr::aa<DTYPE(data), DIM(data), Frame::Inertial>&          \
              char_projected_rhs_dt_v_minus,                                \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_plus,                                    \
          const tnsr::a<DTYPE(data), DIM(data), Frame::Inertial>&           \
              constraint_char_zero_minus,                                   \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>& phi,    \
          const tnsr::ijaa<DTYPE(data), DIM(data), Frame::Inertial>& d_phi, \
          const tnsr::iaa<DTYPE(data), DIM(data), Frame::Inertial>& d_pi,   \
          const std::array<DTYPE(data), 4>& char_speeds) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef DIM
