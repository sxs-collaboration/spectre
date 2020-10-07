// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusHelpers.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusInternals.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryConditions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace GeneralizedHarmonic {
namespace Actions {

namespace BoundaryConditions_detail {}  // namespace BoundaryConditions_detail

namespace BoundaryConditions_detail {
using BoundaryConditions::Bjorhus::VMinusBcMethod;
using BoundaryConditions::Bjorhus::VPlusBcMethod;
using BoundaryConditions::Bjorhus::VSpacetimeMetricBcMethod;
using BoundaryConditions::Bjorhus::VZeroBcMethod;

// \brief This struct sets boundary condition on dt<VSpacetimeMetric>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_v_psi {
  template <typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(
      const VSpacetimeMetricBcMethod Method,
      const gsl::not_null<BjorhusIntermediatesComputer*> intermediates,
      const Variables<VarsTagsList>& /* vars */,
      const Variables<DtVarsTagsList>& /* dt_vars */,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
      /* unit_normal_one_form */) noexcept {
    const auto three_index_constraint = intermediates->get_var(
        Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>{});
    const auto unit_interface_normal_vector = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::interface_normal_vector<
            VolumeDim, DataVector>{});
    const auto char_projected_rhs_dt_u_psi = intermediates->get_var(
        Tags::VSpacetimeMetric<VolumeDim, Frame::Inertial>{});
    const std::array<DataVector, 4> char_speeds{
        {get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vpsi<
                 DataVector>{})),
         get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vzero<
                 DataVector>{})),
         get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vplus<
                 DataVector>{})),
         get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vminus<
                 DataVector>{}))}};

    // Memory allocated for return type
    ReturnType& bc_dt_u_psi = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::bc_dt_u_psi<VolumeDim,
                                                                 DataVector>{});
    std::fill(bc_dt_u_psi.begin(), bc_dt_u_psi.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VSpacetimeMetricBcMethod::Freezing:
        return bc_dt_u_psi;
      case VSpacetimeMetricBcMethod::ConstraintPreserving:
        return apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_psi), unit_interface_normal_vector,
            three_index_constraint, char_projected_rhs_dt_u_psi, char_speeds);
      case VSpacetimeMetricBcMethod::Unknown:
      default:
        ASSERT(false,
               "Requested BC method fo VSpacetimeMetric not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      gsl::not_null<ReturnType*> bc_dt_u_psi,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
          three_index_constraint,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_psi,
      const std::array<DataVector, 4>& char_speeds) noexcept;
};

template <typename ReturnType, size_t VolumeDim>
ReturnType
set_dt_v_psi<ReturnType, VolumeDim>::apply_bjorhus_constraint_preserving(
    const gsl::not_null<ReturnType*> bc_dt_u_psi,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
        three_index_constraint,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_u_psi,
    const std::array<DataVector, 4>& char_speeds) noexcept {
  ASSERT(get_size(get<0, 0>(*bc_dt_u_psi)) ==
             get_size(get<0>(unit_interface_normal_vector)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0>(*bc_dt_u_psi)) << ","
             << get_size(get<0>(unit_interface_normal_vector)));
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      bc_dt_u_psi->get(a, b) += char_projected_rhs_dt_u_psi.get(a, b);
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_u_psi->get(a, b) += char_speeds.at(0) *
                                  unit_interface_normal_vector.get(i + 1) *
                                  three_index_constraint.get(i, a, b);
      }
    }
  }
  return *bc_dt_u_psi;
}

// \brief This struct sets boundary condition on dt<VZero>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_v_zero {
  template <typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(
      const VZeroBcMethod Method,
      const gsl::not_null<BjorhusIntermediatesComputer*> intermediates,
      const Variables<VarsTagsList>& /* vars */,
      const Variables<DtVarsTagsList>& /* dt_vars */,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
      /* unit_normal_one_form */) noexcept {
    // Not using auto below to enforce a loose test on the quantity being
    // fetched from the buffer
    const auto unit_interface_normal_vector = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::interface_normal_vector<
            VolumeDim, DataVector>{});
    const auto four_index_constraint = intermediates->get_var(
        Tags::FourIndexConstraint<VolumeDim, Frame::Inertial>{});
    const auto char_projected_rhs_dt_u_zero =
        intermediates->get_var(Tags::VZero<VolumeDim, Frame::Inertial>{});
    const std::array<DataVector, 4> char_speeds{
        {get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vpsi<
                 DataVector>{})),
         get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vzero<
                 DataVector>{})),
         get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vplus<
                 DataVector>{})),
         get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vminus<
                 DataVector>{}))}};

    // Memory allocated for return type
    ReturnType& bc_dt_u_zero = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::bc_dt_u_zero<
            VolumeDim, DataVector>{});
    std::fill(bc_dt_u_zero.begin(), bc_dt_u_zero.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VZeroBcMethod::Freezing:
        return bc_dt_u_zero;
      case VZeroBcMethod::ConstraintPreserving:
        return apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_zero), unit_interface_normal_vector,
            four_index_constraint, char_projected_rhs_dt_u_zero, char_speeds);
      case VZeroBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo VZero not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      gsl::not_null<ReturnType*> bc_dt_u_zero,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
          four_index_constraint,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_zero,
      const std::array<DataVector, 4>& char_speeds) noexcept;
};

template <typename ReturnType, size_t VolumeDim>
ReturnType
set_dt_v_zero<ReturnType, VolumeDim>::apply_bjorhus_constraint_preserving(
    const gsl::not_null<ReturnType*> bc_dt_u_zero,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
        unit_interface_normal_vector,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
        four_index_constraint,
    const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_u_zero,
    const std::array<DataVector, 4>& char_speeds) noexcept {
  ASSERT(get_size(get<0, 0, 0>(*bc_dt_u_zero)) ==
             get_size(get<0>(unit_interface_normal_vector)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0, 0>(*bc_dt_u_zero)) << ","
             << get_size(get<0>(unit_interface_normal_vector)));

  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t i = 0; i < VolumeDim; ++i) {
        bc_dt_u_zero->get(i, a, b) += char_projected_rhs_dt_u_zero.get(i, a, b);
      }
      // Lets say this term is T2_{kab} := - n_l N^l n^j C_{jkab}.
      // But we store C4_{iab} = LeviCivita^{ijk} dphi_{jkab},
      // which means  C_{jkab} = LeviCivita^{ijk} C4_{iab}
      // where C4 is `four_index_constraint`.
      // therefore, T2_{iab} =  char_speed<VZero> n^j C_{jiab}
      // (since char_speed<VZero> = - n_l N^l), and therefore:
      // T2_{iab} = char_speed<VZero> n^k LeviCivita^{ijk} C4_{jab}.
      // Let LeviCivitaIterator be indexed by
      // it[0] <--> i,
      // it[1] <--> j,
      // it[2] <--> k, then
      // T2_{it[0], ab} += char_speed<VZero> n^it[2] it.sign() C4_{it[1], ab};
      for (LeviCivitaIterator<VolumeDim> it; it; ++it) {
        bc_dt_u_zero->get(it[0], a, b) +=
            it.sign() * char_speeds.at(1) *
            unit_interface_normal_vector.get(it[2] + 1) *
            four_index_constraint.get(it[1], a, b);
      }
    }
  }
  return *bc_dt_u_zero;
}

// \brief This struct sets boundary condition on dt<VPlus>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_v_plus {
  template <typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(
      const VPlusBcMethod Method,
      const gsl::not_null<BjorhusIntermediatesComputer*> intermediates,
      const Variables<VarsTagsList>& /* vars */,
      const Variables<DtVarsTagsList>& /* dt_vars */,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
      /* unit_normal_one_form */) noexcept {
    // Memory allocated for return type
    ReturnType& bc_dt_u_plus = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::bc_dt_u_plus<
            VolumeDim, DataVector>{});
    std::fill(bc_dt_u_plus.begin(), bc_dt_u_plus.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VPlusBcMethod::Freezing:
        return bc_dt_u_plus;
      case VPlusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo VPlus not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
};

// \brief This struct sets boundary condition on dt<VMinus>
template <typename ReturnType, size_t VolumeDim>
struct set_dt_v_minus {
  template <typename VarsTagsList, typename DtVarsTagsList>
  static ReturnType apply(
      const VMinusBcMethod Method,
      const gsl::not_null<BjorhusIntermediatesComputer*> intermediates,
      const Variables<VarsTagsList>& vars,
      const Variables<DtVarsTagsList>& /* dt_vars */,
      const typename domain::Tags::Coordinates<
          VolumeDim, Frame::Inertial>::type& inertial_coords,
      const tnsr::i<DataVector, VolumeDim, Frame::Inertial>&
      /* unit_normal_one_form */) noexcept {
    // Not using auto below to enforce a loose test on the quantity being
    // fetched from the buffer
    const auto constraint_gamma2 =
        intermediates->get_var(Tags::ConstraintGamma2{});
    const auto three_index_constraint = intermediates->get_var(
        Tags::ThreeIndexConstraint<VolumeDim, Frame::Inertial>{});
    const auto unit_interface_normal_one_form = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::interface_normal_one_form<
            VolumeDim, DataVector>{});
    const auto unit_interface_normal_vector = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::interface_normal_vector<
            VolumeDim, DataVector>{});
    const auto spacetime_unit_normal_vector = intermediates->get_var(
        gr::Tags::SpacetimeNormalVector<VolumeDim, Frame::Inertial,
                                        DataVector>{});

    const auto char_projected_rhs_dt_u_psi = intermediates->get_var(
        Tags::VSpacetimeMetric<VolumeDim, Frame::Inertial>{});
    const auto char_projected_rhs_dt_u_minus =
        intermediates->get_var(Tags::VMinus<VolumeDim, Frame::Inertial>{});
    const std::array<DataVector, 4> char_speeds{
        {get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vpsi<
                 DataVector>{})),
         get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vzero<
                 DataVector>{})),
         get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vplus<
                 DataVector>{})),
         get(intermediates->get_var(
             BjorhusIntermediatesComputer::internal_tags::char_speed_vminus<
                 DataVector>{}))}};

    // timelike and spacelike SPACETIME vectors, l^a and k^a
    const auto& outgoing_null_one_form = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::outgoing_null_one_form<
            VolumeDim, DataVector>{});
    const auto& incoming_null_one_form = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::incoming_null_one_form<
            VolumeDim, DataVector>{});
    // timelike and spacelike SPACETIME oneforms, l_a and k_a
    const auto& outgoing_null_vector = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::outgoing_null_vector<
            VolumeDim, DataVector>{});
    const auto& incoming_null_vector = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::incoming_null_vector<
            VolumeDim, DataVector>{});
    // spacetime projection operator P_ab, P^ab, and P^a_b
    const auto& projection_AB = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::projection_AB<
            VolumeDim, DataVector>{});
    const auto& projection_ab = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::projection_ab<
            VolumeDim, DataVector>{});
    const auto& projection_Ab = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::projection_Ab<
            VolumeDim, DataVector>{});
    // constraint characteristics
    const auto& constraint_char_zero_minus = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::constraint_char_zero_minus<
            VolumeDim, DataVector>{});
    const auto& constraint_char_zero_plus = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::constraint_char_zero_plus<
            VolumeDim, DataVector>{});

    const auto& inverse_spatial_metric = intermediates->get_var(
        gr::Tags::InverseSpatialMetric<VolumeDim, Frame::Inertial,
                                       DataVector>{});
    const auto& extrinsic_curvature = intermediates->get_var(
        gr::Tags::ExtrinsicCurvature<VolumeDim, Frame::Inertial, DataVector>{});
    const auto& inverse_spacetime_metric = intermediates->get_var(
        gr::Tags::InverseSpacetimeMetric<VolumeDim, Frame::Inertial,
                                         DataVector>{});

    const typename gr::Tags::SpacetimeMetric<
        VolumeDim, Frame::Inertial, DataVector>::type& spacetime_metric =
        get<gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial, DataVector>>(
            vars);
    const typename Tags::Phi<VolumeDim, Frame::Inertial>::type& phi =
        get<Tags::Phi<VolumeDim, Frame::Inertial>>(vars);
    const auto& d_pi = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::deriv_pi<VolumeDim,
                                                              DataVector>{});
    const auto& d_phi = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::deriv_phi<VolumeDim,
                                                               DataVector>{});
    // Memory allocated for return type
    ReturnType& bc_dt_u_minus = intermediates->get_var(
        BjorhusIntermediatesComputer::internal_tags::bc_dt_u_minus<
            VolumeDim, DataVector>{});
    std::fill(bc_dt_u_minus.begin(), bc_dt_u_minus.end(), 0.);
    // Switch on prescribed boundary condition method
    switch (Method) {
      case VMinusBcMethod::Freezing:
        return apply_gauge_sommerfeld(
            make_not_null(&bc_dt_u_minus), constraint_gamma2, inertial_coords,
            incoming_null_one_form, outgoing_null_one_form,
            incoming_null_vector, outgoing_null_vector, projection_Ab,
            char_projected_rhs_dt_u_psi);
      case VMinusBcMethod::ConstraintPreserving:
        apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_minus), incoming_null_one_form,
            outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
            projection_ab, projection_Ab, projection_AB,
            constraint_char_zero_plus, constraint_char_zero_minus,
            char_projected_rhs_dt_u_minus, char_speeds);
        return apply_gauge_sommerfeld(
            make_not_null(&bc_dt_u_minus), constraint_gamma2, inertial_coords,
            incoming_null_one_form, outgoing_null_one_form,
            incoming_null_vector, outgoing_null_vector, projection_Ab,
            char_projected_rhs_dt_u_psi);
      case VMinusBcMethod::ConstraintPreservingPhysical:
        apply_bjorhus_constraint_preserving(
            make_not_null(&bc_dt_u_minus), incoming_null_one_form,
            outgoing_null_one_form, incoming_null_vector, outgoing_null_vector,
            projection_ab, projection_Ab, projection_AB,
            constraint_char_zero_plus, constraint_char_zero_minus,
            char_projected_rhs_dt_u_minus, char_speeds);
        apply_bjorhus_constraint_preserving_physical(
            make_not_null(&bc_dt_u_minus), constraint_gamma2,
            unit_interface_normal_one_form, unit_interface_normal_vector,
            spacetime_unit_normal_vector, projection_ab, projection_Ab,
            projection_AB, inverse_spatial_metric, extrinsic_curvature,
            spacetime_metric, inverse_spacetime_metric, three_index_constraint,
            char_projected_rhs_dt_u_minus, phi, d_phi, d_pi, char_speeds);
        return apply_gauge_sommerfeld(
            make_not_null(&bc_dt_u_minus), constraint_gamma2, inertial_coords,
            incoming_null_one_form, outgoing_null_one_form,
            incoming_null_vector, outgoing_null_vector, projection_Ab,
            char_projected_rhs_dt_u_psi);
      case VMinusBcMethod::Unknown:
      default:
        ASSERT(false, "Requested BC method fo VMinus not implemented!");
    }
    // dummy return to suppress compiler warning
    return ReturnType{};
  }

 private:
  static ReturnType apply_bjorhus_constraint_preserving(
      gsl::not_null<ReturnType*> bc_dt_u_minus,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          incoming_null_one_form,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          outgoing_null_one_form,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          incoming_null_vector,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          outgoing_null_vector,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
      const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
      const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          constraint_char_zero_plus,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          constraint_char_zero_minus,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_minus,
      const std::array<DataVector, 4>& char_speeds) noexcept;
  static ReturnType apply_bjorhus_constraint_preserving_physical(
      gsl::not_null<ReturnType*> bc_dt_u_minus,
      const Scalar<DataVector>& gamma2,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_one_form,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          unit_interface_normal_vector,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          spacetime_unit_normal_vector,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
      const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
      const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
      const tnsr::II<DataVector, VolumeDim, Frame::Inertial>&
          inverse_spatial_metric,
      const tnsr::ii<DataVector, VolumeDim, Frame::Inertial>&
          extrinsic_curvature,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& spacetime_metric,
      const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>&
          inverse_spacetime_metric,
      const typename Tags::ThreeIndexConstraint<
          VolumeDim, Frame::Inertial>::type& three_index_constraint,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_minus,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi,
      const tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>& d_phi,
      const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_pi,
      const std::array<DataVector, 4>& char_speeds) noexcept;
  static ReturnType apply_gauge_sommerfeld(
      gsl::not_null<ReturnType*> bc_dt_u_minus,
      const Scalar<DataVector>& gamma2,
      const typename domain::Tags::Coordinates<
          VolumeDim, Frame::Inertial>::type& inertial_coords,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          incoming_null_one_form,
      const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
          outgoing_null_one_form,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          incoming_null_vector,
      const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
          outgoing_null_vector,
      const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
      const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
          char_projected_rhs_dt_u_psi) noexcept;
};

template <typename ReturnType, size_t VolumeDim>
ReturnType
set_dt_v_minus<ReturnType, VolumeDim>::apply_bjorhus_constraint_preserving(
    const gsl::not_null<ReturnType*> bc_dt_u_minus,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        incoming_null_one_form,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        outgoing_null_one_form,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
    const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        constraint_char_zero_plus,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        constraint_char_zero_minus,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_u_minus,
    const std::array<DataVector, 4>& char_speeds) noexcept {
  ASSERT(get_size(get<0, 0>(*bc_dt_u_minus)) ==
             get_size(get<0>(incoming_null_one_form)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0>(*bc_dt_u_minus)) << ","
             << get_size(get<0>(incoming_null_one_form)));
  const double mMu = 0.;  // hard-coded value from SpEC Bbh input file Mu = 0
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t c = 0; c <= VolumeDim; ++c) {
        for (size_t d = 0; d <= VolumeDim; ++d) {
          bc_dt_u_minus->get(a, b) +=
              0.25 *
              (incoming_null_vector.get(c) * incoming_null_vector.get(d) *
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
               2.0 * projection_AB.get(c, d) * projection_ab.get(a, b)) *
              char_projected_rhs_dt_u_minus.get(c, d);
        }
      }
      for (size_t c = 0; c <= VolumeDim; ++c) {
        bc_dt_u_minus->get(a, b) +=
            0.5 * char_speeds.at(3) *
            (constraint_char_zero_minus.get(c) -
             mMu * constraint_char_zero_plus.get(c)) *
            (0.5 * outgoing_null_one_form.get(a) *
                 outgoing_null_one_form.get(b) * incoming_null_vector.get(c) +
             projection_ab.get(a, b) * outgoing_null_vector.get(c) -
             projection_Ab.get(c, b) * outgoing_null_one_form.get(a) -
             projection_Ab.get(c, a) * outgoing_null_one_form.get(b));
      }
    }
  }
  return *bc_dt_u_minus;
}

template <typename ReturnType, size_t VolumeDim>
ReturnType set_dt_v_minus<ReturnType, VolumeDim>::
    apply_bjorhus_constraint_preserving_physical(
        const gsl::not_null<ReturnType*> bc_dt_u_minus,
        const Scalar<DataVector>& gamma2,
        const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
            unit_interface_normal_one_form,
        const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
            unit_interface_normal_vector,
        const tnsr::A<DataVector, VolumeDim, Frame::Inertial>&
            spacetime_unit_normal_vector,
        const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>& projection_ab,
        const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
        const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>& projection_AB,
        const tnsr::II<DataVector, VolumeDim, Frame::Inertial>&
            inverse_spatial_metric,
        const tnsr::ii<DataVector, VolumeDim, Frame::Inertial>&
            extrinsic_curvature,
        const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
            spacetime_metric,
        const tnsr::AA<DataVector, VolumeDim, Frame::Inertial>&
            inverse_spacetime_metric,
        const typename Tags::ThreeIndexConstraint<
            VolumeDim, Frame::Inertial>::type& three_index_constraint,
        const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
            char_projected_rhs_dt_u_minus,
        const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& phi,
        const tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>& d_phi,
        const tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>& d_pi,
        const std::array<DataVector, 4>& char_speeds) noexcept {
  ASSERT(get_size(get<0, 0>(*bc_dt_u_minus)) == get_size(get(gamma2)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0>(*bc_dt_u_minus)) << ","
             << get_size(get(gamma2)));
  // hard-coded value from SpEC Bbh input file Mu = MuPhys = 0
  const double mMuPhys = 0.;
  const bool mAdjustPhysUsingC4 = true;
  const bool mGamma2InPhysBc = true;

  // In what follows, we use the notation of Kidder, Scheel & Teukolsky
  // (2001) https://arxiv.org/pdf/gr-qc/0105031.pdf. We will refer to this
  // article as KST henceforth, and use the abbreviation in variable names
  // to disambiguate their origin.
  auto U8p = make_with_value<tnsr::ii<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  auto U8m = make_with_value<tnsr::ii<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  {
    // D_(k,i,j) = (1/2) \partial_k g_(ij) and its derivative
    tnsr::ijj<DataVector, VolumeDim, Frame::Inertial> spatial_phi(
        get_size(get(gamma2)));
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = i; j < VolumeDim; ++j) {
        for (size_t k = 0; k < VolumeDim; ++k) {
          spatial_phi.get(k, i, j) = phi.get(k, i + 1, j + 1);
        }
      }
    }

    // Compute spatial Ricci tensor
    auto ricci_3 = GeneralizedHarmonic::spatial_ricci_tensor(
        phi, d_phi, inverse_spatial_metric);
    tnsr::ijj<DataVector, VolumeDim, Frame::Inertial> CdK(
        get_size(get(gamma2)));
    {
      const auto christoffel_first_kind =
          gr::christoffel_first_kind(spatial_phi);
      const auto christoffel_second_kind = raise_or_lower_first_index(
          christoffel_first_kind, inverse_spatial_metric);

      // Ordinary derivative first
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            CdK.get(k, i, j) = 0.5 * d_pi.get(k, i + 1, j + 1);
            for (size_t a = 0; a <= VolumeDim; ++a) {
              CdK.get(k, i, j) +=
                  0.5 *
                  (d_phi.get(k, i, j + 1, a) + d_phi.get(k, j, i + 1, a)) *
                  spacetime_unit_normal_vector.get(a);
              for (size_t b = 0; b <= VolumeDim; ++b) {
                for (size_t c = 0; c <= VolumeDim; ++c) {
                  CdK.get(k, i, j) -=
                      0.5 * (phi.get(i, j + 1, a) + phi.get(j, i + 1, a)) *
                      spacetime_unit_normal_vector.get(b) *
                      (inverse_spacetime_metric.get(c, a) +
                       0.5 * spacetime_unit_normal_vector.get(c) *
                           spacetime_unit_normal_vector.get(a)) *
                      phi.get(k, c, b);
                }
              }
            }
          }
        }
      }

      // Now add gamma terms
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            for (size_t l = 0; l < VolumeDim; ++l) {
              CdK.get(k, i, j) -= christoffel_second_kind.get(l, i, k) *
                                      extrinsic_curvature.get(l, j) +
                                  christoffel_second_kind.get(l, j, k) *
                                      extrinsic_curvature.get(l, i);
            }
          }
        }
      }
    }

    if (mAdjustPhysUsingC4) {
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
      // These compensate for some of the CdK terms.
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = i; j < VolumeDim; ++j) {
          for (size_t a = 0; a <= VolumeDim; ++a) {
            for (size_t k = 0; k < VolumeDim; ++k) {
              ricci_3.get(i, j) +=
                  0.5 * unit_interface_normal_vector.get(k + 1) *
                  spacetime_unit_normal_vector.get(a) *
                  (d_phi.get(i, k, j + 1, a) - d_phi.get(k, i, j + 1, a) +
                   d_phi.get(j, k, i + 1, a) - d_phi.get(k, j, i + 1, a));
            }
          }
        }
      }
    }

    TempBuffer<tmpl::list<
        // spatial projection operators P_ij, P^ij, and P^i_j
        ::Tags::TempII<0, VolumeDim, Frame::Inertial, DataVector>,
        ::Tags::Tempii<1, VolumeDim, Frame::Inertial, DataVector>,
        ::Tags::TempIj<2, VolumeDim, Frame::Inertial, DataVector>>>
        local_buffer(get_size(get<0>(unit_interface_normal_vector)));
    auto& spatial_projection_IJ =
        get<::Tags::TempII<0, VolumeDim, Frame::Inertial, DataVector>>(
            local_buffer);
    auto& spatial_projection_ij =
        get<::Tags::Tempii<1, VolumeDim, Frame::Inertial, DataVector>>(
            local_buffer);
    auto& spatial_projection_Ij =
        get<::Tags::TempIj<2, VolumeDim, Frame::Inertial, DataVector>>(
            local_buffer);

    // Make spatial projection operators
    GeneralizedHarmonic::spatial_projection_tensor(
        make_not_null(&spatial_projection_IJ), inverse_spatial_metric,
        unit_interface_normal_vector);

    const auto spatial_metric = [&spacetime_metric]() noexcept {
      tnsr::ii<DataVector, VolumeDim, Frame::Inertial> tmp_metric(
          get_size(get<0, 0>(spacetime_metric)));
      for (size_t j = 0; j < VolumeDim; ++j) {
        for (size_t k = j; k < VolumeDim; ++k) {
          tmp_metric.get(j, k) = spacetime_metric.get(1 + j, 1 + k);
        }
      }
      return tmp_metric;
    }
    ();
    GeneralizedHarmonic::spatial_projection_tensor(
        make_not_null(&spatial_projection_ij), spatial_metric,
        unit_interface_normal_one_form);

    GeneralizedHarmonic::spatial_projection_tensor(
        make_not_null(&spatial_projection_Ij), unit_interface_normal_vector,
        unit_interface_normal_one_form);

    GeneralizedHarmonic::weyl_propagating(
        make_not_null(&U8p), ricci_3, extrinsic_curvature,
        inverse_spatial_metric, CdK, unit_interface_normal_vector,
        spatial_projection_IJ, spatial_projection_ij, spatial_projection_Ij, 1);
    GeneralizedHarmonic::weyl_propagating(
        make_not_null(&U8m), ricci_3, extrinsic_curvature,
        inverse_spatial_metric, CdK, unit_interface_normal_vector,
        spatial_projection_IJ, spatial_projection_ij, spatial_projection_Ij,
        -1);
  }

  auto U3p = make_with_value<tnsr::aa<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  auto U3m = make_with_value<tnsr::aa<DataVector, VolumeDim, Frame::Inertial>>(
      unit_interface_normal_vector, 0.);
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t i = 0; i < VolumeDim; ++i) {
        for (size_t j = 0; j < VolumeDim; ++j) {
          U3p.get(a, b) += 2.0 * projection_Ab.get(i + 1, a) *
                           projection_Ab.get(j + 1, b) * U8p.get(i, j);
          U3m.get(a, b) += 2.0 * projection_Ab.get(i + 1, a) *
                           projection_Ab.get(j + 1, b) * U8m.get(i, j);
        }
      }
    }
  }

  // Impose physical boundary condition
  if (mGamma2InPhysBc) {
    DataVector tmp(get_size(get<0>(unit_interface_normal_vector)));
    for (size_t c = 0; c <= VolumeDim; ++c) {
      for (size_t d = c; d <= VolumeDim; ++d) {
        for (size_t a = 0; a <= VolumeDim; ++a) {
          for (size_t b = 0; b <= VolumeDim; ++b) {
            tmp = 0.;
            for (size_t i = 0; i < VolumeDim; ++i) {
              tmp += unit_interface_normal_vector.get(i + 1) *
                     three_index_constraint.get(i, a, b);
            }
            tmp *= get(gamma2);
            bc_dt_u_minus->get(c, d) +=
                (projection_Ab.get(a, c) * projection_Ab.get(b, d) -
                 0.5 * projection_ab.get(c, d) * projection_AB.get(a, b)) *
                (char_projected_rhs_dt_u_minus.get(a, b) +
                 char_speeds.at(3) *
                     (U3m.get(a, b) - tmp - mMuPhys * U3p.get(a, b)));
          }
        }
      }
    }
  } else {
    for (size_t c = 0; c <= VolumeDim; ++c) {
      for (size_t d = c; d <= VolumeDim; ++d) {
        for (size_t a = 0; a <= VolumeDim; ++a) {
          for (size_t b = 0; b <= VolumeDim; ++b) {
            bc_dt_u_minus->get(c, d) +=
                (projection_Ab.get(a, c) * projection_Ab.get(b, d) -
                 0.5 * projection_ab.get(c, d) * projection_AB.get(a, b)) *
                (char_projected_rhs_dt_u_minus.get(a, b) +
                 char_speeds.at(3) * (U3m.get(a, b) - mMuPhys * U3p.get(a, b)));
          }
        }
      }
    }
  }

  return *bc_dt_u_minus;
}

template <typename ReturnType, size_t VolumeDim>
ReturnType set_dt_v_minus<ReturnType, VolumeDim>::apply_gauge_sommerfeld(
    const gsl::not_null<ReturnType*> bc_dt_u_minus,
    const Scalar<DataVector>& gamma2,
    const typename domain::Tags::Coordinates<VolumeDim, Frame::Inertial>::type&
        inertial_coords,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        incoming_null_one_form,
    const tnsr::a<DataVector, VolumeDim, Frame::Inertial>&
        outgoing_null_one_form,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& incoming_null_vector,
    const tnsr::A<DataVector, VolumeDim, Frame::Inertial>& outgoing_null_vector,
    const tnsr::Ab<DataVector, VolumeDim, Frame::Inertial>& projection_Ab,
    const tnsr::aa<DataVector, VolumeDim, Frame::Inertial>&
        char_projected_rhs_dt_u_psi) noexcept {
  ASSERT(get_size(get<0, 0>(*bc_dt_u_minus)) ==
             get_size(get<0>(incoming_null_one_form)),
         "Size of input variables and temporary memory do not match: "
             << get_size(get<0, 0>(*bc_dt_u_minus)) << ","
             << get_size(get<0>(incoming_null_one_form)));
  // gauge_bc_coeff below is hard-coded here to its default value in SpEC
  constexpr double gauge_bc_coeff = 1.;

  DataVector inertial_radius(get_size(get<0>(inertial_coords)), 0.);
  for (size_t i = 0; i < VolumeDim; ++i) {
    inertial_radius += square(inertial_coords.get(i));
  }
  inertial_radius = sqrt(inertial_radius);

  // add in gauge BC
  for (size_t a = 0; a <= VolumeDim; ++a) {
    for (size_t b = a; b <= VolumeDim; ++b) {
      for (size_t c = 0; c <= VolumeDim; ++c) {
        for (size_t d = 0; d <= VolumeDim; ++d) {
          bc_dt_u_minus->get(a, b) +=
              0.5 *
              (incoming_null_one_form.get(a) * projection_Ab.get(c, b) *
                   outgoing_null_vector.get(d) +
               incoming_null_one_form.get(b) * projection_Ab.get(c, a) *
                   outgoing_null_vector.get(d) -
               0.5 * incoming_null_one_form.get(a) *
                   outgoing_null_one_form.get(b) * incoming_null_vector.get(c) *
                   outgoing_null_vector.get(d) -
               0.5 * incoming_null_one_form.get(b) *
                   outgoing_null_one_form.get(a) * incoming_null_vector.get(c) *
                   outgoing_null_vector.get(d) -
               0.5 * incoming_null_one_form.get(a) *
                   incoming_null_one_form.get(b) * outgoing_null_vector.get(c) *
                   outgoing_null_vector.get(d)) *
              (get(gamma2) - gauge_bc_coeff * (1. / inertial_radius)) *
              char_projected_rhs_dt_u_psi.get(c, d);
        }
      }
    }
  }

  return *bc_dt_u_minus;
}

}  // namespace BoundaryConditions_detail
}  // namespace Actions
}  // namespace GeneralizedHarmonic
