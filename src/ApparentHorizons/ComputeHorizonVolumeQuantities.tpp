// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"

#pragma once

#include <cstdint>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Transform.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {

/// Single frame case
template <typename SrcTagList, typename DestTagList>
void ComputeHorizonVolumeQuantities::apply(
    const gsl::not_null<Variables<DestTagList>*> target_vars,
    const Variables<SrcTagList>& src_vars, const Mesh<3>& /*mesh*/) {
  static_assert(
      std::is_same_v<tmpl::list_difference<SrcTagList, allowed_src_tags>,
                     tmpl::list<>>,
      "Found a src tag that is not allowed. If you want to add a new src tag "
      "to allowed_src_tags, you also need to add code in "
      "ComputeHorizonVolumeQuantities.tpp to process that src tag.");
  static_assert(
      std::is_same_v<tmpl::list_difference<required_src_tags, SrcTagList>,
                     tmpl::list<>>,
      "A required src tag is missing");

  static_assert(
      std::is_same_v<tmpl::list_difference<DestTagList,
                                           allowed_dest_tags<Frame::Inertial>>,
                     tmpl::list<>>,
      "Found a dest tag that is not allowed. If you want to add a new dest tag "
      "to allowed_dest_tags, you also need to add code in "
      "ComputeHorizonVolumeQuantities.tpp to compute that dest tag.");
  static_assert(
      std::is_same_v<tmpl::list_difference<required_dest_tags<Frame::Inertial>,
                                           DestTagList>,
                     tmpl::list<>>,
      "A required dest tag is missing");

  const auto& psi = get<gr::Tags::SpacetimeMetric<DataVector, 3>>(src_vars);
  const auto& pi = get<gh::Tags::Pi<DataVector, 3>>(src_vars);
  const auto& phi = get<gh::Tags::Phi<DataVector, 3>>(src_vars);

  if (target_vars->number_of_grid_points() !=
      src_vars.number_of_grid_points()) {
    target_vars->initialize(src_vars.number_of_grid_points());
  }

  using metric_tag = gr::Tags::SpatialMetric<DataVector, 3>;
  using inv_metric_tag = gr::Tags::InverseSpatialMetric<DataVector, 3>;
  using lapse_tag = ::Tags::TempScalar<0, DataVector>;
  using shift_tag = ::Tags::TempI<1, 3, Frame::Inertial, DataVector>;
  using spacetime_normal_vector_tag =
      ::Tags::TempA<2, 3, Frame::Inertial, DataVector>;

  // All of the temporary tags, including some that may be repeated
  // in the target_variables (for now).
  using full_temp_tags_list =
      tmpl::list<metric_tag, inv_metric_tag, lapse_tag, shift_tag,
                 spacetime_normal_vector_tag>;

  // temp tags without variables that are already in DestTagList.
  using temp_tags_list =
      tmpl::list_difference<full_temp_tags_list, DestTagList>;
  TempBuffer<temp_tags_list> buffer(get<0, 0>(psi).size());

  auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(*target_vars);
  auto& spatial_christoffel_second_kind =
      get<gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>>(*target_vars);

  // These are always temporaries.
  auto& lapse = get<lapse_tag>(buffer);
  auto& shift = get<shift_tag>(buffer);
  auto& spacetime_normal_vector = get<spacetime_normal_vector_tag>(buffer);

  // These may or may not be temporaries
  auto& metric = *(get<metric_tag>(target_vars, make_not_null(&buffer)));
  auto& inv_metric =
      *(get<inv_metric_tag>(target_vars, make_not_null(&buffer)));

  // Actual computation starts here

  gr::spatial_metric(make_not_null(&metric), psi);
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse), make_not_null(&inv_metric),
                          metric);
  gr::shift(make_not_null(&shift), psi, inv_metric);
  gr::lapse(make_not_null(&lapse), shift, psi);
  gr::spacetime_normal_vector(make_not_null(&spacetime_normal_vector), lapse,
                              shift);
  gh::extrinsic_curvature(make_not_null(&extrinsic_curvature),
                          spacetime_normal_vector, pi, phi);
  gh::christoffel_second_kind(make_not_null(&spatial_christoffel_second_kind),
                              phi, inv_metric);

  if constexpr (tmpl::list_contains_v<DestTagList,
                                      gr::Tags::SpatialRicci<DataVector, 3>>) {
    static_assert(
        tmpl::list_contains_v<SrcTagList,
                              Tags::deriv<gh::Tags::Phi<DataVector, 3>,
                                          tmpl::size_t<3>, Frame::Inertial>>,
        "If Ricci is requested, SrcTags must include deriv of Phi");
    auto& spatial_ricci =
        get<gr::Tags::SpatialRicci<DataVector, 3>>(*target_vars);
    gh::spatial_ricci_tensor(
        make_not_null(&spatial_ricci), phi,
        get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(src_vars),
        inv_metric);
  }
}

/// Dual frame case
template <typename SrcTagList, typename DestTagList, typename TargetFrame>
void ComputeHorizonVolumeQuantities::apply(
    const gsl::not_null<Variables<DestTagList>*> target_vars,
    const Variables<SrcTagList>& src_vars, const Mesh<3>& mesh,
    const Jacobian<DataVector, 3, TargetFrame, Frame::Inertial>&
        jac_target_to_inertial,
    const InverseJacobian<DataVector, 3, TargetFrame, Frame::Inertial>&
    /*invjac_target_to_inertial*/,
    const Jacobian<DataVector, 3, Frame::ElementLogical, TargetFrame>&
    /*jac_logical_to_target*/,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical, TargetFrame>&
        invjac_logical_to_target,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*inertial_mesh_velocity*/,
    const tnsr::I<DataVector, 3, TargetFrame>&
    /*grid_to_target_frame_mesh_velocity*/) {
  static_assert(
      std::is_same_v<tmpl::list_difference<SrcTagList, allowed_src_tags>,
                     tmpl::list<>>,
      "Found a src tag that is not allowed. If you want to add a new src tag "
      "to allowed_src_tags, you also need to add code in "
      "ComputeHorizonVolumeQuantities.tpp to process that src tag.");
  static_assert(
      std::is_same_v<tmpl::list_difference<required_src_tags, SrcTagList>,
                     tmpl::list<>>,
      "A required src tag is missing");

  static_assert(
      std::is_same_v<
          tmpl::list_difference<DestTagList, allowed_dest_tags<TargetFrame>>,
          tmpl::list<>>,
      "Found a dest tag that is not allowed. If you want to add a new dest tag "
      "to allowed_dest_tags, you also need to add code in "
      "ComputeHorizonVolumeQuantities.tpp to compute that dest tag.");
  static_assert(
      std::is_same_v<
          tmpl::list_difference<required_dest_tags<TargetFrame>, DestTagList>,
          tmpl::list<>>,
      "A required dest tag is missing");

  const auto& psi = get<gr::Tags::SpacetimeMetric<DataVector, 3>>(src_vars);
  const auto& pi = get<gh::Tags::Pi<DataVector, 3>>(src_vars);
  const auto& phi = get<gh::Tags::Phi<DataVector, 3>>(src_vars);

  if (target_vars->number_of_grid_points() !=
      src_vars.number_of_grid_points()) {
    target_vars->initialize(src_vars.number_of_grid_points());
  }

  using metric_tag = gr::Tags::SpatialMetric<DataVector, 3, TargetFrame>;
  using inv_metric_tag =
      gr::Tags::InverseSpatialMetric<DataVector, 3, TargetFrame>;
  using lapse_tag = ::Tags::TempScalar<0, DataVector>;
  using shift_tag = ::Tags::TempI<1, 3, Frame::Inertial, DataVector>;
  using spacetime_normal_vector_tag =
      ::Tags::TempA<2, 3, Frame::Inertial, DataVector>;

  // Additional temporary tags used for multiple frames
  using inertial_metric_tag = gr::Tags::SpatialMetric<DataVector, 3>;
  using inertial_inv_metric_tag = gr::Tags::InverseSpatialMetric<DataVector, 3>;
  using inertial_ex_curvature_tag = gr::Tags::ExtrinsicCurvature<DataVector, 3>;
  using logical_deriv_metric_tag = ::Tags::TempTensor<
      6, Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                index_list<SpatialIndex<3, UpLo::Lo, Frame::ElementLogical>,
                           SpatialIndex<3, UpLo::Lo, TargetFrame>,
                           SpatialIndex<3, UpLo::Lo, TargetFrame>>>>;
  using deriv_metric_tag = ::Tags::Tempijj<7, 3, TargetFrame, DataVector>;
  using inertial_spatial_ricci_tag = gr::Tags::SpatialRicci<DataVector, 3>;

  // All of the temporary tags, including some that may be repeated
  // in the target_variables (for now).
  using temp_tags_list_no_ricci =
      tmpl::list<metric_tag, inv_metric_tag, lapse_tag, shift_tag,
                 spacetime_normal_vector_tag, inertial_metric_tag,
                 inertial_inv_metric_tag, inertial_ex_curvature_tag,
                 logical_deriv_metric_tag, deriv_metric_tag>;
  using full_temp_tags_list = std::conditional_t<
      tmpl::list_contains_v<DestTagList,
                            gr::Tags::SpatialRicci<DataVector, 3, TargetFrame>>,
      tmpl::push_back<temp_tags_list_no_ricci, inertial_spatial_ricci_tag>,
      temp_tags_list_no_ricci>;

  auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3, TargetFrame>>(
          *target_vars);
  auto& spatial_christoffel_second_kind =
      get<gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, TargetFrame>>(
          *target_vars);

  // temp tags without variables that are already in DestTagList.
  using temp_tags_list =
      tmpl::list_difference<full_temp_tags_list, DestTagList>;
  TempBuffer<temp_tags_list> buffer(get<0, 0>(psi).size());

  // These are always temporaries.
  auto& lapse = get<lapse_tag>(buffer);
  auto& shift = get<shift_tag>(buffer);
  auto& spacetime_normal_vector = get<spacetime_normal_vector_tag>(buffer);
  auto& logical_deriv_metric = get<logical_deriv_metric_tag>(buffer);
  auto& deriv_metric = get<deriv_metric_tag>(buffer);

  // These may or may not be temporaries, depending on if they are asked for
  // in target_vars.
  auto& inertial_metric =
      *(get<inertial_metric_tag>(target_vars, make_not_null(&buffer)));
  auto& inertial_inv_metric =
      *(get<inertial_inv_metric_tag>(target_vars, make_not_null(&buffer)));
  auto& inertial_ex_curvature =
      *(get<inertial_ex_curvature_tag>(target_vars, make_not_null(&buffer)));
  auto& metric = *(get<metric_tag>(target_vars, make_not_null(&buffer)));
  auto& inv_metric =
      *(get<inv_metric_tag>(target_vars, make_not_null(&buffer)));

  // Actual computation starts here

  gr::spatial_metric(make_not_null(&inertial_metric), psi);
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse),
                          make_not_null(&inertial_inv_metric), inertial_metric);

  // Compute inertial extrinsic curvature
  gr::shift(make_not_null(&shift), psi, inertial_inv_metric);
  gr::lapse(make_not_null(&lapse), shift, psi);
  gr::spacetime_normal_vector(make_not_null(&spacetime_normal_vector), lapse,
                              shift);
  gh::extrinsic_curvature(make_not_null(&inertial_ex_curvature),
                          spacetime_normal_vector, pi, phi);

  // Transform spatial metric and extrinsic curvature
  transform::to_different_frame(make_not_null(&metric), inertial_metric,
                                jac_target_to_inertial);
  transform::to_different_frame(make_not_null(&extrinsic_curvature),
                                inertial_ex_curvature, jac_target_to_inertial);

  // Invert transformed 3-metric.
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse), make_not_null(&inv_metric),
                          metric);

  // Differentiate 3-metric.
  logical_partial_derivative(make_not_null(&logical_deriv_metric), metric,
                             mesh);
  transform::first_index_to_different_frame(make_not_null(&deriv_metric),
                                            logical_deriv_metric,
                                            invjac_logical_to_target);
  gr::christoffel_second_kind(make_not_null(&spatial_christoffel_second_kind),
                              deriv_metric, inv_metric);

  // We need SpatialChristoffelSecondKind in the inertial frame only if
  // we ask for it; otherwise we don't need to compute it at all.
  if constexpr (tmpl::list_contains_v<
                    DestTagList,
                    gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>>) {
    auto& inertial_christoffel_second_kind =
        get<gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>>(
            *target_vars);
    gh::christoffel_second_kind(
        make_not_null(&inertial_christoffel_second_kind), phi,
        inertial_inv_metric);
  }

  // We need SpatialRicci only if we ask for it (in either frame);
  // otherwise we don't need to compute it at all.
  if constexpr (tmpl::list_contains_v<DestTagList,
                                      gr::Tags::SpatialRicci<DataVector, 3>> or
                tmpl::list_contains_v<
                    DestTagList,
                    gr::Tags::SpatialRicci<DataVector, 3, TargetFrame>>) {
    static_assert(
        tmpl::list_contains_v<SrcTagList,
                              Tags::deriv<gh::Tags::Phi<DataVector, 3>,
                                          tmpl::size_t<3>, Frame::Inertial>>,
        "If Ricci is requested, SrcTags must include deriv of Phi");

    auto& inertial_spatial_ricci =
        *(get<inertial_spatial_ricci_tag>(target_vars, make_not_null(&buffer)));
    gh::spatial_ricci_tensor(
        make_not_null(&inertial_spatial_ricci), phi,
        get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(src_vars),
        inertial_inv_metric);

    if constexpr (tmpl::list_contains_v<
                      DestTagList,
                      gr::Tags::SpatialRicci<DataVector, 3, TargetFrame>>) {
      auto& spatial_ricci =
          get<gr::Tags::SpatialRicci<DataVector, 3, TargetFrame>>(*target_vars);
      transform::to_different_frame(make_not_null(&spatial_ricci),
                                    inertial_spatial_ricci,
                                    jac_target_to_inertial);
    }
  }
}

}  // namespace ah
