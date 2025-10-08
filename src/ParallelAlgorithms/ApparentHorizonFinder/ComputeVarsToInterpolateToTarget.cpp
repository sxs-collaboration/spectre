// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeVarsToInterpolateToTarget.hpp"

#include <type_traits>

#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/FrameTransform.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Composition.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {
namespace {
// The jacobians can be omitted if the frame is truly the inertial one since
// they won't be needed
template <bool ForceInertialFrame = false, typename Fr>
void compute_vars_to_interpolate_to_target_impl(
    const gsl::not_null<Variables<ah::vars_to_interpolate_to_target<3, Fr>>*>
        target_vars,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::ijaa<DataVector, 3>& deriv_phi, const Mesh<3>& mesh,
    const Jacobian<DataVector, 3, Fr, Frame::Inertial>& jac_frame_to_inertial =
        {},
    const InverseJacobian<DataVector, 3, Frame::ElementLogical, Fr>&
        invjac_logical_to_frame = {}) {
  // Handle the time-independent case where the frame could be anything, but we
  // only need the inertial computation
  using FrameToUse =
      tmpl::conditional_t<ForceInertialFrame, ::Frame::Inertial, Fr>;
  Variables<ah::vars_to_interpolate_to_target<3, FrameToUse>>
      vars_to_interpolate_to_target;
  vars_to_interpolate_to_target.set_data_ref(target_vars->data(),
                                             target_vars->size());

  // Inertial tags
  using inertial_metric_tag = gr::Tags::SpatialMetric<DataVector, 3>;
  using inertial_inv_metric_tag = gr::Tags::InverseSpatialMetric<DataVector, 3>;
  using inertial_shift_tag = ::Tags::TempI<1, 3, Frame::Inertial, DataVector>;
  using inertial_ex_curvature_tag = gr::Tags::ExtrinsicCurvature<DataVector, 3>;
  using inertial_spatial_ricci_tag = gr::Tags::SpatialRicci<DataVector, 3>;
  using inertial_spacetime_normal_vector_tag =
      ::Tags::TempA<2, 3, Frame::Inertial, DataVector>;
  using inertial_spatial_christoffel_tag =
      gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>;
  // Frame tags
  using metric_tag = gr::Tags::SpatialMetric<DataVector, 3, FrameToUse>;
  using inv_metric_tag =
      gr::Tags::InverseSpatialMetric<DataVector, 3, FrameToUse>;
  using lapse_tag = gr::Tags::Lapse<DataVector>;
  using phi_tag = ::Tags::Tempijj<7, 3, FrameToUse, DataVector>;
  using ex_curvature_tag =
      gr::Tags::ExtrinsicCurvature<DataVector, 3, FrameToUse>;
  using spatial_christoffel_tag =
      gr::Tags::SpatialChristoffelSecondKind<DataVector, 3, FrameToUse>;
  using spatial_ricci_tag = gr::Tags::SpatialRicci<DataVector, 3, FrameToUse>;

  TempBuffer<tmpl::list<
      inertial_metric_tag, inertial_inv_metric_tag, lapse_tag,
      inertial_shift_tag, inertial_spacetime_normal_vector_tag,
      inertial_ex_curvature_tag, inertial_spatial_ricci_tag, phi_tag>>
      buffer{get<0, 0>(spacetime_metric).size()};

  auto& lapse = get<lapse_tag>(buffer);
  auto& inertial_shift = get<inertial_shift_tag>(buffer);
  auto& inertial_spacetime_normal_vector =
      get<inertial_spacetime_normal_vector_tag>(buffer);

  // As an optimization, get these from the vars to interpolate if they are in
  // the inertial frame so we don't have to do copies below. Otherwise just
  // grab them from the temporary buffer.
  auto& inertial_metric = *(get<inertial_metric_tag>(
      make_not_null(&vars_to_interpolate_to_target), make_not_null(&buffer)));
  auto& inertial_inv_metric = *(get<inertial_inv_metric_tag>(
      make_not_null(&vars_to_interpolate_to_target), make_not_null(&buffer)));
  auto& inertial_ex_curvature = *(get<inertial_ex_curvature_tag>(
      make_not_null(&vars_to_interpolate_to_target), make_not_null(&buffer)));
  auto& inertial_spatial_ricci = *(get<inertial_spatial_ricci_tag>(
      make_not_null(&vars_to_interpolate_to_target), make_not_null(&buffer)));

  gr::spatial_metric(make_not_null(&inertial_metric), spacetime_metric);
  // put determinant of 3-metric temporarily into lapse to save memory.
  determinant_and_inverse(make_not_null(&lapse),
                          make_not_null(&inertial_inv_metric), inertial_metric);

  // Compute inertial extrinsic curvature
  gr::shift(make_not_null(&inertial_shift), spacetime_metric,
            inertial_inv_metric);
  gr::lapse(make_not_null(&lapse), inertial_shift, spacetime_metric);
  gr::spacetime_normal_vector(make_not_null(&inertial_spacetime_normal_vector),
                              lapse, inertial_shift);
  gh::extrinsic_curvature(make_not_null(&inertial_ex_curvature),
                          inertial_spacetime_normal_vector, pi, phi);

  // Compute inertial spatial ricci
  gh::spatial_ricci_tensor(make_not_null(&inertial_spatial_ricci), phi,
                           deriv_phi, inertial_inv_metric);

  if constexpr (std::is_same_v<FrameToUse, ::Frame::Inertial>) {
    (void)jac_frame_to_inertial;
    (void)invjac_logical_to_frame;

    auto& inertial_christoffel_second_kind =
        get<inertial_spatial_christoffel_tag>(vars_to_interpolate_to_target);
    gh::christoffel_second_kind(
        make_not_null(&inertial_christoffel_second_kind), phi,
        inertial_inv_metric);
  } else {
    auto& frame_metric = get<metric_tag>(vars_to_interpolate_to_target);
    transform::to_different_frame(make_not_null(&frame_metric), inertial_metric,
                                  jac_frame_to_inertial);

    // Invert transformed 3-metric and put determinant of 3-metric temporarily
    // into lapse to save memory (since we don't need the lapse anymore)
    auto& frame_inv_metric = get<inv_metric_tag>(vars_to_interpolate_to_target);
    determinant_and_inverse(make_not_null(&lapse),
                            make_not_null(&frame_inv_metric), frame_metric);

    // Transform extrinsic curvature
    auto& frame_extrinsic_curvature =
        get<ex_curvature_tag>(vars_to_interpolate_to_target);
    transform::to_different_frame(make_not_null(&frame_extrinsic_curvature),
                                  inertial_ex_curvature, jac_frame_to_inertial);

    // Differentiate 3-metric to compute spatial christoffel
    auto& frame_phi = get<phi_tag>(buffer);
    auto& frame_christoffel_second_kind =
        get<spatial_christoffel_tag>(vars_to_interpolate_to_target);
    partial_derivative(make_not_null(&frame_phi), frame_metric, mesh,
                       invjac_logical_to_frame);
    gr::christoffel_second_kind(make_not_null(&frame_christoffel_second_kind),
                                frame_phi, frame_inv_metric);

    // Transform spatial ricci
    auto& frame_spatial_ricci =
        get<spatial_ricci_tag>(vars_to_interpolate_to_target);
    transform::to_different_frame(make_not_null(&frame_spatial_ricci),
                                  inertial_spatial_ricci,
                                  jac_frame_to_inertial);
  }
}
}  // namespace

template <typename Fr>
void compute_vars_to_interpolate_to_target(
    const gsl::not_null<Variables<ah::vars_to_interpolate_to_target<3, Fr>>*>
        target_vars,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::ijaa<DataVector, 3>& deriv_phi,
    const LinkedMessageId<double>& time, const Domain<3>& domain,
    const Mesh<3>& mesh, const ElementId<3>& element_id,
    const domain::FunctionsOfTimeMap& functions_of_time) {
  // This will only change the size if it isn't already the correct size
  target_vars->initialize(get<0, 0>(spacetime_metric).size());

  const auto& block = domain.blocks()[element_id.block_id()];
  if (block.is_time_dependent()) {
    ASSERT(not functions_of_time.empty(),
           "Functions of time are not passed to "
           "compute_vars_to_interpolate_to_target but the block is "
           "time-dependent.");
    if constexpr (std::is_same_v<Fr, ::Frame::Grid>) {
      const ElementMap<3, Fr> map_logical_to_grid{
          element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
      const auto logical_coords = logical_coordinates(mesh);
      const auto jac_grid_to_inertial =
          block.moving_mesh_grid_to_inertial_map().jacobian(
              map_logical_to_grid(logical_coords), time.id, functions_of_time);
      const auto invjac_logical_to_grid =
          map_logical_to_grid.inv_jacobian(logical_coords);

      compute_vars_to_interpolate_to_target_impl(
          target_vars, spacetime_metric, pi, phi, deriv_phi, mesh,
          jac_grid_to_inertial, invjac_logical_to_grid);
    } else if constexpr (std::is_same_v<Fr, ::Frame::Distorted>) {
      ASSERT(block.has_distorted_frame(),
             "Block doesn't have a distorted frame but this horizon is in the "
             "distorted frame.");
      const domain::CoordinateMaps::Composition
          element_logical_to_distorted_map{
              domain::element_to_block_logical_map(element_id),
              block.moving_mesh_logical_to_grid_map().get_clone(),
              block.moving_mesh_grid_to_distorted_map().get_clone()};
      const domain::CoordinateMaps::Composition element_logical_to_grid_map{
          domain::element_to_block_logical_map(element_id),
          block.moving_mesh_logical_to_grid_map().get_clone()};
      const auto logical_coords = logical_coordinates(mesh);
      const auto jac_distorted_to_inertial =
          block.moving_mesh_distorted_to_inertial_map().jacobian(
              element_logical_to_distorted_map(logical_coords, time.id,
                                               functions_of_time),
              time.id, functions_of_time);
      const auto invjac_logical_to_distorted =
          element_logical_to_distorted_map.inv_jacobian(logical_coords, time.id,
                                                        functions_of_time);

      compute_vars_to_interpolate_to_target_impl(
          target_vars, spacetime_metric, pi, phi, deriv_phi, mesh,
          jac_distorted_to_inertial, invjac_logical_to_distorted);
    } else {
      compute_vars_to_interpolate_to_target_impl(target_vars, spacetime_metric,
                                                 pi, phi, deriv_phi, mesh);
    }
  } else {
    // No frame transformations needed, since the maps are time-independent
    // and therefore all the frames are the same.
    //
    // The frame tags may be different, but the source vars should all be in the
    // Inertial frame, so we force the inertial frame (True template).
    compute_vars_to_interpolate_to_target_impl<true>(
        target_vars, spacetime_metric, pi, phi, deriv_phi, mesh);
  }
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void compute_vars_to_interpolate_to_target(                        \
      const gsl::not_null<                                                    \
          Variables<ah::vars_to_interpolate_to_target<3, FRAME(data)>>*>      \
          target_vars,                                                        \
      const tnsr::aa<DataVector, 3>& spacetime_metric,                        \
      const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi, \
      const tnsr::ijaa<DataVector, 3>& deriv_phi,                             \
      const LinkedMessageId<double>& time, const Domain<3>& domain,           \
      const Mesh<3>& mesh, const ElementId<3>& element_id,                    \
      const domain::FunctionsOfTimeMap& functions_of_time);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Inertial, Frame::Distorted, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah
