// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Actions/ReceiveGhWorldtubeData.hpp"
#include "Evolution/Systems/Cce/Components/WorldtubeBoundary.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
/// \cond
namespace Tags {
template <typename InterpolationTargetTag>
struct Sphere;
}
/// \endcond

namespace callbacks {

/// \brief post_interpolation_callback that calls Cce::ReceiveGhWorldTubeData
///
/// Uses:
/// - DataBox:
///   - `::gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>`
///   - `::gh::Tags::Pi<DataVector, 3>`
///   - `::gh::Tags::Phi<DataVector, 3>`
///
/// Conforms to the intrp::protocols::PostInterpolationCallback protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
///
/// \note This callback requires the temporal ID in an InterpolationTargetTag be
/// a TimeStepId.
template <typename CceEvolutionComponent, typename InterpolationTargetTag,
          bool DuringSelfStart, bool LocalTimeStepping>
struct SendGhWorldtubeData
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    // This is an ERROR and not an ASSERT because even in Release mode we want
    // to stop execution if there were wrong inputs.
    if (Parallel::get<Tags::Sphere<InterpolationTargetTag>>(cache)
            .radii.size() > 1) {
      ERROR("SendGhWorldtubeData expects a single worldtube radius, not "
            << Parallel::get<Tags::Sphere<InterpolationTargetTag>>(cache)
                   .radii.size());
    }

    auto& cce_gh_boundary_component = Parallel::get_parallel_component<
        Cce::GhWorldtubeBoundary<Metavariables>>(cache);
    if constexpr (DuringSelfStart or LocalTimeStepping) {
      Parallel::simple_action<typename Cce::Actions::ReceiveGhWorldtubeData<
          CceEvolutionComponent, DuringSelfStart>>(
          cce_gh_boundary_component, temporal_id,
          db::get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(box),
          db::get<::gh::Tags::Phi<DataVector, 3>>(box),
          db::get<::gh::Tags::Pi<DataVector, 3>>(box));
    } else {
      // We want to avoid interface manager for global time stepping
      Parallel::simple_action<Cce::Actions::SendToEvolution<
          Cce::GhWorldtubeBoundary<Metavariables>, CceEvolutionComponent>>(
          cce_gh_boundary_component, temporal_id,
          db::get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(box),
          db::get<::gh::Tags::Phi<DataVector, 3>>(box),
          db::get<::gh::Tags::Pi<DataVector, 3>>(box));
    }
  }
};
}  // namespace callbacks
}  // namespace intrp
