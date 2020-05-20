// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic::gauges::Actions {
template <size_t Dim>
struct InitializeDampedHarmonic {
  using frame = Frame::Inertial;

  using const_global_cache_tags = tmpl::list<
      GeneralizedHarmonic::Tags::GaugeHRollOnStartTime,
      GeneralizedHarmonic::Tags::GaugeHRollOnTimeWindow,
      GeneralizedHarmonic::Tags::GaugeHSpatialWeightDecayWidth<frame>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (not db::get<domain::CoordinateMaps::Tags::CoordinateMap<
                Metavariables::volume_dim, Frame::Grid, Frame::Inertial>>(box)
                .is_identity()) {
      ERROR(
          "Cannot use the damped harmonic rollon gauge with a moving mesh "
          "because the rollon is not implemented for a moving mesh. The issue "
          "is that the initial H_a needs to be in the grid frame, and then "
          "transformed to the inertial frame at each time step. Transforming "
          "the spacetime derivative requires spacetime Hessians, which are not "
          "implemented for the maps and there is currently no plan to add them "
          "because we do not need them for anything else.");
    }
    const auto inverse_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::Logical, frame>>(box);

    auto [initial_gauge_h, initial_d4_gauge_h] = impl(
        db::get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
            box),
        db::get<GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(box),
        db::get<GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(box),
        db::get<domain::Tags::Mesh<Dim>>(box), inverse_jacobian);

    // Add gauge tags
    using compute_tags = db::AddComputeTags<
        GeneralizedHarmonic::gauges::DampedHarmonicHCompute<Dim, frame>,
        GeneralizedHarmonic::gauges::SpacetimeDerivDampedHarmonicHCompute<
            Dim, frame>>;

    // Finally, insert gauge related quantities to the box
    return std::make_tuple(
        Initialization::merge_into_databox<
            InitializeDampedHarmonic,
            db::AddSimpleTags<
                GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
                GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim,
                                                                       frame>>,
            compute_tags>(std::move(box), std::move(initial_gauge_h),
                          std::move(initial_d4_gauge_h)));
  }

 private:
  static std::tuple<tnsr::a<DataVector, Dim, Frame::Inertial>,
                    tnsr::ab<DataVector, Dim, Frame::Inertial>>
  impl(const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
       const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
       const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
       const Mesh<Dim>& mesh,
       const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
           inverse_jacobian) noexcept;
};
}  // namespace GeneralizedHarmonic::gauges::Actions
