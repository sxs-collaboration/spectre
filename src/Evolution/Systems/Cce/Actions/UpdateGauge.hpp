// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Updates all of the gauge quantities associated with the additional
 * regularity-preserving gauge transformation on the boundaries for a new set of
 * Cauchy and partially flat Bondi-like coordinates.
 *
 * \details This action is to be called after `Tags::CauchyCartesianCoords`
 * and `Tags::PartiallyFlatCartesianCoords` have been updated, typically via a
 * time step of a set of coordinate evolution equations. It prepares the
 * gauge quantities in the \ref DataBoxGroup for calls to the individual
 * `GaugeAdjustedBoundaryValue` specializations.
 *
 * Internally, this dispatches to `GaugeUpdateAngularFromCartesian`,
 * `GaugeUpdateJacobianFromCoordinates`, `GaugeUpdateInterpolator`, and
 * `GaugeUpdateOmega` to perform the computations. Refer to the documentation
 * for those mutators for mathematical details.
 */
template <bool EvolvePartiallyFlatCartesianCoordinates>
struct UpdateGauge {
  using const_global_cache_tags = tmpl::list<Tags::LMax>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<GaugeUpdateAngularFromCartesian<
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&box));
    db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
        Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
        Tags::CauchyAngularCoords, Tags::CauchyCartesianCoords>>(
        make_not_null(&box));
    db::mutate_apply<GaugeUpdateInterpolator<Tags::CauchyAngularCoords>>(
        make_not_null(&box));
    db::mutate_apply<
        GaugeUpdateOmega<Tags::PartiallyFlatGaugeC, Tags::PartiallyFlatGaugeD,
                         Tags::PartiallyFlatGaugeOmega>>(make_not_null(&box));

    if constexpr (EvolvePartiallyFlatCartesianCoordinates) {
      db::mutate_apply<
          GaugeUpdateAngularFromCartesian<Tags::PartiallyFlatAngularCoords,
                                          Tags::PartiallyFlatCartesianCoords>>(
          make_not_null(&box));
      db::mutate_apply<GaugeUpdateJacobianFromCoordinates<
          Tags::CauchyGaugeC, Tags::CauchyGaugeD,
          Tags::PartiallyFlatAngularCoords,
          Tags::PartiallyFlatCartesianCoords>>(make_not_null(&box));
      db::mutate_apply<
          GaugeUpdateInterpolator<Tags::PartiallyFlatAngularCoords>>(
          make_not_null(&box));
      db::mutate_apply<GaugeUpdateOmega<Tags::CauchyGaugeC, Tags::CauchyGaugeD,
                                        Tags::CauchyGaugeOmega>>(
          make_not_null(&box));
    }
    return {std::move(box)};
  }
};
}  // namespace Actions
}  // namespace Cce
