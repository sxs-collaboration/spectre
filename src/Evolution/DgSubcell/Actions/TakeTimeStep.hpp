// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::subcell::fd::Actions {
/*!
 * \brief Take a finite-difference time step on the subcell grid.
 *
 * The template parameter `TimeDerivative` must have a `static apply` function
 * that takes the `DataBox` by `gsl::not_null` as the first argument, the
 * cell-centered inverse Jacobian from the logical to the grid frame as the
 * second argument, and its determinant as the third argument.
 *
 * GlobalCache: nothing
 *
 * DataBox:
 * - Uses:
 *   - `subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>`
 *   - `subcell::fd::Tags::DetInverseJacobianLogicalToGrid`
 *   - `domain::Tags::ElementMap<Dim, Frame::Grid>`
 *   - `domain::CoordinateMaps::Tags::CoordinateMap<Dim, Grid, Inertial>`
 *   - `subcell::Tags::Coordinates<Dim, Frame::Logical>`
 *   - Anything that `Metavariables::SubcellOptions::TimeDerivative` uses
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - `subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>`
 *   - `subcell::fd::Tags::DetInverseJacobianLogicalToGrid`
 *   - Anything that `Metavariables::SubcellOptions::TimeDerivative` modifies
 */
template <typename TimeDerivative>
struct TakeTimeStep {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, size_t Dim = Metavariables::volume_dim>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ASSERT((db::get<::domain::CoordinateMaps::Tags::CoordinateMap<
                Dim, Frame::Grid, Frame::Inertial>>(box))
               .is_identity(),
           "Do not yet support moving mesh with DG-subcell.");
    db::mutate<fd::Tags::InverseJacobianLogicalToGrid<Dim>,
               fd::Tags::DetInverseJacobianLogicalToGrid>(
        make_not_null(&box),
        [](const auto inv_jac_ptr, const auto det_inv_jac_ptr,
           const auto& logical_to_grid_map,
           const auto& logical_coords) noexcept {
          if (not inv_jac_ptr->has_value()) {
            *inv_jac_ptr = logical_to_grid_map.inv_jacobian(logical_coords);
            *det_inv_jac_ptr = determinant(**inv_jac_ptr);
          }
        },
        db::get<::domain::Tags::ElementMap<Dim, Frame::Grid>>(box),
        db::get<subcell::Tags::Coordinates<Dim, Frame::Logical>>(box));

    TimeDerivative::apply(
        make_not_null(&box),
        *db::get<fd::Tags::InverseJacobianLogicalToGrid<Dim>>(box),
        *db::get<fd::Tags::DetInverseJacobianLogicalToGrid>(box));

    db::mutate<evolution::dg::Tags::MortarData<Dim>>(
        make_not_null(&box), [](const auto mortar_data_ptr) noexcept {
          for (auto& data : *mortar_data_ptr) {
            data.second = evolution::dg::MortarData<Dim>{};
          }
        });
    return {std::move(box)};
  }
};
}  // namespace evolution::dg::subcell::fd::Actions
