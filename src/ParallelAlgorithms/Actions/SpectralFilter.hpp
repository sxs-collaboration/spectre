// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace tuples {
template <typename... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace dg::Actions {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Applies the element's spectral volume filter to the evolved variables.
 *
 * Retrieves the per-element filter from `Filters::Tags::SpectralFilter` from
 * the DataBox and applies the volume filter.
 *
 * The filter is skipped when:
 * - the filter is `Filters::None` (explicit no-op), or
 * - both `apply_volume_filter_on_substep()` and
 *   `apply_volume_filter_on_this_step(step_number)` return `false`.
 *
 * The grid-to-inertial Jacobian and its inverse are only retrieved from the
 * DataBox when `Filters::Filter::need_jacobians()` returns `true`; otherwise
 * `std::nullopt` is passed for both arguments.
 *
 * \note Currently the check for whether to filter on every `N` steps is done
 * relative to the start of the current Slab. This means that for GTS,
 * independent of the value of `N` for every `N` steps, every step has a
 * filter applied since GTS has one step per slab.
 *
 * Uses:
 * - DataBox:
 *   - `Filters::Tags::SpectralFilter<Dim, TagList>`
 *   - `Tags::StepNumberWithinSlab`
 *   - `domain::Tags::Mesh<Dim>`
 *   - `domain::Tags::InverseJacobian<Dim, Frame::Grid, Frame::Inertial>`
 *     (only when `need_jacobians()` is `true`)
 *   - `domain::Tags::Jacobian<Dim, Frame::Grid, Frame::Inertial>`
 *     (only when `need_jacobians()` is `true`)
 * - DataBox changes:
 *   - Modifies: `Metavariables::system::variables_tag`
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 */
struct SpectralFilter {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    constexpr size_t Dim = Metavariables::system::volume_dim;
    using variables_tag = typename Metavariables::system::variables_tag;
    using TagList = typename variables_tag::tags_list;

    const auto& filter =
        db::get<Filters::Tags::SpectralFilter<Dim, TagList>>(box);

    const Mesh<Dim>& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    if (not filter.supports_mesh(mesh)) {
      ERROR("The filter '"
            << filter.name() << "' on element "
            << db::get<domain::Tags::Element<Dim>>(box).id() << " with mesh "
            << mesh
            << " does not support that mesh. This is a bug in initialization "
               "or if AMR projection during h-refinement incorrectly adjusted "
               "the filter.");
    }

    // None is explicitly a no-op; skip immediately.
    if (dynamic_cast<const Filters::None<Dim, TagList>*>(&filter) != nullptr) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const auto step_number =
        static_cast<size_t>(db::get<::Tags::StepNumberWithinSlab>(box));
    if (not(filter.apply_volume_filter_on_substep() or
            filter.apply_volume_filter_on_this_step(step_number))) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const size_t num_pts = mesh.number_of_grid_points();

    std::optional<
        InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>
        inv_jac{std::nullopt};
    std::optional<Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>> jac{
        std::nullopt};
    if (filter.need_jacobians()) {
      if constexpr (db::tag_is_retrievable_v<
                        domain::Tags::InverseJacobian<Dim, Frame::Grid,
                                                      Frame::Inertial>,
                        db::DataBox<DbTags>>) {
        const auto& src = db::get<
            domain::Tags::InverseJacobian<Dim, Frame::Grid, Frame::Inertial>>(
            box);
        inv_jac =
            InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>{};
        for (size_t i = 0; i < src.size(); ++i) {
          make_const_view(make_not_null(&std::as_const((*inv_jac)[i])), src[i],
                          0, num_pts);
        }
      } else {
        ERROR(
            "SpectralFilter: Missing "
            "domain::Tags::InverseJacobian<Dim, Grid, Inertial> in the "
            "DataBox. This should be present for both time-dependent and "
            "time-independent evolutions.");
      }
      if constexpr (db::tag_is_retrievable_v<
                        domain::Tags::Jacobian<Dim, Frame::Grid,
                                               Frame::Inertial>,
                        db::DataBox<DbTags>>) {
        const auto& src =
            db::get<domain::Tags::Jacobian<Dim, Frame::Grid, Frame::Inertial>>(
                box);
        jac = Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>{};
        for (size_t i = 0; i < src.size(); ++i) {
          make_const_view(make_not_null(&std::as_const((*jac)[i])), src[i], 0,
                          num_pts);
        }
      } else {
        ERROR(
            "SpectralFilter: Missing "
            "domain::Tags::Jacobian<Dim, Grid, Inertial> in the "
            "DataBox. This should be present for both time-dependent and "
            "time-independent evolutions.");
      }
    }

    db::mutate<variables_tag>(
        [&filter, &mesh, &inv_jac,
         &jac](const gsl::not_null<typename variables_tag::type*> vars) {
          filter.apply_in_volume(vars, mesh, inv_jac, jac);
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace dg::Actions
