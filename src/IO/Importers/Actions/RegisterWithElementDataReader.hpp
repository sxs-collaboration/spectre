// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Importers/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace importers {
template <typename Metavariables>
struct ElementDataReader;
namespace Actions {
struct RegisterElementWithSelf;
}  // namespace Actions
}  // namespace importers
namespace evolution::dg::subcell {
template <typename DgTag, typename SubcellTag, typename DbTagsList>
const typename DgTag::type& get_active_tag(const db::DataBox<DbTagsList>& box);
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
template <size_t Dim>
struct Mesh;
struct ActiveGrid;
}  // namespace Tags
}  // namespace evolution::dg::subcell
/// \endcond

namespace importers::Actions {

/*!
 * \brief Register an element with the volume data reader component.
 *
 * Invoke this action on each element of an array parallel component to register
 * them for receiving imported volume data.
 *
 * \note If the tags `evolution::dg::subcell::Tags::ActiveGrid` and
 * `evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Inertial>` are
 * retrievable from the DataBox, then interpolation to the FD/subcell grid is
 * possible.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
struct RegisterWithElementDataReader {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& local_reader_component = *Parallel::local_branch(
        Parallel::get_parallel_component<
            importers::ElementDataReader<Metavariables>>(cache));
    auto coords_and_mesh = [&box]() {
      if constexpr (db::tag_is_retrievable_v<
                        evolution::dg::subcell::Tags::ActiveGrid,
                        db::DataBox<DbTagsList>> and
                    db::tag_is_retrievable_v<
                        evolution::dg::subcell::Tags::Coordinates<
                            Dim, Frame::Inertial>,
                        db::DataBox<DbTagsList>>) {
        return std::make_pair(
            evolution::dg::subcell::get_active_tag<
                domain::Tags::Coordinates<Dim, Frame::Inertial>,
                evolution::dg::subcell::Tags::Coordinates<Dim,
                                                          Frame::Inertial>>(
                box),
            evolution::dg::subcell::get_active_tag<
                domain::Tags::Mesh<Dim>,
                evolution::dg::subcell::Tags::Mesh<Dim>>(box));
      } else {
        return std::make_pair(
            db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box),
            db::get<domain::Tags::Mesh<Dim>>(box));
      }
    }();
    Parallel::simple_action<importers::Actions::RegisterElementWithSelf>(
        local_reader_component,
        Parallel::make_array_component_id<ParallelComponent>(array_index),
        std::move(coords_and_mesh));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Invoked on the `importers::ElementDataReader` component to store the
 * registered data.
 *
 * The `importers::Actions::RegisterWithElementDataReader` action, which is
 * performed on each element of an array parallel component, invokes this action
 * on the `importers::ElementDataReader` component.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
struct RegisterElementWithSelf {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, size_t Dim>
  static void apply(
      db::DataBox<DbTagsList>& box,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const Parallel::ArrayComponentId& array_component_id,
      std::pair<tnsr::I<DataVector, Dim, Frame::Inertial>, Mesh<Dim>>
          coords_and_mesh) {
    db::mutate<Tags::RegisteredElements<Dim>>(
        [&array_component_id,
         &coords_and_mesh](const auto registered_elements) {
          (*registered_elements)[array_component_id] =
              std::move(coords_and_mesh);
        },
        make_not_null(&box));
  }
};

}  // namespace importers::Actions
