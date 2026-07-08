// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace Tags {
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
}  // namespace domain

/// \endcond

namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Allocate variables needed for evolution of nonconservative systems
///
/// Uses:
/// - DataBox:
///   * `Tags::Mesh<Dim>`
///
/// DataBox changes:
/// - Adds:
///   * System::variables_tag
///   * `::Tags::Variables<System::auxiliary_variables>` (only if the system
///     declares a non-empty `auxiliary_variables`)
///
/// - Removes: nothing
/// - Modifies: nothing
template <typename System>
struct NonconservativeSystem {
  static_assert(not System::is_in_flux_conservative_form,
                "System is in flux conservative form");
  static constexpr size_t dim = System::volume_dim;
  using variables_tag = typename System::variables_tag;
  using auxiliary_variables =
      evolution::dg::Actions::detail::get_auxiliary_variables_or_default_t<
          System, tmpl::list<>>;
  static constexpr bool has_auxiliary_variables =
      not std::is_same_v<auxiliary_variables, tmpl::list<>>;
  using auxiliary_variables_tag = ::Tags::Variables<auxiliary_variables>;
  using simple_tags = tmpl::conditional_t<
      has_auxiliary_variables,
      db::AddSimpleTags<variables_tag, auxiliary_variables_tag>,
      db::AddSimpleTags<variables_tag>>;
  using compute_tags = db::AddComputeTags<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    using Vars = typename variables_tag::type;
    const size_t number_of_grid_points =
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points();
    if constexpr (has_auxiliary_variables) {
      using AuxVars = typename auxiliary_variables_tag::type;
      Initialization::mutate_assign<simple_tags>(
          make_not_null(&box), Vars{number_of_grid_points},
          AuxVars{number_of_grid_points});
    } else {
      Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 Vars{number_of_grid_points});
    }

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Initialization
