// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // Needs to be included somewhere and here seems most natural.
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame

namespace domain {
namespace Tags {
template <size_t VolumeDim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
}  // namespace domain
// IWYU pragma: no_forward_declare db::DataBox

namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Allocate variables needed for evolution of conservative systems
///
/// Uses:
/// - DataBox:
///   * `Tags::Mesh<Dim>`
///
/// DataBox changes:
/// - Adds:
///   * System::variables_tag
///   * db::add_tag_prefix<Tags::Flux, System::variables_tag>
///   * db::add_tag_prefix<Tags::Source, System::variables_tag>
///
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note This action relies on the `SetupDataBox` aggregated initialization
/// mechanism, so `Actions::SetupDataBox` must be present in the
/// `Initialization` phase action list prior to this action.
template <typename System, typename EquationOfStateTag = NoSuchType>
struct ConservativeSystem {
 private:
  static constexpr size_t dim = System::volume_dim;

  using variables_tag = typename System::variables_tag;

  template <typename LocalSystem,
            bool = LocalSystem::has_primitive_and_conservative_vars>
  struct simple_tags_impl {
    using type = tmpl::list<variables_tag>;
  };

  template <typename LocalSystem>
  struct simple_tags_impl<LocalSystem, true> {
    using type =
        tmpl::list<variables_tag, typename System::primitive_variables_tag,
                   EquationOfStateTag>;
  };

 public:
  using simple_tags = typename simple_tags_impl<System>::type;

  using compute_tags = db::AddComputeTags<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points();
    typename variables_tag::type vars(num_grid_points);

    if constexpr (System::has_primitive_and_conservative_vars) {
      using PrimitiveVars = typename System::primitive_variables_tag::type;

      PrimitiveVars primitive_vars{
          db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points()};
      auto equation_of_state =
          db::get<::Tags::AnalyticSolutionOrData>(box).equation_of_state();
      Initialization::mutate_assign<simple_tags>(
          make_not_null(&box), std::move(vars), std::move(primitive_vars),
          std::move(equation_of_state));
    } else {
      Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(vars));
    }
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Initialization
