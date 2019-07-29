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
#include "ErrorHandling/Error.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // Needs to be included somewhere and here seems most natural.
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "Utilities/Gsl.hpp"
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
struct ConservativeSystem {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    static_assert(system::is_in_flux_conservative_form,
                  "System is not in flux conservative form");
    static constexpr size_t dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;
    using fluxes_tag = db::add_tag_prefix<::Tags::Flux, variables_tag,
                                          tmpl::size_t<dim>, Frame::Inertial>;
    using sources_tag = db::add_tag_prefix<::Tags::Source, variables_tag>;
    using simple_tags =
        db::AddSimpleTags<variables_tag, fluxes_tag, sources_tag>;
    using compute_tags = db::AddComputeTags<>;

    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points();
    typename variables_tag::type vars(num_grid_points);
    typename fluxes_tag::type fluxes(num_grid_points);
    typename sources_tag::type sources(num_grid_points);

    return std::make_tuple(
        impl<Metavariables, simple_tags, compute_tags, system>(
            std::integral_constant<
                bool, system::has_primitive_and_conservative_vars>{},
            std::move(box), std::move(vars), std::move(fluxes),
            std::move(sources)));
  }

 private:
  template <typename Metavariables, typename AddSimpleTagsList,
            typename AddComputeTagsList, typename System, typename DbTagsList,
            typename... Ts>
  static auto impl(std::true_type /*has_primitive_tags*/,
                   db::DataBox<DbTagsList>&& box, Ts&&... ts) noexcept {
    static constexpr size_t dim = System::volume_dim;
    using PrimitiveVars = typename System::primitive_variables_tag::type;

    PrimitiveVars primitive_vars{
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points()};
    auto equation_of_state =
        db::get<::Tags::AnalyticSolutionOrData>(box).equation_of_state();

    return merge_into_databox<
        ConservativeSystem,
        tmpl::push_back<AddSimpleTagsList,
                        typename System::primitive_variables_tag,
                        typename Metavariables::equation_of_state_tag>,
        AddComputeTagsList>(std::move(box), std::forward<Ts>(ts)...,
                            std::move(primitive_vars),
                            std::move(equation_of_state));
  }

  template <typename Metavariables, typename AddSimpleTagsList,
            typename AddComputeTagsList, typename System, typename DbTagsList,
            typename... Ts>
  static auto impl(std::false_type /*has_primitive_tags*/,
                   db::DataBox<DbTagsList>&& box, Ts&&... ts) noexcept {
    return merge_into_databox<ConservativeSystem, AddSimpleTagsList,
                              AddComputeTagsList>(std::move(box),
                                                  std::forward<Ts>(ts)...);
  }
};
}  // namespace Actions
}  // namespace Initialization
