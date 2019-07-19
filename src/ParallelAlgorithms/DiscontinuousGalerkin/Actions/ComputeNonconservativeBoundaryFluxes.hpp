// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace dg {
namespace Actions {

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Compute \f$\hat{n} \cdot F\f$ on the boundaries for a
/// non-conservative system.
///
/// Uses:
/// - System:
///   - `variables_tag`
/// - DataBox:
///   - `DirectionsTag`,
///   - Interface items as required by
///   `Metavariables::system::normal_dot_fluxes`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::Interface<
///   DirectionsTag, db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>>`
template <typename DirectionsTag>
struct ComputeNonconservativeBoundaryFluxes {
 private:
  template <typename Metavariables, typename... NormalDotFluxTags,
            typename... Args>
  static void apply_flux(
      gsl::not_null<Variables<tmpl::list<NormalDotFluxTags...>>*> boundary_flux,
      const Args&... boundary_variables) noexcept {
    Metavariables::system::normal_dot_fluxes::apply(
        make_not_null(&get<NormalDotFluxTags>(*boundary_flux))...,
        boundary_variables...);
  }

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using variables_tag = typename system::variables_tag;

    using interface_normal_dot_fluxes_tag =
        Tags::Interface<DirectionsTag,
                        db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>>;

    db::mutate_apply<
        tmpl::list<interface_normal_dot_fluxes_tag>,
        tmpl::push_front<tmpl::transform<typename Metavariables::system::
                                             normal_dot_fluxes::argument_tags,
                                         tmpl::bind<Tags::Interface,
                                                    DirectionsTag, tmpl::_1>>,
                         DirectionsTag>>(
        [](const gsl::not_null<db::item_type<interface_normal_dot_fluxes_tag>*>
               boundary_fluxes,
           const db::item_type<DirectionsTag>& internal_directions,
           const auto&... tensors) noexcept {
          for (const auto& direction : internal_directions) {
            // Prepending the type name works around an issue with gcc-6
            ComputeNonconservativeBoundaryFluxes::apply_flux<Metavariables>(
                make_not_null(&boundary_fluxes->at(direction)),
                tensors.at(direction)...);
          }
          return boundary_fluxes;
        },
        make_not_null(&box));

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace dg
