// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Initialization {

/// \ingroup InitializationGroup
/// \brief Initialize items related to the structure of a Domain
///
/// \see InitializeItems
///
/// \note This only initializes the Element and the Mesh on each element of
/// an array component.  This is all that is needed about the Domain in order to
/// test the mechanics of adaptive mesh refinement.
template <size_t Dim>
struct Domain {
  using const_global_cache_tags = tmpl::list<::domain::Tags::Domain<Dim>>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options =
      tmpl::list<::domain::Tags::InitialExtents<Dim>,
                 ::domain::Tags::InitialRefinementLevels<Dim>,
                 evolution::dg::Tags::Quadrature>;

  using argument_tags =
      tmpl::append<const_global_cache_tags, simple_tags_from_options,
                   tmpl::list<::Parallel::Tags::ArrayIndex>>;

  using return_tags =
      tmpl::list<::domain::Tags::Mesh<Dim>, ::domain::Tags::Element<Dim>>;

  using simple_tags = return_tags;
  using compute_tags = tmpl::list<>;

  /// Given the items fetched from a DataBox by the argument_tags, mutate
  /// the items in the DataBox corresponding to return_tags
  static void apply(
      const gsl::not_null<Mesh<Dim>*> mesh,
      const gsl::not_null<Element<Dim>*> element, const ::Domain<Dim>& domain,
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const std::vector<std::array<size_t, Dim>>& initial_refinement,
      const Spectral::Quadrature& quadrature,
      const ElementId<Dim>& element_id) {
    const auto& my_block = domain.blocks()[element_id.block_id()];
    *mesh = ::domain::Initialization::create_initial_mesh(
        initial_extents, element_id, quadrature);
    *element = ::domain::Initialization::create_initial_element(
        element_id, my_block, initial_refinement);
  }
};
}  // namespace amr::Initialization
