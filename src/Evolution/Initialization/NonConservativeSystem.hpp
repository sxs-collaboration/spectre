// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Initialization {
/// \brief Allocate variables needed for evolution of non-conservative systems
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
/// \note This only allocates storage for all variables, which are initialized
/// to `signaling_NaN()`. `Flux` and `Source` will be computed during the
/// evolution, but the evolved variables themselves need to be initialized
/// to `AnalyticSolution` or `AnalyticData`.
template <typename System>
struct NonConservativeSystem {
  static_assert(not System::is_in_flux_conservative_form,
                "System is in flux conservative form!");
  static constexpr size_t dim = System::volume_dim;
  using frame = Frame::Inertial;

  using variables_tag = typename System::variables_tag;
  using fluxes_tag =
      db::add_tag_prefix<Tags::Flux, variables_tag, tmpl::size_t<dim>, frame>;
  using sources_tag = db::add_tag_prefix<Tags::Source, variables_tag>;
  using simple_tags = db::AddSimpleTags<fluxes_tag, sources_tag>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box) noexcept {
    const size_t num_grid_points =
        db::get<::Tags::Mesh<dim>>(box).number_of_grid_points();

    // typename variables_tag::type vars{num_grid_points};
    typename fluxes_tag::type fluxes(num_grid_points);
    typename sources_tag::type sources(num_grid_points);

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), /* std::move(vars),*/ std::move(fluxes),
        std::move(sources));
  }
};
}  // namespace Initialization
