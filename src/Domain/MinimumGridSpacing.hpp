// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Index;
template <size_t Dim>
class Mesh;
/// \endcond

/// \ingroup ComputationalDomainGroup
/// Finds the minimum coordinate distance between grid points.
template <size_t Dim, typename Frame>
double minimum_grid_spacing(
    const Index<Dim>& extents,
    const tnsr::I<DataVector, Dim, Frame>& coords) noexcept;

namespace Tags {
/// \ingroup ComputationalDomainGroup
/// \ingroup DataBoxTagsGroup
/// The minimum coordinate distance between grid points.
template <size_t Dim, typename Frame>
struct MinimumGridSpacing : db::ComputeTag {
  static std::string name() noexcept { return "MinimumGridSpacing"; }
  static auto function(
      const ::Mesh<Dim>& mesh,
      const tnsr::I<DataVector, Dim, Frame>& coordinates) noexcept {
    return minimum_grid_spacing(mesh.extents(), coordinates);
  }
  using argument_tags = tmpl::list<Mesh<Dim>, Coordinates<Dim, Frame>>;
};
}  // namespace Tags
