// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>
#include <cstddef>
#include <vector>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class OrientationMap;
template <size_t>
class Index;
template <typename TagsList>
class Variables;

// clang-tidy: redundant declarations
template <typename Tag, typename TagList>
constexpr typename Tag::type& get(Variables<TagList>& v) noexcept;  // NOLINT
template <typename Tag, typename TagList>
constexpr const typename Tag::type& get(  // NOLINT
    const Variables<TagList>& v) noexcept;
/// \endcond

namespace OrientationMapHelpers_detail {

template <typename TagsList>
void orient_each_component(
    const gsl::not_null<Variables<TagsList>*> oriented_variables,
    const Variables<TagsList>& variables,
    const std::vector<size_t>& oriented_offset) noexcept {
  using VectorType = typename Variables<TagsList>::vector_type;
  tmpl::for_each<TagsList>(
      [&oriented_variables, &variables, &oriented_offset](auto tag) {
        using Tag = tmpl::type_from<decltype(tag)>;
        auto& oriented_tensor = get<Tag>(*oriented_variables);
        const auto& tensor = get<Tag>(variables);
        for (decltype(auto) oriented_and_tensor_components :
             boost::combine(oriented_tensor, tensor)) {
          VectorType& oriented_tensor_component =
              boost::get<0>(oriented_and_tensor_components);
          const VectorType& tensor_component =
              boost::get<1>(oriented_and_tensor_components);
          for (size_t s = 0; s < tensor_component.size(); ++s) {
            oriented_tensor_component[oriented_offset[s]] = tensor_component[s];
          }
        }
      });
}

template <size_t VolumeDim>
std::vector<size_t> oriented_offset(
    const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) noexcept;

inline std::vector<size_t> oriented_offset_on_slice(
    const Index<0>& /*slice_extents*/, const size_t /*sliced_dim*/,
    const OrientationMap<1>& /*orientation_of_neighbor*/) noexcept {
  // There is only one point on a slice of a 1D mesh
  return {0};
}

std::vector<size_t> oriented_offset_on_slice(
    const Index<1>& slice_extents, size_t sliced_dim,
    const OrientationMap<2>& orientation_of_neighbor) noexcept;

std::vector<size_t> oriented_offset_on_slice(
    const Index<2>& slice_extents, size_t sliced_dim,
    const OrientationMap<3>& orientation_of_neighbor) noexcept;

}  // namespace OrientationMapHelpers_detail

// @{
/// \ingroup ComputationalDomainGroup
/// Orient variables to the data-storage order of a neighbor element with
/// the given orientation.
template <size_t VolumeDim, typename TagsList>
Variables<TagsList> orient_variables(
    const Variables<TagsList>& variables, const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) noexcept {
  // Skip work (aside from a copy) if neighbor is aligned
  if (orientation_of_neighbor.is_aligned()) {
    return variables;
  }

  const size_t number_of_grid_points = extents.product();
  ASSERT(variables.number_of_grid_points() == number_of_grid_points,
         "Inconsistent `variables` and `extents`:\n"
         "  variables.number_of_grid_points() = "
             << variables.number_of_grid_points()
             << "\n"
                "  extents.product() = "
             << extents.product());
  Variables<TagsList> oriented_variables(number_of_grid_points);
  const auto oriented_offset = OrientationMapHelpers_detail::oriented_offset(
      extents, orientation_of_neighbor);
  OrientationMapHelpers_detail::orient_each_component(
      make_not_null(&oriented_variables), variables, oriented_offset);

  return oriented_variables;
}

template <size_t VolumeDim, typename TagsList>
Variables<TagsList> orient_variables_on_slice(
    const Variables<TagsList>& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) noexcept {
  // Skip work (aside from a copy) if neighbor slice is aligned
  if (orientation_of_neighbor.is_aligned()) {
    return variables_on_slice;
  }

  const size_t number_of_grid_points = slice_extents.product();
  ASSERT(variables_on_slice.number_of_grid_points() == number_of_grid_points,
         "Inconsistent `variables_on_slice` and `slice_extents`:\n"
         "  variables_on_slice.number_of_grid_points() = "
             << variables_on_slice.number_of_grid_points()
             << "\n"
                "  slice_extents.product() = "
             << slice_extents.product());
  Variables<TagsList> oriented_variables(number_of_grid_points);
  const auto oriented_offset =
      OrientationMapHelpers_detail::oriented_offset_on_slice(
          slice_extents, sliced_dim, orientation_of_neighbor);
  OrientationMapHelpers_detail::orient_each_component(
      make_not_null(&oriented_variables), variables_on_slice, oriented_offset);

  return oriented_variables;
}
// }@
