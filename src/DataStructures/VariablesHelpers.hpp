// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for use with Variables class

#pragma once

#include <boost/range/combine.hpp>
#include <boost/tuple/tuple.hpp>
#include <cstddef>
#include <ostream>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

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

/*!
 * \ingroup DataStructuresGroup
 * \brief Slices the data within `vars` to a codimension 1 slice. The
 * slice has a constant logical coordinate in direction `sliced_dim`,
 * slicing the volume at `fixed_index` in that dimension.  For
 * example, to get the lower boundary of `sliced_dim`, pass `0` for
 * `fixed_index`; to get the upper boundary, pass
 * `extents[sliced_dim] - 1`.
 *
 * \see add_slice_to_data
 *
 * \return Variables class sliced to a hypersurface.
 */
template <std::size_t VolumeDim, typename TagsList>
Variables<TagsList> data_on_slice(const Variables<TagsList>& vars,
                                  const Index<VolumeDim>& element_extents,
                                  const size_t sliced_dim,
                                  const size_t fixed_index) noexcept {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  const size_t volume_grid_points = vars.number_of_grid_points();
  constexpr const size_t number_of_independent_components =
      Variables<TagsList>::number_of_independent_components;
  Variables<TagsList> interface_vars(interface_grid_points);
  const double* vars_data = vars.data();
  double* interface_vars_data = interface_vars.data();
  for (SliceIterator si(element_extents, sliced_dim, fixed_index); si; ++si) {
    for (size_t i = 0; i < number_of_independent_components; ++i) {
      // clang-tidy: do not use pointer arithmetic
      interface_vars_data[si.slice_offset() +                      // NOLINT
                          i * interface_grid_points] =             // NOLINT
          vars_data[si.volume_offset() + i * volume_grid_points];  // NOLINT
    }
  }
  return interface_vars;
}

/*!
 * \ingroup DataStructuresGroup
 * \brief Slices volume `Tensor`s into a `Variables`
 *
 * The slice has a constant logical coordinate in direction `sliced_dim`,
 * slicing the volume at `fixed_index` in that dimension.  For
 * example, to get the lower boundary of `sliced_dim`, pass `0` for
 * `fixed_index`; to get the upper boundary, pass
 * `extents[sliced_dim] - 1`.
 */
template <typename... TagsToSlice, size_t VolumeDim>
Variables<tmpl::list<TagsToSlice...>> data_on_slice(
    const Index<VolumeDim>& element_extents,
    const size_t sliced_dim, const size_t fixed_index,
    const typename TagsToSlice::type&... tensors) noexcept {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  Variables<tmpl::list<TagsToSlice...>> interface_vars(interface_grid_points);
  for (SliceIterator si(element_extents, sliced_dim, fixed_index); si; ++si) {
    const auto lambda = [&si](auto& interface_tensor,
                              const auto& volume_tensor) noexcept {
      for (decltype(auto) interface_and_volume_tensor_components :
           boost::combine(interface_tensor, volume_tensor)) {
        boost::get<0>(
            interface_and_volume_tensor_components)[si.slice_offset()] =
            boost::get<1>(
                interface_and_volume_tensor_components)[si.volume_offset()];
      }
    };
    expand_pack((lambda(get<TagsToSlice>(interface_vars), tensors),
                 cpp17::void_type{})...);
  }
  return interface_vars;
}

/*!
 * \ingroup DataStructuresGroup
 * \brief Adds data on a codimension 1 slice to a volume quantity. The
 * slice has a constant logical coordinate in direction `sliced_dim`,
 * slicing the volume at `fixed_index` in that dimension.  For
 * example, to add to the lower boundary of `sliced_dim`, pass `0` for
 * `fixed_index`; to add to the upper boundary, pass
 * `extents[sliced_dim] - 1`.
 *
 * \see data_on_slice
 */
template <std::size_t VolumeDim, typename TagsList>
void add_slice_to_data(const gsl::not_null<Variables<TagsList>*> volume_vars,
                       const Variables<TagsList>& vars_on_slice,
                       const Index<VolumeDim>& extents,
                       const size_t sliced_dim,
                       const size_t fixed_index) noexcept {
  constexpr const size_t number_of_independent_components =
      Variables<TagsList>::number_of_independent_components;
  const size_t volume_grid_points = extents.product();
  const size_t slice_grid_points = extents.slice_away(sliced_dim).product();
  ASSERT(volume_vars->number_of_grid_points() == volume_grid_points,
         "volume_vars has wrong number of grid points.  Expected "
         << volume_grid_points << ", got "
         << volume_vars->number_of_grid_points());
  ASSERT(vars_on_slice.number_of_grid_points() == slice_grid_points,
         "vars_on_slice has wrong number of grid points.  Expected "
         << slice_grid_points << ", got "
         << vars_on_slice.number_of_grid_points());
  double* const volume_data = volume_vars->data();
  const double* const slice_data = vars_on_slice.data();
  for (SliceIterator si(extents, sliced_dim, fixed_index); si; ++si) {
    for (size_t i = 0; i < number_of_independent_components; ++i) {
      // clang-tidy: do not use pointer arithmetic
      volume_data[si.volume_offset() + i * volume_grid_points] +=  // NOLINT
          slice_data[si.slice_offset() + i * slice_grid_points];  // NOLINT
    }
  }
}

namespace OrientVariablesOnSlice_detail {

inline std::vector<size_t> oriented_offset(
    const Index<0>& /*slice_extents*/, const size_t /*sliced_dim*/,
    const OrientationMap<1>& /*orientation_of_neighbor*/) noexcept {
  // There is only one point on a slice of a 1D mesh
  return {0};
}

std::vector<size_t> oriented_offset(
    const Index<1>& slice_extents, size_t sliced_dim,
    const OrientationMap<2>& orientation_of_neighbor) noexcept;

std::vector<size_t> oriented_offset(
    const Index<2>& slice_extents, size_t sliced_dim,
    const OrientationMap<3>& orientation_of_neighbor) noexcept;
}  // namespace OrientVariablesOnSlice_detail

/// \ingroup DataStructuresGroup
/// Orients variables on a slice to the data-storage order of a neighbor with
/// the given orientation.
template <size_t VolumeDim, typename TagsList>
Variables<TagsList> orient_variables_on_slice(
    const Variables<TagsList>& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) noexcept {
  const size_t number_of_grid_points = slice_extents.product();

  Variables<TagsList> oriented_variables(number_of_grid_points);
  const auto oriented_offset = OrientVariablesOnSlice_detail::oriented_offset(
      slice_extents, sliced_dim, orientation_of_neighbor);

  tmpl::for_each<TagsList>([&oriented_variables, &variables_on_slice,
                            &oriented_offset,
                            &number_of_grid_points](auto tag) {
    using Tag = tmpl::type_from<decltype(tag)>;
    auto& oriented_tensor = get<Tag>(oriented_variables);
    const auto& tensor_on_slice = get<Tag>(variables_on_slice);
    for (decltype(auto) oriented_and_slice_tensor_components :
         boost::combine(oriented_tensor, tensor_on_slice)) {
      DataVector& oriented_tensor_component =
          boost::get<0>(oriented_and_slice_tensor_components);
      const DataVector& tensor_component_on_slice =
          boost::get<1>(oriented_and_slice_tensor_components);
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        oriented_tensor_component[oriented_offset[s]] =
            tensor_component_on_slice[s];
      }
    }
  });

  return oriented_variables;
}
