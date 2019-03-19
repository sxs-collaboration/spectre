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
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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

// @{
/*!
 * \ingroup DataStructuresGroup
 * \brief Slices the data within `volume_tensor` to a codimension 1 slice. The
 * slice has a constant logical coordinate in direction `sliced_dim`,
 * slicing the volume at `fixed_index` in that dimension.  For
 * example, to get the lower boundary of `sliced_dim`, pass `0` for
 * `fixed_index`; to get the upper boundary, pass
 * `extents[sliced_dim] - 1`.
 *
 * \see add_slice_to_data
 *
 * returns Tensor class sliced to a hypersurface.
 */
template <std::size_t VolumeDim, typename VectorType, typename... Structure>
void data_on_slice(
    const gsl::not_null<Tensor<VectorType, Structure...>*> interface_tensor,
    const Tensor<VectorType, Structure...>& volume_tensor,
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index) noexcept {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  if (interface_tensor->begin()->size() != interface_grid_points) {
    *interface_tensor = Tensor<VectorType, Structure...>(interface_grid_points);
  }

  for (SliceIterator si(element_extents, sliced_dim, fixed_index); si; ++si) {
    for (decltype(auto) interface_and_volume_tensor_components :
         boost::combine(*interface_tensor, volume_tensor)) {
      boost::get<0>(interface_and_volume_tensor_components)[si.slice_offset()] =
          boost::get<1>(
              interface_and_volume_tensor_components)[si.volume_offset()];
    }
  }
}

template <std::size_t VolumeDim, typename VectorType, typename... Structure>
Tensor<VectorType, Structure...> data_on_slice(
    const Tensor<VectorType, Structure...>& volume_tensor,
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index) noexcept {
  Tensor<VectorType, Structure...> interface_tensor(
      element_extents.slice_away(sliced_dim).product());
  data_on_slice(make_not_null(&interface_tensor), volume_tensor,
                element_extents, sliced_dim, fixed_index);
  return interface_tensor;
}
// @}

// @{
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
 * returns Variables class sliced to a hypersurface.
 */
template <std::size_t VolumeDim, typename TagsList>
void data_on_slice(const gsl::not_null<Variables<TagsList>*> interface_vars,
                   const Variables<TagsList>& vars,
                   const Index<VolumeDim>& element_extents,
                   const size_t sliced_dim, const size_t fixed_index) noexcept {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  const size_t volume_grid_points = vars.number_of_grid_points();
  constexpr const size_t number_of_independent_components =
      Variables<TagsList>::number_of_independent_components;

  if (interface_vars->number_of_grid_points() != interface_grid_points) {
    *interface_vars = Variables<TagsList>(interface_grid_points);
  }
  using value_type = typename Variables<TagsList>::value_type;
  const value_type* vars_data = vars.data();
  value_type* interface_vars_data = interface_vars->data();
  for (SliceIterator si(element_extents, sliced_dim, fixed_index); si; ++si) {
    for (size_t i = 0; i < number_of_independent_components; ++i) {
      // clang-tidy: do not use pointer arithmetic
      interface_vars_data[si.slice_offset() +                      // NOLINT
                          i * interface_grid_points] =             // NOLINT
          vars_data[si.volume_offset() + i * volume_grid_points];  // NOLINT
    }
  }
}

template <std::size_t VolumeDim, typename TagsList>
Variables<TagsList> data_on_slice(const Variables<TagsList>& vars,
                                  const Index<VolumeDim>& element_extents,
                                  const size_t sliced_dim,
                                  const size_t fixed_index) noexcept {
  Variables<TagsList> interface_vars(
      element_extents.slice_away(sliced_dim).product());
  data_on_slice(make_not_null(&interface_vars), vars, element_extents,
                sliced_dim, fixed_index);
  return interface_vars;
}
// @}

// @{
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
void data_on_slice(
    const gsl::not_null<Variables<tmpl::list<TagsToSlice...>>*> interface_vars,
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index,
    const typename TagsToSlice::type&... tensors) noexcept {
  const size_t interface_grid_points =
      element_extents.slice_away(sliced_dim).product();
  if (interface_vars->number_of_grid_points() != interface_grid_points) {
    *interface_vars =
        Variables<tmpl::list<TagsToSlice...>>(interface_grid_points);
  }
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
    expand_pack((lambda(get<TagsToSlice>(*interface_vars), tensors),
                 cpp17::void_type{})...);
  }
}

template <typename... TagsToSlice, size_t VolumeDim>
Variables<tmpl::list<TagsToSlice...>> data_on_slice(
    const Index<VolumeDim>& element_extents, const size_t sliced_dim,
    const size_t fixed_index,
    const typename TagsToSlice::type&... tensors) noexcept {
  Variables<tmpl::list<TagsToSlice...>> interface_vars(
      element_extents.slice_away(sliced_dim).product());
  data_on_slice<TagsToSlice...>(make_not_null(&interface_vars), element_extents,
                                sliced_dim, fixed_index, tensors...);
  return interface_vars;
}
// @}

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
                       const Index<VolumeDim>& extents, const size_t sliced_dim,
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
  using value_type = typename Variables<TagsList>::value_type;
  value_type* const volume_data = volume_vars->data();
  const value_type* const slice_data = vars_on_slice.data();
  for (SliceIterator si(extents, sliced_dim, fixed_index); si; ++si) {
    for (size_t i = 0; i < number_of_independent_components; ++i) {
      // clang-tidy: do not use pointer arithmetic
      volume_data[si.volume_offset() + i * volume_grid_points] +=  // NOLINT
          slice_data[si.slice_offset() + i * slice_grid_points];   // NOLINT
    }
  }
}

namespace OrientVariables_detail {

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

}  // namespace OrientVariables_detail

/// \ingroup DataStructuresGroup
/// Orient variables to the data-storage order of a neighbor element with
/// the given orientation.
/// @{
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
  const auto oriented_offset =
      OrientVariables_detail::oriented_offset(extents, orientation_of_neighbor);
  OrientVariables_detail::orient_each_component(
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
  const auto oriented_offset = OrientVariables_detail::oriented_offset_on_slice(
      slice_extents, sliced_dim, orientation_of_neighbor);
  OrientVariables_detail::orient_each_component(
      make_not_null(&oriented_variables), variables_on_slice, oriented_offset);

  return oriented_variables;
}
/// }@
