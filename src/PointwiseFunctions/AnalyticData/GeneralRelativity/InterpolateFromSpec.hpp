// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <Exporter.hpp>  // The SpEC Exporter
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace gr::AnalyticData {

/*!
 * \brief Interpolate numerical SpEC initial data to arbitrary points
 *
 * \tparam Tags List of tags to load. The tags must correspond exactly to the
 * list of variables that the `spec_exporter` was configured with. This function
 * does not support interpolating only a subset of the variables. This is a
 * limitation of the `spec::Exporter`.
 * \tparam DataType `double` or `DataVector`.
 * \tparam CoordFrame The frame of the coordinates `x`. These coordinates are
 * always assumed to be in SpEC's "grid" frame.
 *
 * \param spec_exporter Has the numerical data already loaded. It must be
 * configured with the same number of variables that were passed as `Tags`, and
 * in the same order.
 * \param x Interpolate to these coordinates. They are assumed to be in SpEC's
 * "grid" frame.
 * \param which_interpolator Index of the interpolator. See `spec::Exporter`
 * documentation for details.
 */
template <typename Tags, typename DataType, typename CoordFrame>
tuples::tagged_tuple_from_typelist<Tags> interpolate_from_spec(
    const gsl::not_null<spec::Exporter*> spec_exporter,
    const tnsr::I<DataType, 3, CoordFrame>& x,
    const size_t which_interpolator) {
  // The `spec::Exporter` doesn't currently expose its `vars_to_interpolate`.
  // Once it does, we can assert the number of variables here.
  // const auto& dataset_names = spec_exporter.vars_to_interpolate();
  // ASSERT(tmpl::size<Tags>::value == dataset_names.size(),
  //        "Mismatch between number of tags and dataset names. The SpEC
  //        exporter " "was configured with " +
  //            std::to_string(dataset_names.size()) +
  //            " variables to interpolate, but requested " +
  //            std::to_string(tmpl::size<Tags>::value) + " tags.");
  // Transform coordinates into SpEC's expected format. SpEC expects grid
  // coordinates. We assume that the grid coordinates coincide with inertial
  // coordinates for the initial data.
  const size_t num_points = get_size(get<0>(x));
  std::vector<std::vector<double>> spec_grid_coords(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    spec_grid_coords[i] = std::vector<double>{get_element(get<0>(x), i),
                                              get_element(get<1>(x), i),
                                              get_element(get<2>(x), i)};
  }
  // Allocate memory for the interpolation and point into it
  tuples::tagged_tuple_from_typelist<Tags> interpolation_buffer{};
  std::vector<std::vector<double*>> buffer_pointers(tmpl::size<Tags>::value);
  size_t var_i = 0;
  tmpl::for_each<Tags>([&interpolation_buffer, &buffer_pointers, &num_points,
                        &var_i](const auto tag_v) {
    using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
    auto& tensor = get<tag>(interpolation_buffer);
    destructive_resize_components(make_not_null(&tensor), num_points);
    const size_t num_components = tensor.size();
    buffer_pointers[var_i].resize(num_components);
    // The SpEC exporter supports tensors up to symmetric rank 2, which are
    // ordered xx, yx, zx, yy, zy, zz. Because this is also the order in
    // which we store components in the Tensor class, we don't have to do
    // anything special here.
    // WARNING: If the Tensor storage order changes for some reason, this
    // code needs to be updated.
    for (size_t component_i = 0; component_i < num_components; ++component_i) {
#ifdef SPECTRE_DEBUG
      const auto component_name =
          tensor.component_name(tensor.get_tensor_index(component_i));
      if constexpr (tensor.rank() == 1) {
        const std::array<std::string, 3> spec_component_order{{"x", "y", "z"}};
        ASSERT(component_name == gsl::at(spec_component_order, component_i),
               "Unexpected Tensor component order. Expected "
                   << spec_component_order);
      } else if constexpr (tensor.rank() == 2 and tensor.size() == 6) {
        const std::array<std::string, 6> spec_component_order{
            {"xx", "yx", "zx", "yy", "zy", "zz"}};
        ASSERT(component_name == gsl::at(spec_component_order, component_i),
               "Unexpected Tensor component order. Expected "
                   << spec_component_order);
      } else if constexpr (tensor.rank() > 0) {
        ASSERT(
            false,
            "Unsupported Tensor type for SpEC import. Only scalars, "
            "vectors, and symmetric rank-2 tensors are currently supported.");
      }
#endif
        auto& component = tensor[component_i];
      auto& component_pointer = buffer_pointers[var_i][component_i];
      if constexpr (std::is_same_v<DataType, double>) {
        component_pointer = &component;
      } else {
        component_pointer = component.data();
      }
    }
    ++var_i;
  });
  // Interpolate!
  spec_exporter->interpolate(buffer_pointers, spec_grid_coords,
                             which_interpolator);
  return interpolation_buffer;
}

}  // namespace gr::AnalyticData
