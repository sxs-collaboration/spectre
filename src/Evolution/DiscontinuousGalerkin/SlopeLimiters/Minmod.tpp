// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <pup.h>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/MaxNumberOfNeighbors.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

namespace SlopeLimiters {
namespace Minmod_detail {

// Implements the minmod limiter for one Tensor<DataVector> at a time.
template <size_t VolumeDim>
bool limit_one_tensor(
    const gsl::not_null<DataVector*> tensor_begin,
    const gsl::not_null<DataVector*> tensor_end,
    const gsl::not_null<DataVector*> u_lin_buffer,
    const gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const SlopeLimiters::MinmodType& minmod_type, const double tvbm_constant,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const FixedHashMap<
        maximum_number_of_neighbors(VolumeDim),
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        gsl::not_null<const double*>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_tensor_begin,
    const FixedHashMap<
        maximum_number_of_neighbors(VolumeDim),
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        std::array<double, VolumeDim>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_sizes,
    const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                               gsl::span<std::pair<size_t, size_t>>>,
                     VolumeDim>& volume_and_slice_indices) noexcept {
  // True if the mesh is linear-order in every direction
  const bool mesh_is_linear = (mesh.extents() == Index<VolumeDim>(2));
  const bool minmod_type_is_linear =
      (minmod_type != SlopeLimiters::MinmodType::LambdaPiN);
  const bool using_linear_limiter_on_non_linear_mesh =
      minmod_type_is_linear and not mesh_is_linear;

  // In each direction, average the size of all different neighbors in that
  // direction. Note that only the component of neighor_size that is normal
  // to the face is needed (and, therefore, computed). Note that this average
  // does not depend on the solution on the neighboring elements, so could be
  // precomputed outside of `limit_one_tensor`. Changing the code to
  // precompute the average may or may not be a measurable optimization.
  const auto effective_neighbor_sizes =
      [&neighbor_sizes, &element ]() noexcept {
    DirectionMap<VolumeDim, double> result;
    for (const auto& dir : Direction<VolumeDim>::all_directions()) {
      const auto& externals = element.external_boundaries();
      const bool neighbors_in_this_dir =
          (externals.find(dir) == externals.end());
      if (neighbors_in_this_dir) {
        const double effective_neighbor_size =
            [&neighbor_sizes, &dir, &element ]() noexcept {
          const size_t dim = dir.dimension();
          const auto& neighbor_ids = element.neighbors().at(dir).ids();
          double size_accumulate = 0.0;
          for (const auto& id : neighbor_ids) {
            size_accumulate +=
                gsl::at(neighbor_sizes.at(std::make_pair(dir, id)), dim);
          }
          return size_accumulate / neighbor_ids.size();
        }
        ();
        result.insert(std::make_pair(dir, effective_neighbor_size));
      }
    }
    return result;
  }
  ();

  bool some_component_was_limited = false;
  for (auto iter = tensor_begin.get(); iter != tensor_end.get(); ++iter) {
    const auto iter_offset = std::distance(tensor_begin.get(), iter);

    // In each direction, average the mean of the `iter` tensor component over
    // all different neighbors in that direction. This produces one effective
    // neighbor per direction.
    const auto effective_neighbor_means =
        [&neighbor_tensor_begin, &element, &iter_offset ]() noexcept {
      DirectionMap<VolumeDim, double> result;
      for (const auto& dir : Direction<VolumeDim>::all_directions()) {
        const auto& externals = element.external_boundaries();
        const bool neighbors_in_this_dir =
            (externals.find(dir) == externals.end());
        if (neighbors_in_this_dir) {
          const double effective_neighbor_mean = [
            &neighbor_tensor_begin, &dir, &element, &iter_offset
          ]() noexcept {
            const auto& neighbor_ids = element.neighbors().at(dir).ids();
            double mean_accumulate = 0.0;
            for (const auto& id : neighbor_ids) {
              // clang-tidy: do not use pointer arithmetic
              mean_accumulate +=
                  *(neighbor_tensor_begin.at(std::make_pair(dir, id)).get() +
                    iter_offset);  // NOLINT
            }
            return mean_accumulate / neighbor_ids.size();
          }
          ();
          result.insert(std::make_pair(dir, effective_neighbor_mean));
        }
      }
      return result;
    }
    ();

    DataVector& u = *iter;
    double u_mean;
    std::array<double, VolumeDim> u_limited_slopes{};
    const bool reduce_slope = troubled_cell_indicator(
        make_not_null(&u_mean), make_not_null(&u_limited_slopes), u_lin_buffer,
        boundary_buffer, minmod_type, tvbm_constant, u, element, mesh,
        element_size, effective_neighbor_means, effective_neighbor_sizes,
        volume_and_slice_indices);

    if (reduce_slope or using_linear_limiter_on_non_linear_mesh) {
      u = u_mean;
      for (size_t d = 0; d < VolumeDim; ++d) {
        u += logical_coords.get(d) * gsl::at(u_limited_slopes, d);
      }
      some_component_was_limited = true;
    }
  }  // end for loop over tensor components

  return some_component_was_limited;
}

}  // namespace Minmod_detail

template <size_t VolumeDim, typename... Tags>
Minmod<VolumeDim, tmpl::list<Tags...>>::Minmod(
    const MinmodType minmod_type, const double tvbm_constant,
    const bool disable_for_debugging) noexcept
    : minmod_type_(minmod_type),
      tvbm_constant_(tvbm_constant),
      disable_for_debugging_(disable_for_debugging) {
  ASSERT(tvbm_constant >= 0.0, "The TVBM constant must be non-negative.");
}

template <size_t VolumeDim, typename... Tags>
void Minmod<VolumeDim, tmpl::list<Tags...>>::pup(PUP::er& p) noexcept {
  p | minmod_type_;
  p | tvbm_constant_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim, typename... Tags>
void Minmod<VolumeDim, tmpl::list<Tags...>>::package_data(
    const gsl::not_null<PackagedData*>& packaged_data,
    const db::item_type<Tags>&... tensors, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not initialize packaged_data
    return;
  }

  const auto wrap_compute_means =
      [&mesh, &packaged_data ](auto tag, const auto& tensor) noexcept {
    for (size_t i = 0; i < tensor.size(); ++i) {
      // Compute the mean using the local orientation of the tensor and mesh:
      // this avoids the work of reorienting the tensor while giving the same
      // result.
      get<::Tags::Mean<decltype(tag)>>(
          packaged_data->means)[i] = mean_value(tensor[i], mesh);
    }
    return '0';
  };
  expand_pack(wrap_compute_means(Tags{}, tensors)...);
  packaged_data->element_size =
      orientation_map.permute_from_neighbor(element_size);
}

template <size_t VolumeDim, typename... Tags>
bool Minmod<VolumeDim, tmpl::list<Tags...>>::operator()(
    const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Allocate temporary buffer to be used in `limit_one_tensor` where we
  // otherwise make 1 + 2 * VolumeDim allocations per tensor component for
  // MUSCL and LambdaPi1, and 1 + 4 * VolumeDim allocations per tensor
  // component for LambdaPiN.
  const size_t half_number_boundary_points = alg::accumulate(
      alg::iota(std::array<size_t, VolumeDim>{{}}, 0_st),
      0_st, [&mesh](const size_t state, const size_t d) noexcept {
        return state + mesh.slice_away(d).number_of_grid_points();
      });
  std::unique_ptr<double[], decltype(&free)> temp_buffer(
      static_cast<double*>(
          malloc(sizeof(double) *
                 (mesh.number_of_grid_points() + half_number_boundary_points))),
      &free);
  size_t alloc_offset = 0;
  DataVector u_lin_buffer(temp_buffer.get() + alloc_offset,
                          mesh.number_of_grid_points());
  alloc_offset += mesh.number_of_grid_points();
  std::array<DataVector, VolumeDim> boundary_buffer{};
  for (size_t d = 0; d < VolumeDim; ++d) {
    const size_t num_points = mesh.slice_away(d).number_of_grid_points();
    gsl::at(boundary_buffer, d)
        .set_data_ref(temp_buffer.get() + alloc_offset, num_points);
    alloc_offset += num_points;
  }
  // Compute the slice indices once since this is (surprisingly) expensive
  const auto volume_and_slice_buffer_and_indices =
      volume_and_slice_indices(mesh.extents());
  const auto volume_and_slice_indices =
      volume_and_slice_buffer_and_indices.second;

  bool limiter_activated = false;
  const auto wrap_limit_one_tensor = [
    this, &limiter_activated, &element, &mesh, &logical_coords, &element_size,
    &neighbor_data, &u_lin_buffer, &boundary_buffer, &volume_and_slice_indices
  ](auto tag, const auto& tensor) noexcept {
    // Because we hide the types of Tags from limit_one_tensor (we do this so
    // that its implementation isn't templated on Tags and can be moved out of
    // this header file), we cannot pass it PackagedData as currently
    // implemented. So we unpack everything from PackagedData. In the future
    // we may want a PackagedData type that erases types inherently, as this
    // would avoid the need for unpacking as done here.
    //
    // Get iterators into the local and neighbor tensors, because these are
    // independent from the structure of the tensor being limited.
    const auto tensor_begin = make_not_null(tensor->begin());
    const auto tensor_end = make_not_null(tensor->end());
    const auto neighbor_tensor_begin = [&neighbor_data]() noexcept {
      FixedHashMap<
          maximum_number_of_neighbors(VolumeDim),
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
          gsl::not_null<const double*>,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
          result;
      for (const auto& neighbor_and_data : neighbor_data) {
        result.insert(std::make_pair(
            neighbor_and_data.first,
            make_not_null(get<::Tags::Mean<decltype(tag)>>(
                              neighbor_and_data.second.means)
                              .cbegin())));
      }
      return result;
    }
    ();
    const auto neighbor_sizes = [&neighbor_data]() noexcept {
      FixedHashMap<
          maximum_number_of_neighbors(VolumeDim),
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
          std::array<double, VolumeDim>,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
          result;
      for (const auto& neighbor_and_data : neighbor_data) {
        result.insert(std::make_pair(neighbor_and_data.first,
                                     neighbor_and_data.second.element_size));
      }
      return result;
    }
    ();

    limiter_activated =
        Minmod_detail::limit_one_tensor<VolumeDim>(
            tensor_begin, tensor_end, &u_lin_buffer, &boundary_buffer,
            minmod_type_, tvbm_constant_, element, mesh, logical_coords,
            element_size, neighbor_tensor_begin, neighbor_sizes,
            volume_and_slice_indices) or
        limiter_activated;
    return '0';
  };
  expand_pack(wrap_limit_one_tensor(Tags{}, tensors)...);
  return limiter_activated;
}

template <size_t LocalDim, typename LocalTagList>
bool operator==(const Minmod<LocalDim, LocalTagList>& lhs,
                const Minmod<LocalDim, LocalTagList>& rhs) noexcept {
  return lhs.minmod_type_ == rhs.minmod_type_ and
         lhs.tvbm_constant_ == rhs.tvbm_constant_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim, typename TagList>
bool operator!=(const Minmod<VolumeDim, TagList>& lhs,
                const Minmod<VolumeDim, TagList>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace SlopeLimiters
