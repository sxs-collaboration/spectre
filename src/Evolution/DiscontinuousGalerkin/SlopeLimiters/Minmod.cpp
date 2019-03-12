// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"

#include <iterator>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodTci.hpp"
#include "Utilities/GenerateInstantiations.hpp"

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

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template bool limit_one_tensor<DIM(data)>(                              \
      const gsl::not_null<DataVector*>, const gsl::not_null<DataVector*>, \
      const gsl::not_null<DataVector*>,                                   \
      const gsl::not_null<std::array<DataVector, DIM(data)>*>,            \
      const SlopeLimiters::MinmodType&, const double,                     \
      const Element<DIM(data)>&, const Mesh<DIM(data)>&,                  \
      const tnsr::I<DataVector, DIM(data), Frame::Logical>&,              \
      const std::array<double, DIM(data)>&,                               \
      const FixedHashMap<                                                 \
          maximum_number_of_neighbors(DIM(data)),                         \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,          \
          gsl::not_null<const double*>,                                   \
          boost::hash<                                                    \
              std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>&,   \
      const FixedHashMap<                                                 \
          maximum_number_of_neighbors(DIM(data)),                         \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>,          \
          std::array<double, DIM(data)>,                                  \
          boost::hash<                                                    \
              std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>&,   \
      const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,    \
                                 gsl::span<std::pair<size_t, size_t>>>,   \
                       DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
}  // namespace Minmod_detail
}  // namespace SlopeLimiters
