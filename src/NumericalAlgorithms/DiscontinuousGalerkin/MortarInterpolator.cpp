// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarInterpolator.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
tnsr::I<DataVector, Dim - 1, Frame::ElementLogical>
compute_neighbor_target_points(
    const tnsr::I<DataVector, Dim, Frame::Grid>& host_grid_coordinates,
    const ElementMap<Dim, Frame::Grid>& element_map,
    const size_t boundary_dimension) {
  tnsr::I<DataVector, Dim - 1, Frame::ElementLogical> result{
      get<0>(host_grid_coordinates).size()};
  tnsr::I<double, Dim, Frame::Grid> x{};
  for (size_t s = 0; s < get<0>(host_grid_coordinates).size(); ++s) {
    for (size_t d = 0; d < Dim; ++d) {
      x.get(d) = host_grid_coordinates.get(d)[s];
    }
    const auto xi = element_map.inverse(x);
    for (size_t d = 0; d < boundary_dimension; ++d) {
      result.get(d)[s] = xi.get(d);
    }
    for (size_t d = boundary_dimension; d < Dim - 1; ++d) {
      result.get(d)[s] = xi.get(d + 1);
    }
  }
  return result;
}
}  // namespace

namespace dg {
template <size_t Dim>
MortarInterpolator<Dim>::MortarInterpolator(
    const ElementId<Dim>& host_id, const DirectionalId<Dim>& neighbor_id,
    const Domain<Dim>& domain, const Mesh<Dim - 1>& host_mortar_mesh,
    const Mesh<Dim - 1>& neighbor_mortar_mesh)
    : host_id_(host_id), neighbor_id_(neighbor_id) {
  reset_if_necessary(domain, host_mortar_mesh, neighbor_mortar_mesh);
}

template <size_t Dim>
void MortarInterpolator<Dim>::interpolate_to_host(
    const gsl::not_null<DataVector*> result,
    const DataVector& neighbor_data) const {
  const size_t number_of_components =
      neighbor_data.size() / neighbor_mortar_mesh_.number_of_grid_points();
  ASSERT(result->size() ==
             host_mortar_mesh_.number_of_grid_points() * number_of_components,
         "Expected result to have size "
             << host_mortar_mesh_.number_of_grid_points() * number_of_components
             << ", not " << result->size());
  gsl::span<double> result_span{result->data(), result->size()};
  neighbor_to_host_interpolant_.interpolate(
      make_not_null(&result_span),
      gsl::make_span(neighbor_data.data(), neighbor_data.size()));
}

template <size_t Dim>
DataVector MortarInterpolator<Dim>::interpolate_to_host(
    const DataVector& neighbor_data) const {
  const size_t number_of_components =
      neighbor_data.size() / neighbor_mortar_mesh_.number_of_grid_points();
  DataVector result{host_mortar_mesh_.number_of_grid_points() *
                    number_of_components};
  interpolate_to_host(make_not_null(&result), neighbor_data);
  return result;
}

template <size_t Dim>
void MortarInterpolator<Dim>::interpolate_to_neighbor(
    const gsl::not_null<DataVector*> result,
    const DataVector& host_data) const {
  const size_t number_of_components =
      host_data.size() / host_mortar_mesh_.number_of_grid_points();
  ASSERT(
      result->size() ==
          interpolated_neighbor_data_offsets_.size() * number_of_components,
      "Expected result to have size "
          << interpolated_neighbor_data_offsets_.size() * number_of_components
          << ", not " << result->size());
  gsl::span<double> result_span{result->data(), result->size()};
  host_to_neighbor_interpolant_.interpolate(
      make_not_null(&result_span),
      gsl::make_span(host_data.data(), host_data.size()));
}

template <size_t Dim>
DataVector MortarInterpolator<Dim>::interpolate_to_neighbor(
    const DataVector& host_data) const {
  const size_t number_of_components =
      host_data.size() / host_mortar_mesh_.number_of_grid_points();
  DataVector result{interpolated_neighbor_data_offsets_.size() *
                    number_of_components};
  interpolate_to_neighbor(make_not_null(&result), host_data);
  return result;
}

template <size_t Dim>
void MortarInterpolator<Dim>::reset_if_necessary(
    const Domain<Dim>& domain, const Mesh<Dim - 1>& host_mortar_mesh,
    const Mesh<Dim - 1>& neighbor_mortar_mesh) {
  if (host_mortar_mesh == host_mortar_mesh_ and
      neighbor_mortar_mesh == neighbor_mortar_mesh_) {
    return;
  }
  host_mortar_mesh_ = host_mortar_mesh;
  neighbor_mortar_mesh_ = neighbor_mortar_mesh;
  const Block<Dim>& host_block = domain.blocks()[host_id_.block_id()];
  const Block<Dim>& neighbor_block =
      domain.blocks()[neighbor_id_.id().block_id()];
  const ElementMap<Dim, Frame::Grid> host_element_logical_to_grid_map{
      host_id_, host_block};
  const ElementMap<Dim, Frame::Grid> neighbor_element_logical_to_grid_map{
      neighbor_id_.id(), neighbor_block};
  const auto& orientation = host_block.neighbors()
                                .at(neighbor_id_.direction())
                                .orientations()
                                .at(neighbor_block.id());

  // Setup interpolator from neighbor (e.g. S2 shell) to host (e.g. cube face)
  // The target points are the points of the host mortar mesh
  // Do not need to worry about which block as there is a single neighbor
  // and it must contain the target points
  {
    const auto host_element_logical_coords = interface_logical_coordinates(
        host_mortar_mesh_, neighbor_id_.direction());
    const auto host_grid_coords =
        host_element_logical_to_grid_map(host_element_logical_coords);
    const auto target_points_in_neighbor = compute_neighbor_target_points(
        host_grid_coords, neighbor_element_logical_to_grid_map,
        orientation(neighbor_id_.direction().dimension()));
    neighbor_to_host_interpolant_ = intrp::Irregular<Dim - 1>{
        neighbor_mortar_mesh_, target_points_in_neighbor};
  }

  // Setup interpolator from host (e.g. cube face) to neighbor (e.g. S2 shell)
  // The target points are those of the neighbor mortar mesh that the host
  // contains.
  const auto neighbor_element_logical_coords = interface_logical_coordinates(
      neighbor_mortar_mesh_, orientation(neighbor_id_.direction().opposite()));
  const auto neighbor_grid_coords =
      neighbor_element_logical_to_grid_map(neighbor_element_logical_coords);
  const size_t npts_neighbor = neighbor_mortar_mesh_.number_of_grid_points();
  interpolated_neighbor_data_offsets_.clear();
  interpolated_neighbor_data_offsets_.reserve(npts_neighbor);
  std::vector<tnsr::I<double, Dim, Frame::ElementLogical>>
      host_element_logical_coords{};
  host_element_logical_coords.reserve(npts_neighbor);
  tnsr::I<double, Dim, Frame::Grid> x{};
  for (size_t s = 0; s < npts_neighbor; ++s) {
    for (size_t d = 0; d < Dim; ++d) {
      x.get(d) = neighbor_grid_coords.get(d)[s];
    }
    const auto block_xi = block_logical_coordinates_single_point(x, host_block);
    if (block_xi.has_value()) {
      const auto xi = element_logical_coordinates(block_xi.value(), host_id_);
      if (xi.has_value()) {
        interpolated_neighbor_data_offsets_.emplace_back(s);
        host_element_logical_coords.emplace_back(xi.value());
      }
    }
  }
  const size_t npts_neighbor_in_host =
      interpolated_neighbor_data_offsets_.size();
  tnsr::I<DataVector, Dim - 1, Frame::ElementLogical> target_points_in_host{
      npts_neighbor_in_host};
  const size_t boundary_dimension = neighbor_id_.direction().dimension();
  for (size_t s = 0; s < npts_neighbor_in_host; ++s) {
    for (size_t d = 0; d < boundary_dimension; ++d) {
      target_points_in_host.get(d)[s] = host_element_logical_coords[s].get(d);
    }
    for (size_t d = boundary_dimension; d < Dim - 1; ++d) {
      target_points_in_host.get(d)[s] =
          host_element_logical_coords[s].get(d + 1);
    }
  }
  host_to_neighbor_interpolant_ =
      intrp::Irregular<Dim - 1>{host_mortar_mesh_, target_points_in_host};
}

template <size_t Dim>
void MortarInterpolator<Dim>::pup(PUP::er& p) {
  p | host_id_;
  p | neighbor_id_;
  p | host_mortar_mesh_;
  p | neighbor_mortar_mesh_;
  p | neighbor_to_host_interpolant_;
  p | host_to_neighbor_interpolant_;
  p | interpolated_neighbor_data_offsets_;
}

template <size_t Dim>
bool operator==(const MortarInterpolator<Dim>& lhs,
                const MortarInterpolator<Dim>& rhs) {
  return lhs.host_id_ == rhs.host_id_ and
         lhs.neighbor_id_ == rhs.neighbor_id_ and
         lhs.host_mortar_mesh_ == rhs.host_mortar_mesh_ and
         lhs.neighbor_mortar_mesh_ == rhs.neighbor_mortar_mesh_ and
         lhs.neighbor_to_host_interpolant_ ==
             rhs.neighbor_to_host_interpolant_ and
         lhs.host_to_neighbor_interpolant_ ==
             rhs.host_to_neighbor_interpolant_ and
         lhs.interpolated_neighbor_data_offsets_ ==
             rhs.interpolated_neighbor_data_offsets_;
}

template <size_t Dim>
bool operator!=(const MortarInterpolator<Dim>& lhs,
                const MortarInterpolator<Dim>& rhs) {
  return not(lhs == rhs);
}

template class MortarInterpolator<2>;
template class MortarInterpolator<3>;
template bool operator==(const MortarInterpolator<2>&,
                         const MortarInterpolator<2>&);
template bool operator==(const MortarInterpolator<3>&,
                         const MortarInterpolator<3>&);
template bool operator!=(const MortarInterpolator<2>&,
                         const MortarInterpolator<2>&);
template bool operator!=(const MortarInterpolator<3>&,
                         const MortarInterpolator<3>&);
}  // namespace dg
