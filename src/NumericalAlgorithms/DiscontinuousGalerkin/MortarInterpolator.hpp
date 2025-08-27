// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

/// \cond
template <size_t Dim>
class Domain;
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace dg {
/// \brief Interpolator to be used for interpolating boundary data from (to) a
/// Neighbor onto a mortar between nonconforming Blocks.
///
/// \details This interpolator is designed to be used when multiple
/// nonconforming elements are neighbors with a single element (e.g deformed
/// cubes of a cubed sphere next to a single S2 spherical shell). Each of the
/// multiple elements (the hosts) should create this class to both interpolate
/// neighbor boundary data to the mortar of the host as well as interpolate the
/// host boundary data onto the subset of points of the mortar of the neighbor.
///
/// \note For nonconforming neighbors, the block logical coordinates of
/// neighboring elements are not related by a discrete rotation (i.e. an
/// OrientationMap). Therefore boundary data cannot be copied or projected to a
/// neighbor, but must be interpolated.  As nonconforming neighbors cannot exist
/// in one dimension, this class is not instantiated for Dim = 1.
template <size_t Dim>
class MortarInterpolator {
 public:
  /// The Element with host_id is one of the multiple nonconforming Elements
  /// abutting the single Element with neighbor_id.  The host_mortar_mesh and
  /// neighbor_mortar_mesh are simply the appropriate face Mesh of the host and
  /// neighbor.
  MortarInterpolator(const ElementId<Dim>& host_id,
                     const DirectionalId<Dim>& neighbor_id,
                     const Domain<Dim>& domain,
                     const Mesh<Dim - 1>& host_mortar_mesh,
                     const Mesh<Dim - 1>& neighbor_mortar_mesh);

  MortarInterpolator() = default;

  /// @{
  /// \brief Interpolates the boundary data of the neighbor to the host mortar
  ///
  /// \details neighbor_data is a type-erased Variables so will have a size
  /// equal to the product of the number of components and the number of grid
  /// points of the neighbor_mortar_mesh.  The returned DataVector will have a
  /// size equal to the number of components and the number of grid points of
  /// the host_mortar_mesh.
  void interpolate_to_host(gsl::not_null<DataVector*> result,
                           const DataVector& neighbor_data) const;
  DataVector interpolate_to_host(const DataVector& neighbor_data) const;
  /// @}

  /// @{
  /// \brief Interpolates the boundary data of the host to a subset of the
  /// points of the neighbor mortar
  ///
  /// \details host_data is a type-erased Variables so will have a size
  /// equal to the product of the number of components and the number of grid
  /// points of the host_mortar_mesh.  The returned DataVector will have a
  /// size equal to the number of components and the number of grid points of
  /// the neighbor_mortar_mesh that the host is able to interpolate to.
  void interpolate_to_neighbor(gsl::not_null<DataVector*> result,
                               const DataVector& host_data) const;
  DataVector interpolate_to_neighbor(const DataVector& host_data) const;
  /// @}

  /// \brief The offsets of the data returned by interpolate_to_neighbor into
  /// the full data on the neighbor_mortar_mesh.
  ///
  /// \details  The DataVectors of type-erased Variables store data such that
  /// the grid point index varies fastest (as opposed to the component index).
  /// The size of the returned vector is equal to the number of grid points of
  /// the neighbor_mortar_mesh that the host is able to interpolate to, and the
  /// values of the returned vector map the grid point index of the partial
  /// DataVector returned by interpolate_to_neighbor to the grid point index of
  /// the full DataVector on the neighbor_mortar_mesh
  const std::vector<size_t>& interpolated_neighbor_data_offsets() const {
    return interpolated_neighbor_data_offsets_;
  }

  /// \brief The neighbor mortar mesh that is used to compute the target points
  /// and offsets when interpolating to the neighbor
  const Mesh<Dim - 1>& neighbor_mortar_mesh() const {
    return neighbor_mortar_mesh_;
  }

  /// \brief Updates the interpolator if either of the mortar Mesh change
  void reset_if_necessary(const Domain<Dim>& domain,
                          const Mesh<Dim - 1>& host_mortar_mesh,
                          const Mesh<Dim - 1>& neighbor_mortar_mesh);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalDim>
  friend bool operator==(const MortarInterpolator<LocalDim>& lhs,
                         const MortarInterpolator<LocalDim>& rhs);

  ElementId<Dim> host_id_{};
  DirectionalId<Dim> neighbor_id_{};
  Mesh<Dim - 1> host_mortar_mesh_{};
  Mesh<Dim - 1> neighbor_mortar_mesh_{};
  intrp::Irregular<Dim - 1> neighbor_to_host_interpolant_{};
  intrp::Irregular<Dim - 1> host_to_neighbor_interpolant_{};
  std::vector<size_t> interpolated_neighbor_data_offsets_{};
};

template <>
class MortarInterpolator<1> {
 public:  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*unused*/) {}
};

template <size_t Dim>
bool operator==(const MortarInterpolator<Dim>& lhs,
                const MortarInterpolator<Dim>& rhs);

template <>
inline bool operator==(const MortarInterpolator<1>& /*unused*/,
                       const MortarInterpolator<1>& /*unused*/) {
  return true;
}
}  // namespace dg
