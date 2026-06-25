// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearOperators/Filters/HollowCylinder.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Filters {
template <typename TagList>
HollowCylinder<TagList>::HollowCylinder(
    const size_t num_modes_to_kill,
    const std::optional<size_t> angular_half_power,
    const std::optional<size_t> radial_half_power,
    const std::optional<size_t> z_half_power, const bool enable,
    const std::optional<std::vector<std::string>>& blocks_to_filter,
    const bool volume_filter_on_substep, const bool boundary_filter_on_substep,
    const std::optional<size_t> volume_filter_every_n_steps,
    const std::optional<size_t> boundary_filter_every_n_steps,
    const Options::Context& context)
    : num_modes_to_kill_(num_modes_to_kill),
      angular_half_power_(angular_half_power),
      radial_half_power_(radial_half_power),
      z_half_power_(z_half_power),
      enable_(enable),
      volume_filter_on_substep_(volume_filter_on_substep),
      boundary_filter_on_substep_(boundary_filter_on_substep),
      volume_filter_every_n_steps_(volume_filter_every_n_steps),
      boundary_filter_every_n_steps_(boundary_filter_every_n_steps) {
  if (blocks_to_filter.has_value()) {
    blocks_and_groups_to_filter_ = std::vector<std::string>{};
    std::unordered_set<std::string> seen{};
    for (const std::string& block_name : blocks_to_filter.value()) {
      if (not seen.emplace(block_name).second) {
        PARSE_ERROR(context,
                    "Duplicate block name '"
                        << block_name
                        << "' found when creating a HollowCylinder filter.");
      }
      blocks_and_groups_to_filter_->push_back(block_name);
    }
  }
}

template <typename TagList>
void HollowCylinder<TagList>::pup(PUP::er& p) {
  Filter<3, TagList>::pup(p);
  p | num_modes_to_kill_;
  p | angular_half_power_;
  p | radial_half_power_;
  p | z_half_power_;
  p | enable_;
  p | blocks_and_groups_to_filter_;
  p | blocks_to_filter_;
  p | volume_filter_on_substep_;
  p | boundary_filter_on_substep_;
  p | volume_filter_every_n_steps_;
  p | boundary_filter_every_n_steps_;
}

template <typename TagList>
std::unique_ptr<Filter<3, TagList>> HollowCylinder<TagList>::get_clone() const {
  return std::make_unique<HollowCylinder>(*this);
}

template <typename TagList>
bool HollowCylinder<TagList>::apply_volume_filter_on_substep() const {
  return enable_ and volume_filter_on_substep_;
}

template <typename TagList>
bool HollowCylinder<TagList>::apply_volume_filter_on_this_step(
    const size_t step_number) const {
  if (not enable_) {
    return false;
  }
  if (volume_filter_every_n_steps_.has_value()) {
    return step_number % volume_filter_every_n_steps_.value() == 0;
  } else {
    return false;
  }
}

template <typename TagList>
bool HollowCylinder<TagList>::apply_boundary_filter_on_substep() const {
  return enable_ and boundary_filter_on_substep_;
}

template <typename TagList>
bool HollowCylinder<TagList>::apply_boundary_filter_on_this_step(
    const size_t step_number) const {
  if (not enable_) {
    return false;
  }
  if (boundary_filter_every_n_steps_.has_value()) {
    return step_number % boundary_filter_every_n_steps_.value() == 0;
  } else {
    return false;
  }
}

template <typename TagList>
const std::optional<std::vector<size_t>>&
HollowCylinder<TagList>::blocks_to_filter() const {
  return blocks_to_filter_;
}

template <typename TagList>
void HollowCylinder<TagList>::set_blocks_to_filter(
    const std::vector<std::string>& all_block_names,
    const std::unordered_map<std::string, std::unordered_set<std::string>>&
        block_groups) {
  if (not blocks_and_groups_to_filter_.has_value()) {
    blocks_to_filter_ = std::nullopt;
    return;
  }
  if (all_block_names.empty()) {
    ERROR(
        "The domain chosen doesn't use block names, but the filter has "
        "specified block names to use.");
  }
  blocks_to_filter_ = domain::block_ids_from_names(
      blocks_and_groups_to_filter_.value(), all_block_names, block_groups);
}

namespace HollowCylinder_detail {
inline bool is_legendre_or_chebyshev(const Spectral::Basis basis,
                                     const Spectral::Quadrature quadrature) {
  return (basis == Spectral::Basis::Legendre or
          basis == Spectral::Basis::Chebyshev) and
         (quadrature == Spectral::Quadrature::Gauss or
          quadrature == Spectral::Quadrature::GaussLobatto);
}
}  // namespace HollowCylinder_detail

template <typename TagList>
bool HollowCylinder<TagList>::supports_mesh(const Mesh<3>& mesh) const {
  // Radial direction (dim 0): Legendre or Chebyshev
  if (not HollowCylinder_detail::is_legendre_or_chebyshev(mesh.basis(0),
                                                          mesh.quadrature(0))) {
    return false;
  }
  // Angular direction (dim 1): Fourier with Equiangular quadrature
  if (mesh.basis(1) != Spectral::Basis::Fourier or
      mesh.quadrature(1) != Spectral::Quadrature::Equiangular) {
    return false;
  }
  // Axial z direction (dim 2): Legendre or Chebyshev
  return HollowCylinder_detail::is_legendre_or_chebyshev(mesh.basis(2),
                                                         mesh.quadrature(2));
}

template <typename TagList>
const Matrix& HollowCylinder<TagList>::exponential_filter_matrix(
    const std::optional<size_t> half_power, const Mesh<1>& mesh_1d,
    SingleExtentCache& cache) const {
  if (not half_power.has_value()) {
    return empty_matrix_;
  }
  const size_t extent = mesh_1d.extents(0);
  if (cache.extent != extent or cache.half_power != half_power) {
    cache.matrix = Spectral::filtering::exponential_filter(
        mesh_1d, 36.0, static_cast<unsigned>(*half_power));
    cache.extent = extent;
    cache.half_power = half_power;
  }
  return cache.matrix;
}

template <typename TagList>
const Matrix& HollowCylinder<TagList>::angular_filter_matrix(
    const Mesh<1>& mesh_1d) const {
  if (num_modes_to_kill_ == 0 and not angular_half_power_.has_value()) {
    return empty_matrix_;
  }
  const size_t extent = mesh_1d.extents(0);
  if (cached_angular_filter_.extent != extent or
      cached_angular_filter_.half_power != angular_half_power_ or
      cached_angular_filter_.num_modes_to_kill != num_modes_to_kill_) {
    Matrix combined{};
    if (angular_half_power_.has_value()) {
      combined = Spectral::filtering::exponential_filter(
          mesh_1d, 36.0, static_cast<unsigned>(*angular_half_power_));
    }
    if (num_modes_to_kill_ > 0) {
      const Matrix& cutoff =
          Spectral::filtering::zero_highest_modes(mesh_1d, num_modes_to_kill_);
      combined =
          angular_half_power_.has_value() ? Matrix{cutoff * combined} : cutoff;
    }
    cached_angular_filter_.matrix = std::move(combined);
    cached_angular_filter_.extent = extent;
    cached_angular_filter_.half_power = angular_half_power_;
    cached_angular_filter_.num_modes_to_kill = num_modes_to_kill_;
  }
  return cached_angular_filter_.matrix;
}

template <typename TagList>
void HollowCylinder<TagList>::apply_in_volume(
    const gsl::not_null<Variables<TagList>*> vars, const Mesh<3>& mesh,
    const std::optional<
        InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
    /*inv_jac_grid_to_inertial*/,
    const std::optional<Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
    /*jac_grid_to_inertial*/) const {
  const Matrix& radial = exponential_filter_matrix(
      radial_half_power_, mesh.slice_through(0), cached_radial_filter_);
  const Matrix& angular = angular_filter_matrix(mesh.slice_through(1));
  const Matrix& z = exponential_filter_matrix(
      z_half_power_, mesh.slice_through(2), cached_z_filter_);
  if (radial.columns() == 0 and angular.columns() == 0 and z.columns() == 0) {
    return;
  }
  const std::array<std::reference_wrapper<const Matrix>, 3> filter{
      std::cref(radial), std::cref(angular), std::cref(z)};
  *vars = apply_matrices(filter, *vars, mesh.extents());
}

template <typename TagList>
void HollowCylinder<TagList>::apply_on_boundary(
    const gsl::not_null<Variables<TagList>*> vars, const Mesh<2>& mesh,
    const std::optional<
        InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
    /*inv_jac_grid_to_inertial*/,
    const std::optional<Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
    /*jac_grid_to_inertial*/) const {
  const bool fourier_0 = mesh.basis(0) == Spectral::Basis::Fourier;
  const bool fourier_1 = mesh.basis(1) == Spectral::Basis::Fourier;
  // A hollow-cylinder face drops one volume dimension while preserving the
  // order of the remaining two, so the basis pattern of the 2-D face mesh
  // uniquely identifies which physical directions it spans:
  //   (Fourier, Legendre) -> (angular, z)   [radial face]
  //   (Legendre, Fourier) -> (radial, angular)   [z face]
  // The angular (Fourier) direction is periodic, so it has no boundary faces;
  // a face with no Fourier direction would require slicing it away, which
  // cannot occur.
  std::array<std::reference_wrapper<const Matrix>, 2> filter{
      std::cref(empty_matrix_), std::cref(empty_matrix_)};
  if (fourier_0 and not fourier_1) {
    filter[0] = std::cref(angular_filter_matrix(mesh.slice_through(0)));
    filter[1] = std::cref(exponential_filter_matrix(
        z_half_power_, mesh.slice_through(1), cached_z_filter_));
  } else if (not fourier_0 and fourier_1) {
    filter[0] = std::cref(exponential_filter_matrix(
        radial_half_power_, mesh.slice_through(0), cached_radial_filter_));
    filter[1] = std::cref(angular_filter_matrix(mesh.slice_through(1)));
  } else if (not fourier_0 and not fourier_1) {
    ERROR(
        "The HollowCylinder boundary filter was given a face with no angular "
        "(Fourier) direction (face basis "
        << mesh.basis(0) << ", " << mesh.basis(1)
        << "). Such a face is obtained by slicing away the angular direction "
           "of the volume mesh, but the angular direction is periodic and so "
           "has no boundary faces. The boundary filter is only valid on the "
           "radial face (angular, z) and the axial face (radial, angular); "
           "there is no angular face to filter.");
  } else {
    ERROR(
        "The HollowCylinder boundary filter was given a face with two angular "
        "(Fourier) directions (face basis "
        << mesh.basis(0) << ", " << mesh.basis(1)
        << "). A hollow-cylinder volume mesh has exactly one angular (Fourier) "
           "direction, so any face can contain at most one.");
  }
  if (filter[0].get().columns() == 0 and filter[1].get().columns() == 0) {
    return;
  }
  *vars = apply_matrices(filter, *vars, mesh.extents());
}

template <typename TagList>
bool operator==(const HollowCylinder<TagList>& lhs,
                const HollowCylinder<TagList>& rhs) {
  return lhs.num_modes_to_kill_ == rhs.num_modes_to_kill_ and
         lhs.angular_half_power_ == rhs.angular_half_power_ and
         lhs.radial_half_power_ == rhs.radial_half_power_ and
         lhs.z_half_power_ == rhs.z_half_power_ and
         lhs.enable_ == rhs.enable_ and
         lhs.blocks_and_groups_to_filter_ ==
             rhs.blocks_and_groups_to_filter_ and
         lhs.blocks_to_filter_ == rhs.blocks_to_filter_ and
         lhs.volume_filter_on_substep_ == rhs.volume_filter_on_substep_ and
         lhs.boundary_filter_on_substep_ == rhs.boundary_filter_on_substep_ and
         lhs.volume_filter_every_n_steps_ ==
             rhs.volume_filter_every_n_steps_ and
         lhs.boundary_filter_every_n_steps_ ==
             rhs.boundary_filter_every_n_steps_;
}

template <typename TagList>
bool operator!=(const HollowCylinder<TagList>& lhs,
                const HollowCylinder<TagList>& rhs) {
  return not(lhs == rhs);
}

template <typename TagList>
bool HollowCylinder<TagList>::is_equal(const Filter<3, TagList>& other) const {
  const auto* const other_cylinder =
      dynamic_cast<const HollowCylinder<TagList>*>(&other);
  if (other_cylinder == nullptr) {
    return false;
  }
  return *this == *other_cylinder;
}

template <typename TagList>
PUP::able::PUP_ID HollowCylinder<TagList>::my_PUP_ID = 0;  // NOLINT
}  // namespace Filters
