// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.hpp"

#include <array>
#include <functional>
#include <optional>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Filters {
template <typename TagList>
SphericalShell<TagList>::SphericalShell(
    const size_t num_modes_to_kill,
    const std::optional<size_t> angular_half_power,
    const std::optional<size_t> radial_half_power, const bool enable,
    const std::optional<std::vector<std::string>>& blocks_to_filter,
    const bool volume_filter_on_substep, const bool boundary_filter_on_substep,
    const std::optional<size_t> volume_filter_every_n_steps,
    const std::optional<size_t> boundary_filter_every_n_steps,
    const Options::Context& context)
    : num_modes_to_kill_(num_modes_to_kill),
      angular_half_power_(angular_half_power),
      radial_half_power_(radial_half_power),
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
                        << "' found when creating a SphericalShell filter.");
      }
      blocks_and_groups_to_filter_->push_back(block_name);
    }
  }
}

template <typename TagList>
void SphericalShell<TagList>::pup(PUP::er& p) {
  Filter<3, TagList>::pup(p);
  p | num_modes_to_kill_;
  p | angular_half_power_;
  p | radial_half_power_;
  p | enable_;
  p | blocks_and_groups_to_filter_;
  p | blocks_to_filter_;
  p | volume_filter_on_substep_;
  p | boundary_filter_on_substep_;
  p | volume_filter_every_n_steps_;
  p | boundary_filter_every_n_steps_;
}

template <typename TagList>
std::unique_ptr<Filter<3, TagList>> SphericalShell<TagList>::get_clone() const {
  return std::make_unique<SphericalShell>(*this);
}

template <typename TagList>
bool SphericalShell<TagList>::apply_volume_filter_on_substep() const {
  return enable_ and volume_filter_on_substep_;
}

template <typename TagList>
bool SphericalShell<TagList>::apply_volume_filter_on_this_step(
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
bool SphericalShell<TagList>::apply_boundary_filter_on_substep() const {
  return enable_ and boundary_filter_on_substep_;
}

template <typename TagList>
bool SphericalShell<TagList>::apply_boundary_filter_on_this_step(
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
SphericalShell<TagList>::blocks_to_filter() const {
  return blocks_to_filter_;
}

template <typename TagList>
void SphericalShell<TagList>::set_blocks_to_filter(
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

template <typename TagList>
bool SphericalShell<TagList>::supports_mesh(const Mesh<3>& mesh) const {
  // Radial direction (dim 0): Legendre or Chebyshev with Gauss or GaussLobatto
  const auto radial_basis = mesh.basis(0);
  const auto radial_quadrature = mesh.quadrature(0);
  const bool radial_ok =
      (radial_basis == Spectral::Basis::Legendre or
       radial_basis == Spectral::Basis::Chebyshev) and
      (radial_quadrature == Spectral::Quadrature::Gauss or
       radial_quadrature == Spectral::Quadrature::GaussLobatto);
  if (not radial_ok) {
    return false;
  }
  // Angular directions (dims 1 & 2): SphericalHarmonic basis
  // dim 1: Gauss quadrature; dim 2: Equiangular quadrature
  return mesh.basis(1) == Spectral::Basis::SphericalHarmonic and
         mesh.quadrature(1) == Spectral::Quadrature::Gauss and
         mesh.basis(2) == Spectral::Basis::SphericalHarmonic and
         mesh.quadrature(2) == Spectral::Quadrature::Equiangular;
}

template <typename TagList>
void SphericalShell<TagList>::apply_in_volume(
    const gsl::not_null<Variables<TagList>*> vars, const Mesh<3>& mesh,
    const std::optional<
        InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
        inv_jac_grid_to_inertial,
    const std::optional<Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
        jac_grid_to_inertial) const {
  if (not inv_jac_grid_to_inertial.has_value()) {
    ERROR(
        "The inv_jac_grid_to_inertial must be set when using the "
        "SphericalShell filter.");
  }
  if (not jac_grid_to_inertial.has_value()) {
    ERROR(
        "The jac_grid_to_inertial must be set when using the "
        "SphericalShell filter.");
  }
  const size_t radial_extents = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;

  // Cache the angular filter matrices
  if (cached_l_max_ != l_max) {
    fill_tensor_ylm_filters<TagList>(make_not_null(&filter_matrices_), l_max,
                                     num_modes_to_kill_, angular_half_power_,
                                     normalization_);
    cached_l_max_ = l_max;
  }

  // Resize the temporary spectral buffer used by apply_tensor_ylm_filter.
  // It must hold at least radial_extents * spectral_size grid points (the
  // spectral coefficient array is larger than the physical-space grid).
  const size_t needed_temp_size =
      radial_extents * ::ylm::get_spherepack_cache(l_max).spectral_size();
  if (temp_storage_.number_of_grid_points() != needed_temp_size) {
    temp_storage_.initialize(needed_temp_size);
  }

  // Apply the angular filter. apply_tensor_ylm_filter expects
  //   3rd arg: InverseJacobian<Inertial, Grid> (== Jacobian<Grid, Inertial>)
  //   4th arg: InverseJacobian<Grid, Inertial>
  apply_tensor_ylm_filter(vars, make_not_null(&temp_storage_),
                          jac_grid_to_inertial.value(),
                          inv_jac_grid_to_inertial.value(), filter_matrices_,
                          l_max, radial_extents);

  // Apply the optional radial exponential filter
  if (radial_half_power_.has_value()) {
    if (cached_radial_extents_ != radial_extents) {
      cached_radial_filter_matrix_ = Spectral::filtering::exponential_filter(
          mesh.slice_through(0), 36.0,
          static_cast<unsigned>(*radial_half_power_));
      cached_radial_extents_ = radial_extents;
    }
    const Matrix empty{};
    const std::array<std::reference_wrapper<const Matrix>, 3> radial_filter{
        std::cref(cached_radial_filter_matrix_), std::cref(empty),
        std::cref(empty)};
    *vars = apply_matrices(radial_filter, *vars, mesh.extents());
  }
}

template <typename TagList>
void SphericalShell<TagList>::apply_on_boundary(
    const gsl::not_null<Variables<TagList>*> vars, const Mesh<2>& mesh,
    const std::optional<
        InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
        inv_jac_grid_to_inertial,
    const std::optional<Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
        jac_grid_to_inertial) const {
  if (mesh.basis(0) != Spectral::Basis::SphericalHarmonic or
      mesh.basis(1) != Spectral::Basis::SphericalHarmonic) {
    ERROR(
        "SphericalShell filter called on a face mesh whose angular dimensions "
        "do not use the SphericalHarmonic basis. Got basis ("
        << mesh.basis(0) << ", " << mesh.basis(1) << ").");
  }
  if (not inv_jac_grid_to_inertial.has_value()) {
    ERROR(
        "The inv_jac_grid_to_inertial must be set when using the "
        "SphericalShell filter.");
  }
  if (not jac_grid_to_inertial.has_value()) {
    ERROR(
        "The jac_grid_to_inertial must be set when using the "
        "SphericalShell filter.");
  }
  const size_t l_max = mesh.extents(0) - 1;

  // Cache the angular filter matrices
  if (cached_l_max_ != l_max) {
    fill_tensor_ylm_filters<TagList>(make_not_null(&filter_matrices_), l_max,
                                     num_modes_to_kill_, angular_half_power_,
                                     normalization_);
    cached_l_max_ = l_max;
  }

  const size_t needed_temp_size =
      ::ylm::get_spherepack_cache(l_max).spectral_size();
  if (temp_storage_.number_of_grid_points() != needed_temp_size) {
    temp_storage_.initialize(needed_temp_size);
  }

  apply_tensor_ylm_filter(vars, make_not_null(&temp_storage_),
                          jac_grid_to_inertial.value(),
                          inv_jac_grid_to_inertial.value(), filter_matrices_,
                          l_max, 1);
}

template <typename TagList>
bool operator==(const SphericalShell<TagList>& lhs,
                const SphericalShell<TagList>& rhs) {
  return lhs.num_modes_to_kill_ == rhs.num_modes_to_kill_ and
         lhs.angular_half_power_ == rhs.angular_half_power_ and
         lhs.radial_half_power_ == rhs.radial_half_power_ and
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
bool operator!=(const SphericalShell<TagList>& lhs,
                const SphericalShell<TagList>& rhs) {
  return not(lhs == rhs);
}

template <typename TagList>
bool SphericalShell<TagList>::is_equal(const Filter<3, TagList>& other) const {
  const auto* const other_shell =
      dynamic_cast<const SphericalShell<TagList>*>(&other);
  if (other_shell == nullptr) {
    return false;
  }
  return *this == *other_shell;
}

template <typename TagList>
PUP::able::PUP_ID SphericalShell<TagList>::my_PUP_ID = 0;  // NOLINT
}  // namespace Filters
