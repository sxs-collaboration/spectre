// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <pup_stl.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StaticCache.hpp"

namespace Filters {
template <size_t Dim, typename TagList>
Hypercube<Dim, TagList>::Hypercube() = default;

template <size_t Dim, typename TagList>
Hypercube<Dim, TagList>::Hypercube(
    const unsigned half_power, const bool enable,
    const std::optional<std::vector<std::string>>& blocks_to_filter,
    const bool volume_filter_on_substep, const bool boundary_filter_on_substep,
    const std::optional<size_t> volume_filter_every_n_steps,
    const std::optional<size_t> boundary_filter_every_n_steps,
    const Options::Context& context)
    : half_power_(half_power),
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
                        << "' found when creating an Exponential filter.");
      }
      blocks_and_groups_to_filter_->push_back(block_name);
    }
  }
}

template <size_t Dim, typename TagList>
void Hypercube<Dim, TagList>::pup(PUP::er& p) {
  Filter<Dim, TagList>::pup(p);
  p | half_power_;
  p | enable_;
  p | blocks_and_groups_to_filter_;
  p | blocks_to_filter_;
  p | volume_filter_on_substep_;
  p | boundary_filter_on_substep_;
  p | volume_filter_every_n_steps_;
  p | boundary_filter_every_n_steps_;
}

template <size_t Dim, typename TagList>
std::unique_ptr<Filter<Dim, TagList>> Hypercube<Dim, TagList>::get_clone()
    const {
  return std::make_unique<Hypercube>(*this);
}

template <size_t Dim, typename TagList>
bool Hypercube<Dim, TagList>::apply_volume_filter_on_substep() const {
  return enable_ and volume_filter_on_substep_;
}

template <size_t Dim, typename TagList>
bool Hypercube<Dim, TagList>::apply_volume_filter_on_this_step(
    const size_t step_number) const {
  if (not enable_ or not volume_filter_every_n_steps_.has_value()) {
    return false;
  }
  return step_number % volume_filter_every_n_steps_.value() == 0;
}

template <size_t Dim, typename TagList>
bool Hypercube<Dim, TagList>::apply_boundary_filter_on_substep() const {
  return enable_ and boundary_filter_on_substep_;
}

template <size_t Dim, typename TagList>
bool Hypercube<Dim, TagList>::apply_boundary_filter_on_this_step(
    const size_t step_number) const {
  if (not enable_ or not boundary_filter_every_n_steps_.has_value()) {
    return false;
  }
  return step_number % boundary_filter_every_n_steps_.value() == 0;
}

template <size_t Dim, typename TagList>
bool Hypercube<Dim, TagList>::supports_mesh(const Mesh<Dim>& mesh) const {
  for (size_t d = 0; d < Dim; ++d) {
    const auto basis = mesh.basis(d);
    const auto quadrature = mesh.quadrature(d);
    const bool supported =
        (basis == Spectral::Basis::Legendre and
         (quadrature == Spectral::Quadrature::Gauss or
          quadrature == Spectral::Quadrature::GaussLobatto)) or
        (basis == Spectral::Basis::Chebyshev and
         (quadrature == Spectral::Quadrature::Gauss or
          quadrature == Spectral::Quadrature::GaussLobatto)) or
        (basis == Spectral::Basis::Fourier and
         quadrature == Spectral::Quadrature::Equiangular) or
        (basis == Spectral::Basis::Cartoon and
         (quadrature == Spectral::Quadrature::AxialSymmetry or
          quadrature == Spectral::Quadrature::SphericalSymmetry));
    if (not supported) {
      return false;
    }
  }
  return true;
}

template <size_t Dim, typename TagList>
const std::optional<std::vector<size_t>>&
Hypercube<Dim, TagList>::blocks_to_filter() const {
  return blocks_to_filter_;
}

template <size_t Dim, typename TagList>
void Hypercube<Dim, TagList>::set_blocks_to_filter(
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

template <size_t Dim, typename TagList>
void Hypercube<Dim, TagList>::apply_in_volume(
    const gsl::not_null<Variables<TagList>*> vars, const Mesh<Dim>& mesh,
    const std::optional<
        InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
    /*inv_jac_grid_to_inertial*/,
    const std::optional<
        Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
    /*jac_grid_to_inertial*/) const {
  const Matrix empty{};
  std::array<std::reference_wrapper<const Matrix>, Dim> filter =
      make_array<Dim>(std::cref(empty));
  for (size_t d = 0; d < Dim; d++) {
    gsl::at(filter, d) = std::cref(filter_matrix(mesh.slice_through(d)));
  }
  *vars = apply_matrices(filter, *vars, mesh.extents());
}

template <size_t Dim, typename TagList>
void Hypercube<Dim, TagList>::apply_on_boundary(
    const gsl::not_null<Variables<TagList>*> vars, const Mesh<Dim - 1>& mesh,
    const std::optional<
        InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
    /*inv_jac_grid_to_inertial*/,
    const std::optional<
        Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
    /*jac_grid_to_inertial*/) const {
  if constexpr (Dim > 1) {
    const Matrix empty{};
    std::array<std::reference_wrapper<const Matrix>, Dim - 1> filter =
        make_array<Dim - 1>(std::cref(empty));
    for (size_t d = 0; d < Dim - 1; d++) {
      gsl::at(filter, d) = std::cref(filter_matrix(mesh.slice_through(d)));
    }
    *vars = apply_matrices(filter, *vars, mesh.extents());
  } else {
    (void)vars;
    (void)mesh;
  }
}

template <size_t Dim, typename TagList>
const Matrix& Hypercube<Dim, TagList>::filter_matrix(
    const Mesh<1>& mesh) const {
  const auto compute_filter = [half_power = half_power_](
                                  const size_t extents,
                                  const Spectral::Basis basis,
                                  const Spectral::Quadrature quadrature) {
    return Spectral::filtering::exponential_filter(
        Mesh<1>{extents, basis, quadrature}, 36.0, half_power);
  };
  const static auto cache = std::make_tuple(
      half_power_,
      make_static_cache<
          CacheRange<1_st, Spectral::maximum_number_of_points<
                               Spectral::Basis::Legendre> +
                               1>,
          CacheEnumeration<Spectral::Basis, Spectral::Basis::Legendre,
                           Spectral::Basis::Chebyshev,
                           Spectral::Basis::Cartoon>,
          CacheEnumeration<Spectral::Quadrature, Spectral::Quadrature::Gauss,
                           Spectral::Quadrature::GaussLobatto,
                           Spectral::Quadrature::AxialSymmetry,
                           Spectral::Quadrature::SphericalSymmetry>>(
          compute_filter),
      make_static_cache<CacheRange<
          1_st, Spectral::maximum_number_of_points<Spectral::Basis::Fourier> +
                    1>>([compute_filter](const size_t extents) {
        return compute_filter(extents, Spectral::Basis::Fourier,
                              Spectral::Quadrature::Equiangular);
      }));
  if (std::get<0>(cache) != half_power_) {
    ERROR("Filter was cached with half power = "
          << std::get<0>(cache) << ", but half power is now " << half_power_
          << ".\nWe currently only support 1 half power per executable per "
             "TagList in the Hypercube filter.");
  }
  if (mesh.basis(0) == Spectral::Basis::Fourier) {
    return std::get<2>(cache)(mesh.extents(0));
  }
  return std::get<1>(cache)(mesh.extents(0), mesh.basis(0), mesh.quadrature(0));
}

template <size_t Dim, typename TagList>
bool operator==(const Hypercube<Dim, TagList>& lhs,
                const Hypercube<Dim, TagList>& rhs) {
  return lhs.half_power_ == rhs.half_power_ and
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

template <size_t Dim, typename TagList>
bool operator!=(const Hypercube<Dim, TagList>& lhs,
                const Hypercube<Dim, TagList>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim, typename TagList>
bool Hypercube<Dim, TagList>::is_equal(
    const Filter<Dim, TagList>& other) const {
  const auto* const other_hypercube =
      dynamic_cast<const Hypercube<Dim, TagList>*>(&other);
  if (other_hypercube == nullptr) {
    return false;
  }
  return *this == *other_hypercube;
}

template <size_t Dim, typename TagList>
PUP::able::PUP_ID Hypercube<Dim, TagList>::my_PUP_ID = 0;  // NOLINT
}  // namespace Filters
