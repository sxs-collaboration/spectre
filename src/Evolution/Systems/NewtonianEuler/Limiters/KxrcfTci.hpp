// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace NewtonianEuler {
namespace Limiters {
namespace Tci {

/// \ingroup LimitersGroup
/// \brief Implements the troubled-cell indicator from Krivodonova et al, 2004.
///
/// The KXRCF (these are the author initials) TCI is described in
/// \cite Krivodonova2004.
///
/// In summary, this TCI uses the size of discontinuities between neighboring DG
/// elements to determine the smoothness of the solution. This works because the
/// discontinuities converge rapidly for smooth solutions, therefore a large
/// discontinuity suggests a lack of smoothness and the need to apply a limiter.
///
/// The reference sets the constant we call `kxrcf_constant` to 1. This should
/// generally be a good threshold to use, though it might not be the optimal
/// value (in balancing robustness vs accuracy) for any particular problem.
///
/// This implementation
/// - does not support h- or p-refinement; this is checked by assertion.
/// - chooses not to check external boundaries, because this adds complexity.
///   However, by not checking external boundaries, the implementation may not
///   be robust for problems that feed in shocks through boundary conditions.
template <size_t VolumeDim, typename PackagedData>
bool kxrcf_indicator(
    const double kxrcf_constant, const Scalar<DataVector>& cons_mass_density,
    const tnsr::I<DataVector, VolumeDim>& cons_momentum_density,
    const Scalar<DataVector>& cons_energy_density, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian,
    const std::unordered_map<Direction<VolumeDim>,
                             tnsr::i<DataVector, VolumeDim>>&
        unnormalized_normals,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  // Enforce restrictions on h-refinement, p-refinement
  if (UNLIKELY(alg::any_of(element.neighbors(),
                           [](const auto& direction_neighbors) noexcept {
                             return direction_neighbors.second.size() != 1;
                           }))) {
    ERROR("The Kxrcf TCI does not yet support h-refinement");
    // Removing this limitation will require adapting the surface integrals to
    // correctly acount for,
    // - multiple (smaller) neighbors contributing to the integral
    // - only a portion of a (larger) neighbor contributing to the integral
  }
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) noexcept {
    if (UNLIKELY(neighbor_and_data.second.mesh != mesh)) {
      ERROR("The Kxrcf TCI does not yet support p-refinement");
      // Removing this limitation will require generalizing the surface
      // integrals to make sure the meshes are consistent.
    }
  });
  // Check the mesh matches expectations:
  // - the extents must be uniform, because the TCI expects a unique "order" for
  //   the DG scheme. This shows up when computing the threshold parameter,
  //   h^(degree+1)/2
  // - the quadrature must be GL, because the current implementation doesn't
  //   handle extrapolation to the boundary, though this could be changed.
  // - the basis in principle could be changed, but until we need to change it
  //   we just check it's the expected Legendre for simplicity
  ASSERT(mesh == Mesh<VolumeDim>(mesh.extents(0), Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto),
         "The Kxrcf TCI expects a uniform LGL mesh, but got mesh = " << mesh);

  bool inflow_boundaries_present = false;
  double inflow_area = 0.;
  double inflow_delta_density = 0.;
  double inflow_delta_energy = 0.;

  for (const auto& [neighbor, data] : neighbor_data) {
    // Skip computations on boundary if the boundary is external. This choice
    // might be problematic for evolutions (likely only simple test cases) that
    // feed in shocks through the boundary condition: the limiter might fail to
    // activate in the cell that touches the boundary.
    //
    // Note that to do this properly we would need the limiter to know about the
    // boundary condition in general, which may be difficult. Furthermore, such
    // a change may also require changing the tags at the call site to ensure
    // that ALL face normals are grabbed from the databox (and not only the
    // internal face normals, as occurs when grabbing
    // `Tags::Interface<Tags::InternalDirections, ...>`).
    const auto& dir = neighbor.first;
    if (unnormalized_normals.find(dir) == unnormalized_normals.end()) {
      continue;
    }

    const size_t sliced_dim = dir.dimension();
    const size_t index_of_slice =
        (dir.side() == Side::Lower ? 0 : mesh.extents()[sliced_dim] - 1);
    const auto momentum_on_slice = data_on_slice(
        cons_momentum_density, mesh.extents(), sliced_dim, index_of_slice);
    const auto momentum_dot_normal =
        dot_product(momentum_on_slice, unnormalized_normals.at(dir));

    // Skip boundaries with no significant inflow
    // Note: the cutoff value here is small but arbitrarily chosen.
    if (min(get(momentum_dot_normal)) > -1e-12) {
      continue;
    }
    inflow_boundaries_present = true;

    // This mask has value 1. for momentum_dot_normal < 0.
    //                     0. for momentum_dot_normal >= 0.
    const DataVector inflow_mask = 1. - step_function(get(momentum_dot_normal));
    // Mask is then weighted pointwise by the Jacobian determinant giving
    // surface integrals in inertial coordinates. This Jacobian determinant is
    // given by the product of the volume Jacobian determinant with the norm of
    // the unnormalized face normals.
    const DataVector weighted_inflow_mask =
        inflow_mask *
        get(data_on_slice(det_logical_to_inertial_jacobian, mesh.extents(),
                          sliced_dim, index_of_slice)) *
        get(magnitude(unnormalized_normals.at(dir)));

    inflow_area +=
        definite_integral(weighted_inflow_mask, mesh.slice_away(sliced_dim));

    // This is the step that is incompatible with h/p refinement. For use with
    // h/p refinement, would need to correctly obtain the neighbor solution on
    // the local grid points.
    const auto neighbor_vars_on_slice = data_on_slice(
        data.volume_data, mesh.extents(), sliced_dim, index_of_slice);

    const auto density_on_slice = data_on_slice(
        cons_mass_density, mesh.extents(), sliced_dim, index_of_slice);
    const auto& neighbor_density_on_slice =
        get<NewtonianEuler::Tags::MassDensityCons>(neighbor_vars_on_slice);
    inflow_delta_density += definite_integral(
        (get(density_on_slice) - get(neighbor_density_on_slice)) *
            weighted_inflow_mask,
        mesh.slice_away(sliced_dim));

    const auto energy_on_slice = data_on_slice(
        cons_energy_density, mesh.extents(), sliced_dim, index_of_slice);
    const auto& neighbor_energy_on_slice =
        get<NewtonianEuler::Tags::EnergyDensity>(neighbor_vars_on_slice);
    inflow_delta_energy += definite_integral(
        (get(energy_on_slice) - get(neighbor_energy_on_slice)) *
            weighted_inflow_mask,
        mesh.slice_away(sliced_dim));
  }

  if (not inflow_boundaries_present) {
    // No boundaries had inflow, so not a troubled cell
    return false;
  }

  // KXRCF take h to be the radius of the circumscribed circle
  const double h = 0.5 * magnitude(element_size);
  const double h_pow = pow(h, 0.5 * mesh.extents(0));

  ASSERT(inflow_area > 0.,
         "Sanity check failed: negative area of inflow boundaries");

  const double norm_squared_density = mean_value(
      get(det_logical_to_inertial_jacobian) * square(get(cons_mass_density)),
      mesh);
  ASSERT(norm_squared_density > 0.,
         "Sanity check failed: negative density norm over element");
  const double norm_density = sqrt(norm_squared_density);
  const double ratio_for_density =
      abs(inflow_delta_density) / (h_pow * inflow_area * norm_density);

  const double norm_squared_energy = mean_value(
      get(det_logical_to_inertial_jacobian) * square(get(cons_energy_density)),
      mesh);
  ASSERT(norm_squared_energy > 0.,
         "Sanity check failed: negative energy norm over element");
  const double norm_energy = sqrt(norm_squared_energy);
  const double ratio_for_energy =
      abs(inflow_delta_energy) / (h_pow * inflow_area * norm_energy);

  return (ratio_for_density > kxrcf_constant or
          ratio_for_energy > kxrcf_constant);
}

}  // namespace Tci
}  // namespace Limiters
}  // namespace NewtonianEuler
