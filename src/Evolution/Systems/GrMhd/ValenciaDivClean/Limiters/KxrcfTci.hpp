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
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace grmhd::ValenciaDivClean::Limiters::Tci {

/// \ingroup LimitersGroup
/// \brief Implements the troubled-cell indicator from Krivodonova et al, 2004,
/// but generalized to the ValenciaDivClean system.
///
/// The KXRCF (these are the author initials) TCI is described in
/// \cite Krivodonova2004.
///
/// Here, instead of applying the TCI to the Newtonian mass density and
/// energy density, we apply to the relativistic \f${\tilde D}\f$ and
/// \f${\tilde \tau}\f$.
///
/// TODO: maybe TildeTau - B^2 is the better analog, to avoid the magnetic
/// field dependence?
template <typename PackagedData>
bool kxrcf_indicator(
    const double kxrcf_constant, const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& tilde_tau, const tnsr::i<DataVector, 3>& tilde_s,
    const Mesh<3>& mesh, const Element<3>& element,
    const std::array<double, 3>& element_size,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian,
    const typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type&
        normals_and_magnitudes,
    const std::unordered_map<
        std::pair<Direction<3>, ElementId<3>>, PackagedData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>&
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
  ASSERT(mesh == Mesh<3>(mesh.extents(0), Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto),
         "The Kxrcf TCI expects a uniform LGL mesh, but got mesh = " << mesh);

  bool inflow_boundaries_present = false;
  double inflow_area = 0.;
  double inflow_delta_tilde_d = 0.;
  double inflow_delta_tilde_tau = 0.;

  // Skip boundary integrations on external boundaries. This choice might be
  // problematic for evolutions (likely only simple test cases) that feed in
  // shocks through the boundary condition: the limiter might fail to activate
  // in the cell that touches the boundary.
  //
  // To properly compute the limiter at external boundaries we would need the
  // limiter to know about the boundary condition, which may be difficult to
  // do in a general way.
  for (const auto& [neighbor, data] : neighbor_data) {
    const auto& dir = neighbor.first;

    // Check consistency of neighbor_data with element and normals
    ASSERT(element.neighbors().contains(dir),
           "Received neighbor data from dir = "
               << dir << ", but element has no neighbor in this dir");
    ASSERT(normals_and_magnitudes.contains(dir),
           "Received neighbor data from dir = "
               << dir
               << ", but normals_and_magnitudes has no normal in this dir");
    ASSERT(normals_and_magnitudes.at(dir).has_value(),
           "The normals_and_magnitudes are not up-to-date in dir = " << dir);
    const auto& normal = get<evolution::dg::Tags::NormalCovector<3>>(
        normals_and_magnitudes.at(dir).value());
    const auto& magnitude_of_normal =
        get<evolution::dg::Tags::MagnitudeOfNormal>(
            normals_and_magnitudes.at(dir).value());

    const size_t sliced_dim = dir.dimension();
    const size_t index_of_slice =
        (dir.side() == Side::Lower ? 0 : mesh.extents()[sliced_dim] - 1);
    const auto momentum_on_slice =
        data_on_slice(tilde_s, mesh.extents(), sliced_dim, index_of_slice);
    const auto momentum_dot_normal = dot_product(momentum_on_slice, normal);

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
    // given by the product of the volume Jacobian determinant with the
    // magnitude of the unnormalized normal covectors.
    const DataVector weighted_inflow_mask =
        inflow_mask *
        get(data_on_slice(det_logical_to_inertial_jacobian, mesh.extents(),
                          sliced_dim, index_of_slice)) *
        get(magnitude_of_normal);

    inflow_area +=
        definite_integral(weighted_inflow_mask, mesh.slice_away(sliced_dim));

    // This is the step that is incompatible with h/p refinement. For use with
    // h/p refinement, would need to correctly obtain the neighbor solution on
    // the local grid points.
    const auto neighbor_vars_on_slice = data_on_slice(
        data.volume_data, mesh.extents(), sliced_dim, index_of_slice);

    const auto tilde_d_on_slice =
        data_on_slice(tilde_d, mesh.extents(), sliced_dim, index_of_slice);
    const auto& neighbor_tilde_d_on_slice =
        get<grmhd::ValenciaDivClean::Tags::TildeD>(neighbor_vars_on_slice);
    inflow_delta_tilde_d += definite_integral(
        (get(tilde_d_on_slice) - get(neighbor_tilde_d_on_slice)) *
            weighted_inflow_mask,
        mesh.slice_away(sliced_dim));

    // TODO: below we might want to use a variable that better matches the
    // Newtonian energy density. would need to convert surface data as well.
    const auto tilde_tau_on_slice =
        data_on_slice(tilde_tau, mesh.extents(), sliced_dim, index_of_slice);
    const auto& neighbor_tilde_tau_on_slice =
        get<grmhd::ValenciaDivClean::Tags::TildeTau>(neighbor_vars_on_slice);
    inflow_delta_tilde_tau += definite_integral(
        (get(tilde_tau_on_slice) - get(neighbor_tilde_tau_on_slice)) *
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

  const double norm_squared_tilde_d = mean_value(
      get(det_logical_to_inertial_jacobian) * square(get(tilde_d)), mesh);
  ASSERT(norm_squared_tilde_d > 0.,
         "Sanity check failed: negative TildeD norm over element");
  const double norm_tilde_d = sqrt(norm_squared_tilde_d);
  const double ratio_for_tilde_d =
      abs(inflow_delta_tilde_d) / (h_pow * inflow_area * norm_tilde_d);

  const double norm_squared_tilde_tau = mean_value(
      get(det_logical_to_inertial_jacobian) * square(get(tilde_tau)), mesh);
  ASSERT(norm_squared_tilde_tau > 0.,
         "Sanity check failed: negative energy norm over element");
  const double norm_tilde_tau = sqrt(norm_squared_tilde_tau);
  const double ratio_for_tilde_tau =
      abs(inflow_delta_tilde_tau) / (h_pow * inflow_area * norm_tilde_tau);

  return (ratio_for_tilde_d > kxrcf_constant or
          ratio_for_tilde_tau > kxrcf_constant);
}

}  // namespace grmhd::ValenciaDivClean::Limiters::Tci
