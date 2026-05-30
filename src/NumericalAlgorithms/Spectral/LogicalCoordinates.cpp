// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"

#include <array>
#include <numbers>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/Spherepack.hpp"

template <size_t VolumeDim>
void logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, VolumeDim, Frame::ElementLogical>*>
        logical_coords,
    const Mesh<VolumeDim>& mesh) {
  set_number_of_grid_points(logical_coords, mesh.number_of_grid_points());

  const auto I1 = [&logical_coords, &mesh](const size_t dim) {
    const auto& collocation_points_in_this_dim =
        Spectral::collocation_points(mesh.slice_through(dim));
    for (IndexIterator<VolumeDim> index(mesh.extents()); index; ++index) {
      logical_coords->get(dim)[index.collapsed_index()] =
          collocation_points_in_this_dim[index()[dim]];
    }
  };
  const auto S1 = [&logical_coords, &mesh](const size_t dim) {
    const size_t n_phi = mesh.extents(dim);
    const double two_pi_over_n_phi = 2.0 * std::numbers::pi / n_phi;
    for (IndexIterator<VolumeDim> index(mesh.extents()); index; ++index) {
      logical_coords->get(dim)[index.collapsed_index()] =
          two_pi_over_n_phi * index()[dim];
    }
  };
  const auto S2_gauss = [&logical_coords, &mesh](const size_t dim) {
    const size_t n_theta = mesh.extents(dim);
    std::vector<double> theta(n_theta + 1);
    DataVector temp(2 * n_theta + 1);
    auto work = gsl::make_span(temp.data(), n_theta);
    auto unused_weights = gsl::make_span(temp.data() + n_theta, n_theta + 1);

    int err = 0;
    gaqd_(static_cast<int>(n_theta), theta.data(), unused_weights.data(),
          work.data(), static_cast<int>(unused_weights.size()), &err);
    if (UNLIKELY(err != 0)) {
      ERROR("gaqd error " << err << " in LogicalCoordinates");
    }
    for (IndexIterator<VolumeDim> index(mesh.extents()); index; ++index) {
      logical_coords->get(dim)[index.collapsed_index()] = theta[index()[dim]];
    }
  };

  for (size_t d = 0; d < VolumeDim; ++d) {
    switch (mesh.basis(d)) {
      case Spectral::Basis::SphericalHarmonic: {
        switch (mesh.quadrature(d)) {
          case Spectral::Quadrature::Gauss: {
            S2_gauss(d);
            break;
          }
          case Spectral::Quadrature::Equiangular: {
            S1(d);
            break;
          }
          default:
            ERROR(
                "Quadrature must be Gauss or Equiangular for Basis "
                "SphericalHarmonic");
        }
        break;
      }
      case Spectral::Basis::Fourier: {
        switch (mesh.quadrature(d)) {
          case Spectral::Quadrature::Equiangular: {
            S1(d);
            break;
          }
          default:
            ERROR("Quadrature must be Equiangular for Basis Fourier");
        }
        break;
      }
      case Spectral::Basis::ZernikeB2: {
        switch (mesh.quadrature(d)) {
          case Spectral::Quadrature::GaussRadauUpper: {
            I1(d);
            break;
          }
          case Spectral::Quadrature::Equiangular: {
            S1(d);
            break;
          }
          default:
            ERROR(
                "Quadrature must be GaussRadauUpper or Equiangular for Basis "
                "ZernikeB2");
        }
        break;
      }
      case Spectral::Basis::ZernikeB3: {
        switch (mesh.quadrature(d)) {
          case Spectral::Quadrature::GaussRadauUpper: {
            I1(d);
            break;
          }
          case Spectral::Quadrature::Gauss: {
            S2_gauss(d);
            break;
          }
          case Spectral::Quadrature::Equiangular: {
            S1(d);
            break;
          }
          default:
            ERROR(
                "Quadrature must be GaussRadauUpper, Equiangular, for Gauss "
                "for Basis ZernikeB3");
        }
        break;
      }
      case Spectral::Basis::Cartoon: {
        if (mesh.extents(d) != 1) {
          ERROR("Only 1 grid point is allowed in a Cartoon basis.");
        }
        for (IndexIterator<VolumeDim> index(mesh.extents()); index; ++index) {
          logical_coords->get(d)[index.collapsed_index()] = 0.0;
        }
        break;
      }
      // NOLINTNEXTLINE(bugprone-branch-clone)
      case Spectral::Basis::Chebyshev:
        [[fallthrough]];
      case Spectral::Basis::Legendre:
        [[fallthrough]];
      case Spectral::Basis::ZernikeB1:
        [[fallthrough]];
      case Spectral::Basis::FiniteDifference: {
        I1(d);
        break;
      }
      default:
        ERROR("Missing basis case for logical_coordinates");
    }
  }
}

template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::ElementLogical> logical_coordinates(
    const Mesh<VolumeDim>& mesh) {
  tnsr::I<DataVector, VolumeDim, Frame::ElementLogical> result{};
  logical_coordinates(make_not_null(&result), mesh);
  return result;
}

// Explicit instantiations
template tnsr::I<DataVector, 1, Frame::ElementLogical> logical_coordinates(
    const Mesh<1>&);
template tnsr::I<DataVector, 2, Frame::ElementLogical> logical_coordinates(
    const Mesh<2>&);
template tnsr::I<DataVector, 3, Frame::ElementLogical> logical_coordinates(
    const Mesh<3>&);
template void logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, 1, Frame::ElementLogical>*>,
    const Mesh<1>&);
template void logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, 2, Frame::ElementLogical>*>,
    const Mesh<2>&);
template void logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::ElementLogical>*>,
    const Mesh<3>&);
