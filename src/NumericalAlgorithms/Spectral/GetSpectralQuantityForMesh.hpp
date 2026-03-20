// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Spectral::detail {
template <typename F>
decltype(auto) get_spectral_quantity_for_mesh(F&& f, const Mesh<1>& mesh) {
  const auto num_points = mesh.extents(0);
  // Switch on runtime values of basis and quadrature to select
  // corresponding template specialization. For basis functions spanning
  // multiple dimensions we can generalize this function to take a
  // higher-dimensional Mesh.
  switch (mesh.basis(0)) {
    case Basis::Legendre:
      switch (mesh.quadrature(0)) {
        case Quadrature::Gauss:
          return f(std::integral_constant<Basis, Basis::Legendre>{},
                   std::integral_constant<Quadrature, Quadrature::Gauss>{},
                   num_points);
        case Quadrature::GaussLobatto:
          return f(
              std::integral_constant<Basis, Basis::Legendre>{},
              std::integral_constant<Quadrature, Quadrature::GaussLobatto>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    case Basis::Chebyshev:
      switch (mesh.quadrature(0)) {
        case Quadrature::Gauss:
          return f(std::integral_constant<Basis, Basis::Chebyshev>{},
                   std::integral_constant<Quadrature, Quadrature::Gauss>{},
                   num_points);
        case Quadrature::GaussLobatto:
          return f(
              std::integral_constant<Basis, Basis::Chebyshev>{},
              std::integral_constant<Quadrature, Quadrature::GaussLobatto>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    case Basis::Cartoon:
      switch (mesh.quadrature(0)) {
        case Quadrature::AxialSymmetry:
          return f(
              std::integral_constant<Basis, Basis::Cartoon>{},
              std::integral_constant<Quadrature, Quadrature::AxialSymmetry>{},
              num_points);
        case Quadrature::SphericalSymmetry:
          return f(std::integral_constant<Basis, Basis::Cartoon>{},
                   std::integral_constant<Quadrature,
                                          Quadrature::SphericalSymmetry>{},
                   num_points);
        default:
          ERROR(
              "Only Axial and Spherical Symmetry quadratures are allowed for "
              "a Cartoon basis.");
      }
    case Basis::Fourier:
      switch (mesh.quadrature(0)) {
        case Quadrature::Equiangular:
          return f(
              std::integral_constant<Basis, Basis::Fourier>{},
              std::integral_constant<Quadrature, Quadrature::Equiangular>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    case Basis::ZernikeB1:
      switch (mesh.quadrature(0)) {
        case Quadrature::GaussRadauUpper:
          return f(
              std::integral_constant<Basis, Basis::ZernikeB1>{},
              std::integral_constant<Quadrature, Quadrature::GaussRadauUpper>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    case Basis::ZernikeB2:
      switch (mesh.quadrature(0)) {
        case Quadrature::GaussRadauUpper:
          return f(
              std::integral_constant<Basis, Basis::ZernikeB2>{},
              std::integral_constant<Quadrature, Quadrature::GaussRadauUpper>{},
              num_points);
        case Quadrature::Equiangular:
          // While ZernikeB2 with Equiangular is treated differently in
          // operators (e.g. partial derivatives, interpolation), in terms of
          // its spectral quantities it is Fourier with Equiangular
          return f(
              std::integral_constant<Basis, Basis::Fourier>{},
              std::integral_constant<Quadrature, Quadrature::Equiangular>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    case Basis::ZernikeB3:
      switch (mesh.quadrature(0)) {
        case Quadrature::GaussRadauUpper:
          return f(
              std::integral_constant<Basis, Basis::ZernikeB3>{},
              std::integral_constant<Quadrature, Quadrature::GaussRadauUpper>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    case Basis::FiniteDifference:
      switch (mesh.quadrature(0)) {
        case Quadrature::CellCentered:
          return f(
              std::integral_constant<Basis, Basis::FiniteDifference>{},
              std::integral_constant<Quadrature, Quadrature::CellCentered>{},
              num_points);
        case Quadrature::FaceCentered:
          return f(
              std::integral_constant<Basis, Basis::FiniteDifference>{},
              std::integral_constant<Quadrature, Quadrature::FaceCentered>{},
              num_points);
        default:
          ERROR(
              "Only CellCentered and FaceCentered are supported for finite "
              "difference quadrature.");
      }
    case Basis::SphericalHarmonic:
      ERROR(
          "Basis::SphericalHarmonic is a two-dimensional basis and is not "
          "supported for this function.  If you want the collocation points, "
          "use the function logical_coordinates.");
    default:
      ERROR("Missing basis case for spectral quantity. The missing basis is: "
            << mesh.basis(0));
  }
}

template <typename F>
decltype(auto) get_two_indexed_spectral_quantity_for_mesh(F&& f,
                                                          const Mesh<1>& mesh,
                                                          const size_t m,
                                                          const size_t N) {
  const auto num_points = mesh.extents(0);
  switch (mesh.basis(0)) {
    case Basis::ZernikeB1:
      switch (mesh.quadrature(0)) {
        case Quadrature::GaussRadauUpper:
          return f(
              std::integral_constant<Basis, Basis::ZernikeB1>{},
              std::integral_constant<Quadrature, Quadrature::GaussRadauUpper>{},
              num_points, m, N);
        default:
          ERROR("Missing quadrature case for two-indexed spectral quantity");
      }
    case Basis::ZernikeB2:
      switch (mesh.quadrature(0)) {
        case Quadrature::GaussRadauUpper:
          return f(
              std::integral_constant<Basis, Basis::ZernikeB2>{},
              std::integral_constant<Quadrature, Quadrature::GaussRadauUpper>{},
              num_points, m, N);
        default:
          ERROR("Missing quadrature case for two-indexed spectral quantity");
      }
    case Basis::ZernikeB3:
      switch (mesh.quadrature(0)) {
        case Quadrature::GaussRadauUpper:
          return f(
              std::integral_constant<Basis, Basis::ZernikeB3>{},
              std::integral_constant<Quadrature, Quadrature::GaussRadauUpper>{},
              num_points, m, N);
        default:
          ERROR("Missing quadrature case for two-indexed spectral quantity");
      }
    default:
      ERROR(
          "Missing basis case for two-indexed spectral quantity. The "
          "missing basis is: "
          << mesh.basis(0));
  }
}

}  // namespace Spectral::detail
