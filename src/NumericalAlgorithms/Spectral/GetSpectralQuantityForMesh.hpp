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
}  // namespace Spectral::detail
