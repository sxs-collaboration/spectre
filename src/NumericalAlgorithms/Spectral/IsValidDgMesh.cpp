// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/IsValidDgMesh.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Limits.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
template <size_t Dim>
bool is_valid_dg_mesh(const Mesh<Dim>& mesh) {
  const auto& extents = mesh.extents();
  const auto& bases = mesh.basis();
  const auto& quadratures = mesh.quadrature();

  for (size_t d = 0; d < Dim; ++d) {
    const auto extent = gsl::at(extents.indices(), d);
    const auto basis = gsl::at(bases, d);
    const auto quadrature = gsl::at(quadratures, d);
    if (extent < limits::min(basis, quadrature)) {
      return false;
    }
    if (extent > limits::max(basis, quadrature)) {
      return false;
    }

    switch (basis) {
      case Basis::Uninitialized:
        ERROR("Uninitialized basis");
      case Basis::Legendre:
        [[fallthrough]];
      case Basis::Chebyshev: {
        if (quadrature == Quadrature::GaussLobatto or
            quadrature == Quadrature::Gauss) {
          break;
        } else {
          return false;
        }
      }
      case Basis::FiniteDifference:
        return false;
      case Basis::SphericalHarmonic: {
        if constexpr (Dim == 1) {
          return false;
        } else if constexpr (Dim == 2) {
          if (d == 0) {
            if (bases[1] == Basis::SphericalHarmonic and
                quadrature == Quadrature::Gauss and
                quadratures[1] == Quadrature::Equiangular and
                extents[1] == 2 * extents[0] - 1) {
              break;
            } else {
              return false;
            }
          } else {
            if (bases[0] == Basis::SphericalHarmonic) {
              break;
            } else {
              return false;
            }
          }
        } else {
          ASSERT(Dim == 3, "Invalid Dim " << Dim);
          if (d == 0) {
            return false;
          } else if (d == 1) {
            if (bases[2] == Basis::SphericalHarmonic and
                quadrature == Quadrature::Gauss and
                quadratures[2] == Quadrature::Equiangular and
                extents[2] == 2 * extents[1] - 1 and
                (bases[0] == Basis::Legendre or bases[0] == Basis::Chebyshev)) {
              break;
            } else {
              return false;
            }
          } else {
            if (bases[1] == Basis::SphericalHarmonic) {
              break;
            } else {
              return false;
            }
          }
        }
      }
      case Basis::Fourier: {
        if (quadrature == Quadrature::Equiangular and extent % 2 == 1) {
          break;
        } else {
          return false;
        }
      }
      case Basis::ZernikeB1: {
        if constexpr (Dim == 3) {
          if (d == 0 and quadrature == Quadrature::GaussRadauUpper and
              (bases[1] == Basis::Legendre or bases[1] == Basis::Cartoon or
               bases[1] == Basis::Chebyshev) and
              bases[2] == Basis::Cartoon) {
            break;
          } else {
            return false;
          }
        } else {
          return false;
        }
      }
      case Basis::ZernikeB2: {
        if constexpr (Dim == 1) {
          return false;
        } else if constexpr (Dim == 2) {
          if (d == 0) {
            if (bases[1] == Basis::ZernikeB2 and
                quadrature == Quadrature::GaussRadauUpper and
                quadratures[1] == Quadrature::Equiangular and
                extents[1] == 4 * extents[0] - 3) {
              break;
            } else {
              return false;
            }
          } else {
            if (bases[0] == Basis::ZernikeB2) {
              break;
            } else {
              return false;
            }
          }
        } else {
          ASSERT(Dim == 3, "Invalid Dim " << Dim);
          if (d == 2) {
            return false;
          } else if (d == 0) {
            if (bases[1] == Basis::ZernikeB2 and
                quadrature == Quadrature::GaussRadauUpper and
                quadratures[1] == Quadrature::Equiangular and
                extents[1] == 4 * extents[0] - 3 and
                (bases[2] == Basis::Legendre or bases[2] == Basis::Chebyshev)) {
              break;
            } else {
              return false;
            }
          } else {
            if (bases[0] == Basis::ZernikeB2) {
              break;
            } else {
              return false;
            }
          }
        }
      }
      // NOLINTNEXTLINE(bugprone-branch-clone)
      case Basis::ZernikeB3: {
        if constexpr (Dim == 3) {
          if (d == 0) {
            if (bases[1] == Basis::ZernikeB3 and
                bases[2] == Basis::ZernikeB3 and
                quadrature == Quadrature::GaussRadauUpper and
                quadratures[1] == Quadrature::Gauss and
                quadratures[2] == Quadrature::Equiangular and
                extents[2] == 2 * extents[1] - 1 and
                extents[1] == 2 * extents[0] - 1) {
              break;
            } else {
              return false;
            }
          } else {
            if (bases[0] == Basis::ZernikeB3) {
              break;
            } else {
              return false;
            }
          }
        } else {
          return false;
        }
      }
      case Basis::Cartoon: {
        if constexpr (Dim == 3) {
          if (d == 1) {
            if ((bases[0] == Basis::Legendre or bases[0] == Basis::ZernikeB1 or
                 bases[0] == Basis::Chebyshev) and
                bases[2] == Basis::Cartoon and
                quadrature == Quadrature::SphericalSymmetry and
                quadratures[2] == Quadrature::SphericalSymmetry) {
              break;
            } else {
              return false;
            }
          } else if (d == 2) {
            if (bases[1] == Basis::Cartoon or
                ((bases[0] == Basis::Legendre or bases[0] == Basis::ZernikeB1 or
                  bases[0] == Basis::Chebyshev) and
                 (bases[1] == Basis::Legendre or bases[1] == Basis::ZernikeB1 or
                  bases[1] == Basis::Chebyshev) and
                 quadrature == Quadrature::AxialSymmetry)) {
              break;
            } else {
              return false;
            }
          } else {
            return false;
          }
        } else {
          return false;
        }
      }
      default:
        ERROR("Invalid basis");
    }
  }
  return true;
}

template bool is_valid_dg_mesh(const Mesh<1>& mesh);
template bool is_valid_dg_mesh(const Mesh<2>& mesh);
template bool is_valid_dg_mesh(const Mesh<3>& mesh);
}  // namespace Spectral
