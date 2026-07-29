// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/IsValidDgMesh.hpp"

#include <array>
#include <cstddef>

#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/IsValidDgMesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
template <size_t Dim>
bool is_valid_dg_mesh(const Mesh<Dim>& mesh, const Element<Dim>& element) {
  // only need to verify consistency of topology and basis as quadrature will
  // be checked in is_valid_dg_mesh(mesh)
  const auto& topologies = element.topologies();
  for (size_t d = 0; d < Dim; ++d) {
    const auto topology = gsl::at(topologies, d);
    const auto basis = mesh.basis(d);
    const auto& segment_id = element.id().segment_id(d);
    const auto refinement_level = segment_id.refinement_level();
    const auto segment_id_index = segment_id.index();
    switch (topology) {
      case Topology::I1: {
        if (basis == Spectral::Basis::Legendre or
            basis == Spectral::Basis::Chebyshev) {
          break;
        } else {
          return false;
        }
      }
      case Topology::S1: {
        if (basis == Spectral::Basis::Fourier and refinement_level == 0_st) {
          break;
        } else {
          return false;
        }
      }
      case Topology::S2Colatitude:
        [[fallthrough]];
      case Topology::S2Longitude: {
        if (basis == Spectral::Basis::SphericalHarmonic and
            refinement_level == 0_st) {
          break;
        } else {
          return false;
        }
      }
      case Topology::B1Radial: {
        if (basis == Spectral::Basis::ZernikeB1 and segment_id_index == 0_st) {
          break;
        } else {
          return false;
        }
      }
      case Topology::B2Radial:
        if (basis == Spectral::Basis::ZernikeB2 and segment_id_index == 0_st) {
          break;
        } else {
          return false;
        }
      case Topology::B2Angular: {
        if (basis == Spectral::Basis::ZernikeB2 and refinement_level == 0_st) {
          break;
        } else {
          return false;
        }
      }
      case Topology::B3Radial:
        if (basis == Spectral::Basis::ZernikeB3 and segment_id_index == 0_st) {
          break;
        } else {
          return false;
        }
      case Topology::B3Colatitude:
        [[fallthrough]];
      case Topology::B3Longitude: {
        if (basis == Spectral::Basis::ZernikeB3 and refinement_level == 0_st) {
          break;
        } else {
          return false;
        }
      }
      case Topology::CartoonSphere:
        [[fallthrough]];
      case Topology::CartoonCylinder: {
        if (basis == Spectral::Basis::Cartoon and refinement_level == 0_st) {
          break;
        } else {
          return false;
        }
      }
      default:
        ERROR("Invalid basis");
    }
  }
  return Spectral::is_valid_dg_mesh(mesh);
}

template bool is_valid_dg_mesh(const Mesh<1>& mesh, const Element<1>& element);
template bool is_valid_dg_mesh(const Mesh<2>& mesh, const Element<2>& element);
template bool is_valid_dg_mesh(const Mesh<3>& mesh, const Element<3>& element);
}  // namespace domain
