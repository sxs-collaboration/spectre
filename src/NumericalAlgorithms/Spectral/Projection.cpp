// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Projection.hpp"

#include <algorithm>
#include <initializer_list>
#include <ostream>

#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral {

namespace {
constexpr auto supported_quadratures = {Quadrature::Gauss,
                                        Quadrature::GaussLobatto};

size_t encode_quadrature(const Quadrature quadrature) noexcept {
  return static_cast<size_t>(std::find(supported_quadratures.begin(),
                                       supported_quadratures.end(),
                                       quadrature) -
                             supported_quadratures.begin());
}

Quadrature decode_quadrature(const size_t encoded) noexcept {
  return *(supported_quadratures.begin() + encoded);
}

void check_quadrature_supported(const Quadrature quadrature) noexcept {
  ASSERT(std::find(supported_quadratures.begin(), supported_quadratures.end(),
                   quadrature) != supported_quadratures.end(),
         "Unsupported quadrature: " << quadrature);
}
}  // namespace

const Matrix& projection_matrix_mortar_to_element(
    const MortarSize size, const Mesh<1>& element_mesh,
    const Mesh<1>& mortar_mesh) noexcept {
  ASSERT(element_mesh.basis(0) == Basis::Legendre and
             mortar_mesh.basis(0) == Basis::Legendre,
         "Projections only implemented on Legendre basis");
  check_quadrature_supported(element_mesh.quadrature(0));
  check_quadrature_supported(mortar_mesh.quadrature(0));
  ASSERT(
      element_mesh.extents(0) <= maximum_number_of_points<Basis::Legendre> and
          mortar_mesh.extents(0) <= maximum_number_of_points<Basis::Legendre>,
      "Mesh has more points than supported by its quadrature.");
  ASSERT(element_mesh.extents(0) <= mortar_mesh.extents(0),
         "Requested projection matrix from mortar with fewer points ("
         << mortar_mesh.extents(0) << ") than the element ("
         << element_mesh.extents(0) << ")");

  switch (size) {
    case MortarSize::Full: {
      const static StaticCache<
          Matrix, CacheRange<0, supported_quadratures.size()>,
          CacheRange<2, maximum_number_of_points<Basis::Legendre> + 1>,
          CacheRange<0, supported_quadratures.size()>,
          CacheRange<2, maximum_number_of_points<Basis::Legendre> + 1>>
          cache([](const size_t encoded_quadrature_element,
                   const size_t extents_element,
                   const size_t encoded_quadrature_mortar,
                   const size_t extents_mortar) noexcept {
            if (extents_element > extents_mortar) {
              return Matrix{};
            }
            const Mesh<1> mesh_element(
                extents_element, Basis::Legendre,
                decode_quadrature(encoded_quadrature_element));
            const Mesh<1> mesh_mortar(
                extents_mortar, Basis::Legendre,
                decode_quadrature(encoded_quadrature_mortar));

            // The projection in spectral space is just a truncation
            // of the modes.
            const auto& spectral_to_grid_element =
                spectral_to_grid_points_matrix(mesh_element);
            const auto& grid_to_spectral_mortar =
                grid_points_to_spectral_matrix(mesh_mortar);
            Matrix projection(extents_element, extents_mortar, 0.);
            for (size_t i = 0; i < extents_element; ++i) {
              for (size_t j = 0; j < extents_mortar; ++j) {
                for (size_t k = 0; k < extents_element; ++k) {
                  projection(i, j) += spectral_to_grid_element(i, k) *
                                      grid_to_spectral_mortar(k, j);
                }
              }
            }

            return projection;
          });
      return cache(encode_quadrature(element_mesh.quadrature(0)),
                   element_mesh.extents(0),
                   encode_quadrature(mortar_mesh.quadrature(0)),
                   mortar_mesh.extents(0));
    }

    case MortarSize::UpperHalf:
    case MortarSize::LowerHalf:
      ERROR("h-refinement not yet implemented");

    default:
      ERROR("Invalid MortarSize");
  }
}

const Matrix& projection_matrix_element_to_mortar(
    const MortarSize size, const Mesh<1>& mortar_mesh,
    const Mesh<1>& element_mesh) noexcept {
  ASSERT(mortar_mesh.basis(0) == Basis::Legendre and
             element_mesh.basis(0) == Basis::Legendre,
         "Projections only implemented on Legendre basis");
  check_quadrature_supported(mortar_mesh.quadrature(0));
  check_quadrature_supported(element_mesh.quadrature(0));
  ASSERT(
      mortar_mesh.extents(0) <= maximum_number_of_points<Basis::Legendre> and
          element_mesh.extents(0) <= maximum_number_of_points<Basis::Legendre>,
      "Mesh has more points than supported by its quadrature.");
  ASSERT(mortar_mesh.extents(0) >= element_mesh.extents(0),
         "Requested projection matrix to mortar with fewer points ("
         << mortar_mesh.extents(0) << ") than the element ("
         << element_mesh.extents(0) << ")");

  // Element-to-mortar projections are always interpolations.

  switch (size) {
    case MortarSize::Full: {
      const static StaticCache<
          Matrix, CacheRange<0, supported_quadratures.size()>,
          CacheRange<2, maximum_number_of_points<Basis::Legendre> + 1>,
          CacheRange<0, supported_quadratures.size()>,
          CacheRange<2, maximum_number_of_points<Basis::Legendre> + 1>>
          cache([](const size_t encoded_quadrature_mortar,
                   const size_t extents_mortar,
                   const size_t encoded_quadrature_element,
                   const size_t extents_element) noexcept {
            if (extents_mortar < extents_element) {
              return Matrix{};
            }
            const Mesh<1> mesh_element(
                extents_element, Basis::Legendre,
                decode_quadrature(encoded_quadrature_element));
            const Mesh<1> mesh_mortar(
                extents_mortar, Basis::Legendre,
                decode_quadrature(encoded_quadrature_mortar));

            return interpolation_matrix(mesh_element,
                                        collocation_points(mesh_mortar));
          });
      return cache(encode_quadrature(mortar_mesh.quadrature(0)),
                   mortar_mesh.extents(0),
                   encode_quadrature(element_mesh.quadrature(0)),
                   element_mesh.extents(0));
    }

    case MortarSize::UpperHalf:
    case MortarSize::LowerHalf:
      ERROR("h-refinement not yet implemented");

    default:
      ERROR("Invalid MortarSize");
  }
}

}  // namespace Spectral
