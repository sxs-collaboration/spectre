// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Projection.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral {

std::ostream& operator<<(std::ostream& os, MortarSize mortar_size) noexcept {
  switch (mortar_size) {
    case MortarSize::Full:
      return os << "Full";
    case MortarSize::UpperHalf:
      return os << "UpperHalf";
    case MortarSize::LowerHalf:
      return os << "LowerHalf";
    default:
      ERROR("Invalid MortarSize");
  }
}

const Matrix& projection_matrix_mortar_to_element(
    const MortarSize size, const Mesh<1>& element_mesh,
    const Mesh<1>& mortar_mesh) noexcept {
  ASSERT(element_mesh.basis(0) == Basis::Legendre and
             mortar_mesh.basis(0) == Basis::Legendre,
         "Projections only implemented on Legendre basis");
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
      const static auto cache = make_static_cache<
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>,
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>>(
          [](const Quadrature quadrature_element, const size_t extents_element,
             const Quadrature quadrature_mortar,
             const size_t extents_mortar) noexcept {
            if (extents_element > extents_mortar) {
              return Matrix{};
            }
            const Mesh<1> mesh_element(extents_element, Basis::Legendre,
                                       quadrature_element);
            const Mesh<1> mesh_mortar(extents_mortar, Basis::Legendre,
                                      quadrature_mortar);

            // The projection in spectral space is just a truncation
            // of the modes.
            const auto& spectral_to_grid_element =
                modal_to_nodal_matrix(mesh_element);
            const auto& grid_to_spectral_mortar =
                nodal_to_modal_matrix(mesh_mortar);
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
      return cache(element_mesh.quadrature(0), element_mesh.extents(0),
                   mortar_mesh.quadrature(0), mortar_mesh.extents(0));
    }

    case MortarSize::UpperHalf: {
      const static auto cache = make_static_cache<
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>,
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>>([](
          const Quadrature quadrature_element, const size_t extents_element,
          const Quadrature quadrature_mortar,
          const size_t extents_mortar) noexcept {
        if (extents_element > extents_mortar) {
          return Matrix{};
        }
        const Mesh<1> mesh_element(extents_element, Basis::Legendre,
                                   quadrature_element);
        const Mesh<1> mesh_mortar(extents_mortar, Basis::Legendre,
                                  quadrature_mortar);

        // The transformation from the small interval to the large
        // interval in spectral space.  This is a rearranged
        // version of the equation given in the header.  This form
        // was easier to code.
        const auto spectral_transformation =
            [](const size_t large_index, const size_t small_index) noexcept {
          ASSERT(large_index >= small_index,
                 "Above-diagonal entries are zero.  Don't use them.");
          double result = 1.;
          for (size_t i = (large_index - small_index) / 2; i > 0; --i) {
            result = 1 - result *
                             static_cast<double>(
                                 (large_index + small_index + 3 - 2 * i) *
                                 (large_index + small_index + 2 - 2 * i) *
                                 (large_index - small_index + 2 - 2 * i) *
                                 (large_index - small_index + 1 - 2 * i)) /
                             static_cast<double>(2 * i *
                                                 (2 * large_index + 1 - 2 * i) *
                                                 (large_index + 2 - 2 * i) *
                                                 (large_index + 1 - 2 * i));
          }

          for (size_t i = 1; i <= large_index - small_index; ++i) {
            result *=
                1. + static_cast<double>(large_index + small_index + 1) / i;
          }
          result /= pow(2., static_cast<int>(large_index) + 1);
          return result;
        };

        const auto& spectral_to_grid_element =
            modal_to_nodal_matrix(mesh_element);
        const auto& grid_to_spectral_mortar =
            nodal_to_modal_matrix(mesh_mortar);

        Matrix temp(extents_element, extents_element, 0.);
        for (size_t j = 0; j < extents_element; ++j) {
          for (size_t k = j; k < extents_element; ++k) {
            const double transformation_entry = spectral_transformation(k, j);
            for (size_t i = 0; i < extents_element; ++i) {
              temp(i, j) +=
                  spectral_to_grid_element(i, k) * transformation_entry;
            }
          }
        }

        Matrix projection(extents_element, extents_mortar, 0.);
        for (size_t i = 0; i < extents_element; ++i) {
          for (size_t j = 0; j < extents_mortar; ++j) {
            for (size_t k = 0; k < extents_element; ++k) {
              projection(i, j) += temp(i, k) * grid_to_spectral_mortar(k, j);
            }
          }
        }

        return projection;
      });
      return cache(element_mesh.quadrature(0), element_mesh.extents(0),
                   mortar_mesh.quadrature(0), mortar_mesh.extents(0));
    }

    case MortarSize::LowerHalf: {
      const static auto cache = make_static_cache<
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>,
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>>([](
          const Quadrature quadrature_element, const size_t extents_element,
          const Quadrature quadrature_mortar,
          const size_t extents_mortar) noexcept {
        if (extents_element > extents_mortar) {
          return Matrix{};
        }
        const Mesh<1> mesh_element(extents_element, Basis::Legendre,
                                   quadrature_element);
        const Mesh<1> mesh_mortar(extents_mortar, Basis::Legendre,
                                  quadrature_mortar);

        // The lower-half matrices are generated from the upper-half
        // matrices using symmetry.
        const auto& projection_upper_half = projection_matrix_mortar_to_element(
            MortarSize::UpperHalf, mesh_element, mesh_mortar);

        Matrix projection_lower_half(extents_element, extents_mortar);
        for (size_t i = 0; i < extents_element; ++i) {
          for (size_t j = 0; j < extents_mortar; ++j) {
            projection_lower_half(i, j) = projection_upper_half(
                extents_element - i - 1, extents_mortar - j - 1);
          }
        }

        return projection_lower_half;
      });
      return cache(element_mesh.quadrature(0), element_mesh.extents(0),
                   mortar_mesh.quadrature(0), mortar_mesh.extents(0));
    }

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
  ASSERT(
      mortar_mesh.extents(0) <= maximum_number_of_points<Basis::Legendre> and
          element_mesh.extents(0) <= maximum_number_of_points<Basis::Legendre>,
      "Mesh has more points than supported by its quadrature. Has "
          << mortar_mesh.extents(0) << " and max allowed is "
          << (maximum_number_of_points<Basis::Legendre>)
          << " for the mortar mesh, while for the element mesh has "
          << element_mesh.extents(0) << " points.");
  ASSERT(mortar_mesh.extents(0) >= element_mesh.extents(0),
         "Requested projection matrix to mortar with fewer points ("
             << mortar_mesh.extents(0) << ") than the element ("
             << element_mesh.extents(0) << ")");

  // Element-to-mortar projections are always interpolations.
  const auto make_interpolators = [](auto interval_transform) noexcept {
    return [interval_transform = std::move(interval_transform)](
        const Quadrature quadrature_mortar, const size_t extents_mortar,
        const Quadrature quadrature_element,
        const size_t extents_element) noexcept {
      if (extents_mortar < extents_element) {
        return Matrix{};
      }
      const Mesh<1> mesh_element(extents_element, Basis::Legendre,
                                 quadrature_element);
      const Mesh<1> mesh_mortar(extents_mortar, Basis::Legendre,
                                quadrature_mortar);
      return interpolation_matrix(
          mesh_element, interval_transform(collocation_points(mesh_mortar)));
    };
  };

  switch (size) {
    case MortarSize::Full: {
      const static auto cache = make_static_cache<
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>,
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>>(
          make_interpolators([](const DataVector& x) noexcept { return x; }));
      return cache(mortar_mesh.quadrature(0), mortar_mesh.extents(0),
                   element_mesh.quadrature(0), element_mesh.extents(0));
    }

    case MortarSize::UpperHalf: {
      const static auto cache = make_static_cache<
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>,
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>>(
          make_interpolators([](const DataVector& x) noexcept {
            return DataVector(0.5 * (x + 1.));
          }));
      return cache(mortar_mesh.quadrature(0), mortar_mesh.extents(0),
                   element_mesh.quadrature(0), element_mesh.extents(0));
    }

    case MortarSize::LowerHalf: {
      const static auto cache = make_static_cache<
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>,
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, maximum_number_of_points<Basis::Legendre> + 1>>(
          make_interpolators([](const DataVector& x) noexcept {
            return DataVector(0.5 * (x - 1.));
          }));
      return cache(mortar_mesh.quadrature(0), mortar_mesh.extents(0),
                   element_mesh.quadrature(0), element_mesh.extents(0));
    }

    default:
      ERROR("Invalid MortarSize");
  }
}

}  // namespace Spectral
