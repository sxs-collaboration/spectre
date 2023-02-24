// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral {

std::ostream& operator<<(std::ostream& os, ChildSize mortar_size) {
  switch (mortar_size) {
    case ChildSize::Full:
      return os << "Full";
    case ChildSize::UpperHalf:
      return os << "UpperHalf";
    case ChildSize::LowerHalf:
      return os << "LowerHalf";
    default:
      ERROR(
          "Invalid ChildSize. Expected one of: 'Full', 'UpperHalf', "
          "'LowerHalf'");
  }
}

template <size_t Dim>
bool needs_projection(const Mesh<Dim>& mesh1, const Mesh<Dim>& mesh2,
                      const std::array<ChildSize, Dim>& child_sizes) {
  return mesh1 != mesh2 or
         alg::any_of(child_sizes, [](const Spectral::MortarSize child_size) {
           return child_size != Spectral::ChildSize::Full;
         });
}

const Matrix& projection_matrix_child_to_parent(const Mesh<1>& child_mesh,
                                                const Mesh<1>& parent_mesh,
                                                const ChildSize size,
                                                const bool operand_is_massive) {
  constexpr size_t max_points = maximum_number_of_points<Basis::Legendre>;
  ASSERT(parent_mesh.basis(0) == Basis::Legendre and
             child_mesh.basis(0) == Basis::Legendre,
         "Projections only implemented on Legendre basis");
  ASSERT(parent_mesh.extents(0) <= max_points and
             child_mesh.extents(0) <= max_points,
         "Mesh has more points than supported by its quadrature.");
  ASSERT(parent_mesh.extents(0) <= child_mesh.extents(0),
         "Requested projection matrix from child with fewer points ("
             << child_mesh.extents(0) << ") than the parent ("
             << parent_mesh.extents(0) << ")");

  if (operand_is_massive) {
    // The restriction operator for massive quantities is just the interpolation
    // transpose
    const static auto cache = make_static_cache<
        CacheEnumeration<Quadrature, Quadrature::Gauss,
                         Quadrature::GaussLobatto>,
        CacheRange<2_st, max_points + 1>,
        CacheEnumeration<Quadrature, Quadrature::Gauss,
                         Quadrature::GaussLobatto>,
        CacheRange<2_st, max_points + 1>,
        CacheEnumeration<ChildSize, ChildSize::Full, ChildSize::UpperHalf,
                         ChildSize::LowerHalf>>(
        [](const Quadrature child_quadrature, const size_t child_extent,
           const Quadrature parent_quadrature, const size_t parent_extent,
           const ChildSize local_child_size) -> Matrix {
          const auto& prolongation_operator = projection_matrix_parent_to_child(
              {parent_extent, Spectral::Basis::Legendre, parent_quadrature},
              {child_extent, Spectral::Basis::Legendre, child_quadrature},
              local_child_size);
          return blaze::trans(prolongation_operator);
        });
    return cache(child_mesh.quadrature(0), child_mesh.extents(0),
                 parent_mesh.quadrature(0), parent_mesh.extents(0), size);
  }

  switch (size) {
    case ChildSize::Full: {
      const static auto cache =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>>(
              [](const Quadrature quadrature_element,
                 const size_t extents_element,
                 const Quadrature quadrature_mortar,
                 const size_t extents_mortar) {
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
      return cache(parent_mesh.quadrature(0), parent_mesh.extents(0),
                   child_mesh.quadrature(0), child_mesh.extents(0));
    }

    case ChildSize::UpperHalf: {
      const static auto cache = make_static_cache<
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, max_points + 1>,
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st,
                     max_points + 1>>([](const Quadrature quadrature_element,
                                         const size_t extents_element,
                                         const Quadrature quadrature_mortar,
                                         const size_t extents_mortar) {
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
        const auto spectral_transformation = [](const size_t large_index,
                                                const size_t small_index) {
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
      return cache(parent_mesh.quadrature(0), parent_mesh.extents(0),
                   child_mesh.quadrature(0), child_mesh.extents(0));
    }

    case ChildSize::LowerHalf: {
      const static auto cache =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>>(
              [](const Quadrature quadrature_element,
                 const size_t extents_element,
                 const Quadrature quadrature_mortar,
                 const size_t extents_mortar) {
                if (extents_element > extents_mortar) {
                  return Matrix{};
                }
                const Mesh<1> mesh_element(extents_element, Basis::Legendre,
                                           quadrature_element);
                const Mesh<1> mesh_mortar(extents_mortar, Basis::Legendre,
                                          quadrature_mortar);

                // The lower-half matrices are generated from the upper-half
                // matrices using symmetry.
                const auto& projection_upper_half =
                    projection_matrix_child_to_parent(mesh_mortar, mesh_element,
                                                      ChildSize::UpperHalf);

                Matrix projection_lower_half(extents_element, extents_mortar);
                for (size_t i = 0; i < extents_element; ++i) {
                  for (size_t j = 0; j < extents_mortar; ++j) {
                    projection_lower_half(i, j) = projection_upper_half(
                        extents_element - i - 1, extents_mortar - j - 1);
                  }
                }

                return projection_lower_half;
              });
      return cache(parent_mesh.quadrature(0), parent_mesh.extents(0),
                   child_mesh.quadrature(0), child_mesh.extents(0));
    }

    default:
      ERROR("Invalid ChildSize");
  }
}

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim>
projection_matrix_child_to_parent(const Mesh<Dim>& child_mesh,
                                  const Mesh<Dim>& parent_mesh,
                                  const std::array<ChildSize, Dim>& child_sizes,
                                  const bool operand_is_massive) {
  static const Matrix identity{};
  auto projection_matrix = make_array<Dim>(std::cref(identity));
  const auto child_mesh_slices = child_mesh.slices();
  const auto parent_mesh_slices = parent_mesh.slices();
  for (size_t d = 0; d < Dim; ++d) {
    const auto child_mesh_slice = gsl::at(child_mesh_slices, d);
    const auto parent_mesh_slice = gsl::at(parent_mesh_slices, d);
    const auto child_size = gsl::at(child_sizes, d);
    if (child_size == ChildSize::Full and
        child_mesh_slice == parent_mesh_slice) {
      // No projection necessary, keep matrix the identity in this dimension
      continue;
    }
    gsl::at(projection_matrix, d) = projection_matrix_child_to_parent(
        child_mesh_slice, parent_mesh_slice, child_size, operand_is_massive);
  }
  return projection_matrix;
}

const Matrix& projection_matrix_parent_to_child(const Mesh<1>& parent_mesh,
                                                const Mesh<1>& child_mesh,
                                                const ChildSize size) {
  constexpr size_t max_points = maximum_number_of_points<Basis::Legendre>;
  ASSERT(child_mesh.basis(0) == Basis::Legendre and
             parent_mesh.basis(0) == Basis::Legendre,
         "Projections only implemented on Legendre basis");
  ASSERT(child_mesh.extents(0) <= max_points and
             parent_mesh.extents(0) <= max_points,
         "Mesh has more points than supported by its quadrature. Has "
             << child_mesh.extents(0) << " and max allowed is " << (max_points)
             << " for the mortar mesh, while for the element mesh has "
             << parent_mesh.extents(0) << " points.");
  ASSERT(child_mesh.extents(0) >= parent_mesh.extents(0),
         "Requested projection matrix to mortar with fewer points ("
             << child_mesh.extents(0) << ") than the element ("
             << parent_mesh.extents(0) << ")");

  // Element-to-mortar projections are always interpolations.
  const auto make_interpolators = [](auto interval_transform) {
    return [interval_transform = std::move(interval_transform)](
               const Quadrature quadrature_mortar, const size_t extents_mortar,
               const Quadrature quadrature_element,
               const size_t extents_element) {
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
    case ChildSize::Full: {
      const static auto cache =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>>(
              make_interpolators([](const DataVector& x) { return x; }));
      return cache(child_mesh.quadrature(0), child_mesh.extents(0),
                   parent_mesh.quadrature(0), parent_mesh.extents(0));
    }

    case ChildSize::UpperHalf: {
      const static auto cache =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>>(
              make_interpolators([](const DataVector& x) {
                return DataVector(0.5 * (x + 1.));
              }));
      return cache(child_mesh.quadrature(0), child_mesh.extents(0),
                   parent_mesh.quadrature(0), parent_mesh.extents(0));
    }

    case ChildSize::LowerHalf: {
      const static auto cache =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points + 1>>(
              make_interpolators([](const DataVector& x) {
                return DataVector(0.5 * (x - 1.));
              }));
      return cache(child_mesh.quadrature(0), child_mesh.extents(0),
                   parent_mesh.quadrature(0), parent_mesh.extents(0));
    }

    default:
      ERROR("Invalid ChildSize");
  }
}

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim>
projection_matrix_parent_to_child(
    const Mesh<Dim>& parent_mesh, const Mesh<Dim>& child_mesh,
    const std::array<ChildSize, Dim>& child_sizes) {
  static const Matrix identity{};
  auto projection_matrix = make_array<Dim>(std::cref(identity));
  const auto child_mesh_slices = child_mesh.slices();
  const auto parent_mesh_slices = parent_mesh.slices();
  for (size_t d = 0; d < Dim; ++d) {
    const auto child_mesh_slice = gsl::at(child_mesh_slices, d);
    const auto parent_mesh_slice = gsl::at(parent_mesh_slices, d);
    const auto child_size = gsl::at(child_sizes, d);
    if (child_size == ChildSize::Full and
        child_mesh_slice == parent_mesh_slice) {
      // No projection necessary, keep matrix the identity in this dimension
      continue;
    }
    gsl::at(projection_matrix, d) = projection_matrix_parent_to_child(
        parent_mesh_slice, child_mesh_slice, child_size);
  }
  return projection_matrix;
}

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim> p_projection_matrices(
    const Mesh<Dim>& source_mesh, const Mesh<Dim>& target_mesh) {
  static const Matrix identity{};
  auto projection_matrices = make_array<Dim>(std::cref(identity));
  const auto source_mesh_slices = source_mesh.slices();
  const auto target_mesh_slices = target_mesh.slices();
  for (size_t d = 0; d < Dim; ++d) {
    const auto source_mesh_slice = gsl::at(source_mesh_slices, d);
    const auto target_mesh_slice = gsl::at(target_mesh_slices, d);
    if (source_mesh_slice == target_mesh_slice) {
      // No projection necessary, keep matrix the identity in this dimension
    } else if (source_mesh_slice.extents(0) <= target_mesh_slice.extents(0)) {
      gsl::at(projection_matrices, d) = projection_matrix_parent_to_child(
          source_mesh_slice, target_mesh_slice, ChildSize::Full);
    } else {
      gsl::at(projection_matrices, d) = projection_matrix_child_to_parent(
          source_mesh_slice, target_mesh_slice, ChildSize::Full);
    }
  }
  return projection_matrices;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data)                                                 \
  template bool needs_projection(                                            \
      const Mesh<DIM(data)>& mesh1, const Mesh<DIM(data)>& mesh2,            \
      const std::array<ChildSize, DIM(data)>& child_sizes);                  \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>       \
  projection_matrix_child_to_parent(                                         \
      const Mesh<DIM(data)>& child_mesh, const Mesh<DIM(data)>& parent_mesh, \
      const std::array<ChildSize, DIM(data)>& child_sizes,                   \
      bool operand_is_massive);                                              \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>       \
  projection_matrix_parent_to_child(                                         \
      const Mesh<DIM(data)>& parent_mesh, const Mesh<DIM(data)>& child_mesh, \
      const std::array<ChildSize, DIM(data)>& child_sizes);                  \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>       \
  p_projection_matrices(const Mesh<DIM(data)>& source_mesh,                  \
                        const Mesh<DIM(data)>& target_mesh);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Spectral
