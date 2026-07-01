// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <type_traits>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral {
namespace {
void verify_meshes(const Mesh<1>& parent_mesh, const Mesh<1>& child_mesh,
                   const SegmentSize size) {
  constexpr size_t max_points_l = maximum_number_of_points<Basis::Legendre>;
  constexpr size_t max_points_f = maximum_number_of_points<Basis::Fourier>;

  // Legendre with Legendre (any segment size)
  const bool both_legendre = parent_mesh.basis(0) == Basis::Legendre and
                             child_mesh.basis(0) == Basis::Legendre;

  // Fourier with Fourier (full segment only)
  const bool both_fourier = parent_mesh.basis(0) == Basis::Fourier and
                            child_mesh.basis(0) == Basis::Fourier and
                            size == SegmentSize::Full;

  // ZernikeB2 Equiangular with ZernikeB2 Equiangular (full segment only)
  const bool both_zernike =
      parent_mesh.basis(0) == Basis::ZernikeB2 and
      parent_mesh.quadrature(0) == Quadrature::Equiangular and
      child_mesh.basis(0) == Basis::ZernikeB2 and
      child_mesh.quadrature(0) == Quadrature::Equiangular and
      size == SegmentSize::Full;

  // Fourier with ZernikeB2 Equiangular (full segment only)
  const bool fourier_zernike =
      parent_mesh.basis(0) == Basis::Fourier and
      child_mesh.basis(0) == Basis::ZernikeB2 and
      child_mesh.quadrature(0) == Quadrature::Equiangular and
      size == SegmentSize::Full;

  // ZernikeB2 Equiangular with Fourier (full segment only)
  const bool zernike_fourier =
      parent_mesh.basis(0) == Basis::ZernikeB2 and
      parent_mesh.quadrature(0) == Quadrature::Equiangular and
      child_mesh.basis(0) == Basis::Fourier and size == SegmentSize::Full;

  ASSERT(both_legendre or both_fourier or both_zernike or fourier_zernike or
             zernike_fourier,
         "Unsupported projection combination. Supported: Legendre <-> Legendre "
         "or Fourier <-> Fourier (where ZernikeB2 with Equiangular quadrature "
         "also works) with no h-refinement (size must be Full). Got parent: "
             << parent_mesh << ", child: " << child_mesh << ", size: " << size);

  ASSERT((both_legendre and (parent_mesh.extents(0) <= max_points_l and
                             child_mesh.extents(0) <= max_points_l)) or
             (parent_mesh.extents(0) <= max_points_f and
              child_mesh.extents(0) <= max_points_f),
         "Mesh has more points than supported by its quadrature.");
}
}  // namespace

template <size_t Dim>
bool needs_projection(const Mesh<Dim>& mesh1, const Mesh<Dim>& mesh2,
                      const std::array<SegmentSize, Dim>& child_sizes) {
  return mesh1 != mesh2 or
         alg::any_of(child_sizes, [](const Spectral::SegmentSize child_size) {
           ASSERT(child_size != Spectral::SegmentSize::Uninitialized,
                  "Received uninitialized child_size.");
           return child_size != Spectral::SegmentSize::Full;
         });
}

const Matrix& projection_matrix_child_to_parent(const Mesh<1>& child_mesh,
                                                const Mesh<1>& parent_mesh,
                                                const SegmentSize size,
                                                const bool operand_is_massive) {
  constexpr size_t max_points_l = maximum_number_of_points<Basis::Legendre>;
  constexpr size_t max_points_f = maximum_number_of_points<Basis::Fourier>;
  verify_meshes(parent_mesh, child_mesh, size);

  if (operand_is_massive) {
    ASSERT(parent_mesh.extents(0) <= child_mesh.extents(0),
           "Requested projection matrix from child with fewer points ("
               << child_mesh.extents(0) << ") than the parent ("
               << parent_mesh.extents(0) << ")");
    // The restriction operator for massive quantities is just the interpolation
    // transpose
    const auto compute_matrix =
        [](const Basis parent_basis, const Quadrature parent_quadrature,
           const size_t parent_extent, const Basis child_basis,
           const Quadrature child_quadrature, const size_t child_extent,
           const SegmentSize local_child_size) -> Matrix {
      const auto& prolongation_operator = projection_matrix_parent_to_child(
          {parent_extent, parent_basis, parent_quadrature},
          {child_extent, child_basis, child_quadrature}, local_child_size);
      return blaze::trans(prolongation_operator);
    };
    const static auto cache_legendre = make_static_cache<
        CacheEnumeration<Quadrature, Quadrature::Gauss,
                         Quadrature::GaussLobatto>,
        CacheRange<2_st, max_points_l + 1>,
        CacheEnumeration<Quadrature, Quadrature::Gauss,
                         Quadrature::GaussLobatto>,
        CacheRange<2_st, max_points_l + 1>,
        CacheEnumeration<SegmentSize, SegmentSize::Full, SegmentSize::UpperHalf,
                         SegmentSize::LowerHalf>>(
        [compute_matrix](const Quadrature q_parent, const size_t n_parent,
                         const Quadrature q_child, const size_t n_child,
                         const SegmentSize local_child_size) {
          return compute_matrix(Basis::Legendre, q_parent, n_parent,
                                Basis::Legendre, q_child, n_child,
                                local_child_size);
        });
    const static auto cache_fourier =
        make_static_cache<CacheRange<2_st, max_points_f + 1>,
                          CacheRange<2_st, max_points_f + 1>>(
            [compute_matrix](const size_t n_parent, const size_t n_child) {
              return compute_matrix(Basis::Fourier, Quadrature::Equiangular,
                                    n_parent, Basis::Fourier,
                                    Quadrature::Equiangular, n_child,
                                    SegmentSize::Full);
            });
    if (parent_mesh.basis(0) == Basis::Legendre) {
      return cache_legendre(parent_mesh.quadrature(0), parent_mesh.extents(0),
                            child_mesh.quadrature(0), child_mesh.extents(0),
                            size);
    } else {
      // Both Fourier and ZernikeB2 get here
      // In terms of angular interpolation, ZernikeB2 with Equiangular
      // quadrature is identical to Fourier with Equiangular quadrature, so the
      // matrices do not make a distinction
      return cache_fourier(parent_mesh.extents(0), child_mesh.extents(0));
    }
  }

  switch (size) {
    case SegmentSize::Full: {
      const auto compute_matrix =
          [](const Basis basis_element, const Quadrature quadrature_element,
             const size_t extents_element, const Basis basis_mortar,
             const Quadrature quadrature_mortar, const size_t extents_mortar) {
            const Mesh<1> mesh_element(extents_element, basis_element,
                                       quadrature_element);
            const Mesh<1> mesh_mortar(extents_mortar, basis_mortar,
                                      quadrature_mortar);

            // The projection in spectral space is just a truncation
            // of the modes.
            const auto& spectral_to_grid_element =
                modal_to_nodal_matrix(mesh_element);
            const auto& grid_to_spectral_mortar =
                nodal_to_modal_matrix(mesh_mortar);
            const auto num_modes = std::min(extents_element, extents_mortar);
            Matrix projection(extents_element, extents_mortar, 0.);
            for (size_t i = 0; i < extents_element; ++i) {
              for (size_t j = 0; j < extents_mortar; ++j) {
                for (size_t k = 0; k < num_modes; ++k) {
                  projection(i, j) += spectral_to_grid_element(i, k) *
                                      grid_to_spectral_mortar(k, j);
                }
              }
            }

            return projection;
          };
      const static auto cache_legendre =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>>(
              [compute_matrix](const Quadrature q_parent, const size_t n_parent,
                               const Quadrature q_child, const size_t n_child) {
                return compute_matrix(Basis::Legendre, q_parent, n_parent,
                                      Basis::Legendre, q_child, n_child);
              });
      const static auto cache_fourier =
          make_static_cache<CacheRange<2_st, max_points_f + 1>,
                            CacheRange<2_st, max_points_f + 1>>(
              [compute_matrix](const size_t n_parent, const size_t n_child) {
                return compute_matrix(Basis::Fourier, Quadrature::Equiangular,
                                      n_parent, Basis::Fourier,
                                      Quadrature::Equiangular, n_child);
              });
      if (parent_mesh.basis(0) == Basis::Legendre) {
        return cache_legendre(parent_mesh.quadrature(0), parent_mesh.extents(0),
                              child_mesh.quadrature(0), child_mesh.extents(0));
      } else {
        // Both Fourier and ZernikeB2 get here
        return cache_fourier(parent_mesh.extents(0), child_mesh.extents(0));
      }
    }

    case SegmentSize::UpperHalf: {
      const static auto cache = make_static_cache<
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, max_points_l + 1>,
          CacheEnumeration<Quadrature, Quadrature::Gauss,
                           Quadrature::GaussLobatto>,
          CacheRange<2_st, max_points_l + 1>>(
          [](const Quadrature quadrature_element, const size_t extents_element,
             const Quadrature quadrature_mortar, const size_t extents_mortar) {
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
                                 static_cast<double>(
                                     2 * i * (2 * large_index + 1 - 2 * i) *
                                     (large_index + 2 - 2 * i) *
                                     (large_index + 1 - 2 * i));
              }

              for (size_t i = 1; i <= large_index - small_index; ++i) {
                result *=
                    1. + static_cast<double>(large_index + small_index + 1) /
                             static_cast<double>(i);
              }
              result /= pow(2., static_cast<int>(large_index) + 1);
              return result;
            };

            const auto& spectral_to_grid_element =
                modal_to_nodal_matrix(mesh_element);
            const auto& grid_to_spectral_mortar =
                nodal_to_modal_matrix(mesh_mortar);

            const auto num_modes = std::min(extents_element, extents_mortar);
            Matrix temp(extents_element, num_modes, 0.);
            for (size_t j = 0; j < num_modes; ++j) {
              for (size_t k = j; k < extents_element; ++k) {
                const double transformation_entry =
                    spectral_transformation(k, j);
                for (size_t i = 0; i < extents_element; ++i) {
                  temp(i, j) +=
                      spectral_to_grid_element(i, k) * transformation_entry;
                }
              }
            }

            Matrix projection(extents_element, extents_mortar, 0.);
            for (size_t i = 0; i < extents_element; ++i) {
              for (size_t j = 0; j < extents_mortar; ++j) {
                for (size_t k = 0; k < num_modes; ++k) {
                  projection(i, j) +=
                      temp(i, k) * grid_to_spectral_mortar(k, j);
                }
              }
            }

            return projection;
          });
      return cache(parent_mesh.quadrature(0), parent_mesh.extents(0),
                   child_mesh.quadrature(0), child_mesh.extents(0));
    }

    case SegmentSize::LowerHalf: {
      const static auto cache =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>>(
              [](const Quadrature quadrature_element,
                 const size_t extents_element,
                 const Quadrature quadrature_mortar,
                 const size_t extents_mortar) {
                const Mesh<1> mesh_element(extents_element, Basis::Legendre,
                                           quadrature_element);
                const Mesh<1> mesh_mortar(extents_mortar, Basis::Legendre,
                                          quadrature_mortar);

                // The lower-half matrices are generated from the upper-half
                // matrices using symmetry.
                const auto& projection_upper_half =
                    projection_matrix_child_to_parent(mesh_mortar, mesh_element,
                                                      SegmentSize::UpperHalf);

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
      ERROR("Invalid SegmentSize");
  }
}

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim>
projection_matrix_child_to_parent(
    const Mesh<Dim>& child_mesh, const Mesh<Dim>& parent_mesh,
    const std::array<SegmentSize, Dim>& child_sizes,
    const bool operand_is_massive) {
  static const Matrix identity{};
  auto projection_matrix = make_array<Dim>(std::cref(identity));
  const auto child_mesh_slices = child_mesh.slices();
  const auto parent_mesh_slices = parent_mesh.slices();
  for (size_t d = 0; d < Dim; ++d) {
    const auto child_mesh_slice = gsl::at(child_mesh_slices, d);
    const auto parent_mesh_slice = gsl::at(parent_mesh_slices, d);
    const auto child_size = gsl::at(child_sizes, d);
    ASSERT(child_mesh_slice.basis(0) == Basis::Legendre or
               child_size == SegmentSize::Full,
           "The Fourier basis does not support h-refinement: SegmentSize must "
           "be Full in dimension "
               << d << ", but got " << child_size);
    if (child_size == SegmentSize::Full and
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
                                                const SegmentSize size) {
  constexpr size_t max_points_l = maximum_number_of_points<Basis::Legendre>;
  verify_meshes(parent_mesh, child_mesh, size);

  // For Fourier with SegmentSize::Full, the element-to-mortar projection is
  // the same spectral truncation/zero-padding as child_to_parent with roles
  // swapped. Delegate to avoid duplicating the cached computation.
  if (parent_mesh.basis(0) == Basis::Fourier or
      parent_mesh.basis(0) == Basis::ZernikeB2) {
    return projection_matrix_child_to_parent(parent_mesh, child_mesh,
                                             SegmentSize::Full);
  }

  ASSERT(child_mesh.extents(0) >= parent_mesh.extents(0),
         "Requested projection matrix to mortar with fewer points ("
             << child_mesh.extents(0) << ") than the element ("
             << parent_mesh.extents(0) << ")");

  // Element-to-mortar projections on Legendre meshes are interpolations.
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
    case SegmentSize::Full: {
      const static auto cache =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>>(
              make_interpolators([](const DataVector& x) { return x; }));
      return cache(child_mesh.quadrature(0), child_mesh.extents(0),
                   parent_mesh.quadrature(0), parent_mesh.extents(0));
    }

    case SegmentSize::UpperHalf: {
      const static auto cache =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>>(
              make_interpolators([](const DataVector& x) {
                return DataVector(0.5 * (x + 1.));
              }));
      return cache(child_mesh.quadrature(0), child_mesh.extents(0),
                   parent_mesh.quadrature(0), parent_mesh.extents(0));
    }

    case SegmentSize::LowerHalf: {
      const static auto cache =
          make_static_cache<CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>,
                            CacheEnumeration<Quadrature, Quadrature::Gauss,
                                             Quadrature::GaussLobatto>,
                            CacheRange<2_st, max_points_l + 1>>(
              make_interpolators([](const DataVector& x) {
                return DataVector(0.5 * (x - 1.));
              }));
      return cache(child_mesh.quadrature(0), child_mesh.extents(0),
                   parent_mesh.quadrature(0), parent_mesh.extents(0));
    }

    default:
      ERROR("Invalid SegmentSize");
  }
}

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim>
projection_matrix_parent_to_child(
    const Mesh<Dim>& parent_mesh, const Mesh<Dim>& child_mesh,
    const std::array<SegmentSize, Dim>& child_sizes) {
  static const Matrix identity{};
  auto projection_matrix = make_array<Dim>(std::cref(identity));
  const auto child_mesh_slices = child_mesh.slices();
  const auto parent_mesh_slices = parent_mesh.slices();
  for (size_t d = 0; d < Dim; ++d) {
    const auto child_mesh_slice = gsl::at(child_mesh_slices, d);
    const auto parent_mesh_slice = gsl::at(parent_mesh_slices, d);
    const auto child_size = gsl::at(child_sizes, d);
    ASSERT(child_mesh_slice.basis(0) == Basis::Legendre or
               child_size == SegmentSize::Full,
           "The Fourier basis does not support h-refinement: SegmentSize must "
           "be Full in dimension "
               << d << ", but got " << child_size);
    if (child_size == SegmentSize::Full and
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
std::array<std::reference_wrapper<const Matrix>, Dim> projection_matrices(
    const Mesh<Dim>& source_mesh, const Mesh<Dim>& target_mesh,
    const std::array<SegmentSize, Dim>& source_sizes,
    const std::array<SegmentSize, Dim>& target_sizes,
    const bool operand_is_massive) {
  static const Matrix identity{};
  auto projection_matrices = make_array<Dim>(std::cref(identity));
  const auto source_mesh_slices = source_mesh.slices();
  const auto target_mesh_slices = target_mesh.slices();
  for (size_t d = 0; d < Dim; ++d) {
    const auto& source_mesh_slice = gsl::at(source_mesh_slices, d);
    const auto& target_mesh_slice = gsl::at(target_mesh_slices, d);
    const auto source_size = gsl::at(source_sizes, d);
    const auto target_size = gsl::at(target_sizes, d);
    if (source_size == target_size) {
      if (source_mesh_slice == target_mesh_slice) {
        // No projection necessary, keep matrix the identity in this dimension
      } else {
        // p-restriction and p-prolongation are the same (mode
        // truncation/zero-padding)
        gsl::at(projection_matrices, d) = projection_matrix_child_to_parent(
            source_mesh_slice, target_mesh_slice, SegmentSize::Full,
            operand_is_massive);
      }
    } else if (source_size == SegmentSize::Full) {
      // h-prolongation
      ASSERT(not operand_is_massive,
             "Massive projection is a restriction (child to parent); cannot "
             "prolong a massive quantity from parent to child.");
      gsl::at(projection_matrices, d) = projection_matrix_parent_to_child(
          source_mesh_slice, target_mesh_slice, target_size);
    } else {
      // h-restriction
      ASSERT(target_size == SegmentSize::Full,
             "Cannot project from one half of segment to the other.");
      gsl::at(projection_matrices, d) = projection_matrix_child_to_parent(
          source_mesh_slice, target_mesh_slice, source_size,
          operand_is_massive);
    }
  }
  return projection_matrices;
}

template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim> p_projection_matrices(
    const Mesh<Dim>& source_mesh, const Mesh<Dim>& target_mesh) {
  return projection_matrices(source_mesh, target_mesh,
                             make_array<Dim>(SegmentSize::Full),
                             make_array<Dim>(SegmentSize::Full));
}

void project_spherical_harmonics(const gsl::not_null<double*> result_data,
                                 const double* const source_data,
                                 const size_t num_components,
                                 const size_t num_radial_points,
                                 const size_t l_max_source,
                                 const size_t l_max_target,
                                 const bool operand_is_massive) {
  const ylm::Spherepack& ylm_source = ylm::get_spherepack_cache(l_max_source);
  const ylm::Spherepack& ylm_target = ylm::get_spherepack_cache(l_max_target);
  const size_t source_points_per_component =
      num_radial_points * ylm_source.physical_size();
  const size_t target_points_per_component =
      num_radial_points * ylm_target.physical_size();
  DataVector source_coefficients(ylm_source.spectral_size() *
                                 num_radial_points);
  DataVector target_coefficients(ylm_target.spectral_size() *
                                 num_radial_points);
  // For massive restriction multiply by the angular quadrature weights:
  // I^T = W_target * P_L2 * W_source^{-1}
  const auto scale_by_angular_weights = [num_radial_points](
                                            double* const data,
                                            const std::vector<double>& weights,
                                            const bool invert) {
    for (size_t angular = 0; angular < weights.size(); ++angular) {
      const double factor = invert ? 1.0 / weights[angular] : weights[angular];
      for (size_t radial = 0; radial < num_radial_points; ++radial) {
        data[angular * num_radial_points + radial] *= factor;
      }
    }
  };
  DataVector weighted_source(operand_is_massive ? source_points_per_component
                                                : 0);
  for (size_t component = 0; component < num_components; ++component) {
    const double* source_component =
        source_data + component * source_points_per_component;
    if (operand_is_massive) {
      std::copy(source_component,
                source_component + source_points_per_component,
                weighted_source.data());
      scale_by_angular_weights(weighted_source.data(),
                               ylm_source.integration_weights(), true);
      source_component = weighted_source.data();
    }
    ylm_source.phys_to_spec_all_offsets(
        make_not_null(source_coefficients.data()),
        make_not_null(source_component), num_radial_points);
    ylm::Spherepack::prolong_or_restrict(
        make_not_null(&target_coefficients), source_coefficients, l_max_source,
        l_max_source, l_max_target, l_max_target, num_radial_points);
    double* const result_component =
        result_data.get() + component * target_points_per_component;
    ylm_target.spec_to_phys_all_offsets(
        make_not_null(result_component),
        make_not_null(target_coefficients.data()), num_radial_points);
    if (operand_is_massive) {
      scale_by_angular_weights(result_component,
                               ylm_target.integration_weights(), false);
    }
  }
}

template <typename VectorType, size_t Dim>
void project(const gsl::not_null<VectorType*> result, const VectorType& source,
             const Mesh<Dim>& source_mesh, const Mesh<Dim>& target_mesh,
             const std::array<SegmentSize, Dim>& source_sizes,
             const std::array<SegmentSize, Dim>& target_sizes,
             const bool operand_is_massive) {
  ASSERT(result->data() != source.data(),
         "The result must not alias the source, because the result is resized "
         "to the target number of grid points.");
  // The source may hold several components laid out component-major (e.g. a
  // `Variables` or the multiple fields of DG boundary data), so infer the
  // component count from its size.
  ASSERT(source.size() % source_mesh.number_of_grid_points() == 0,
         "The source size (" << source.size()
                             << ") must be a multiple of the number of grid "
                                "points in the source mesh ("
                             << source_mesh.number_of_grid_points() << ").");
  const size_t num_components =
      source.size() / source_mesh.number_of_grid_points();
  result->destructive_resize(num_components *
                             target_mesh.number_of_grid_points());
  // Handle spherical shells. This is only supported for real data because the
  // Spherepack angular transforms operate on `double`.
  if constexpr (std::is_same_v<VectorType, DataVector> and
                (Dim == 2 or Dim == 3)) {
    constexpr size_t theta_dim = Dim - 2;
    if (source_mesh.basis(theta_dim) == Basis::SphericalHarmonic and
        source_mesh.basis(theta_dim + 1) == Basis::SphericalHarmonic) {
      ASSERT(target_mesh.basis(theta_dim) == Basis::SphericalHarmonic and
                 target_mesh.basis(theta_dim + 1) == Basis::SphericalHarmonic,
             "Both source and target meshes must be spherical-shell meshes.");
      ASSERT(gsl::at(source_sizes, theta_dim) == SegmentSize::Full and
                 gsl::at(source_sizes, theta_dim + 1) == SegmentSize::Full and
                 gsl::at(target_sizes, theta_dim) == SegmentSize::Full and
                 gsl::at(target_sizes, theta_dim + 1) == SegmentSize::Full,
             "Spherical shells don't support angular h-refinement, so "
             "the angular segment sizes must be Full.");
      const size_t l_max_source = source_mesh.extents(theta_dim) - 1;
      const size_t l_max_target = target_mesh.extents(theta_dim) - 1;
      ASSERT(source_mesh.extents(theta_dim + 1) == 2 * l_max_source + 1 and
                 target_mesh.extents(theta_dim + 1) == 2 * l_max_target + 1,
             "Spherical-shell projection assumes m_max == l_max, i.e. "
             "n_phi == 2 * l_max + 1.");
      size_t num_radial_points = 1;
      if constexpr (Dim == 3) {
        // In 3D, project the radial dimension first
        num_radial_points = target_mesh.extents(0);
        const Matrix& radial_matrix = projection_matrices<1>(
            source_mesh.slice_through(0), target_mesh.slice_through(0),
            {{gsl::at(source_sizes, 0)}}, {{gsl::at(target_sizes, 0)}},
            operand_is_massive)[0];
        static const Matrix identity{};
        if (radial_matrix != identity) {
          const auto radial_matrices =
              std::array{std::cref(radial_matrix), std::cref(identity),
                         std::cref(identity)};
          if (l_max_source == l_max_target) {
            // Only radial projection needed, write directly into `result`
            apply_matrices(result, radial_matrices, source,
                           source_mesh.extents());
          } else {
            // First radial, then angular projection
            const auto radially_projected =
                apply_matrices(radial_matrices, source, source_mesh.extents());
            project_spherical_harmonics(
                make_not_null(result->data()), radially_projected.data(),
                num_components, num_radial_points, l_max_source, l_max_target,
                operand_is_massive);
          }
          return;
        }
      }
      if (l_max_source == l_max_target) {
        *result = source;
      } else {
        project_spherical_harmonics(
            make_not_null(result->data()), source.data(), num_components,
            num_radial_points, l_max_source, l_max_target, operand_is_massive);
      }
      return;
    }
  }
  // Fall back to tensor-product projection for non-spherical meshes
  apply_matrices(result,
                 projection_matrices(source_mesh, target_mesh, source_sizes,
                                     target_sizes, operand_is_massive),
                 source, source_mesh.extents());
}

template <size_t DimMinusOne>
size_t hash(const std::array<Spectral::SegmentSize, DimMinusOne>& mortar_size) {
  if constexpr (DimMinusOne == 2) {
    ASSERT(mortar_size[0] != Spectral::SegmentSize::Uninitialized and
               mortar_size[1] != Spectral::SegmentSize::Uninitialized,
           "Cannot hash an uninitialized mortar_size: " << mortar_size[0] << " "
                                                        << mortar_size[1]);
    return (static_cast<size_t>(mortar_size[0]) - 1) % 2 +
           2 * ((static_cast<size_t>(mortar_size[1]) - 1) % 2);
  } else if constexpr (DimMinusOne == 1) {
    ASSERT(mortar_size[0] != Spectral::SegmentSize::Uninitialized,
           "Cannot hash an uninitialized mortar_size: " << mortar_size[0]);
    return (static_cast<size_t>(mortar_size[0]) - 1) % 2;
  } else if constexpr (DimMinusOne == 0) {
    (void)mortar_size;
    return 0;
  } else {
    static_assert(DimMinusOne == 2 or DimMinusOne == 1 or DimMinusOne == 0);
  }
}

template <size_t Dim>
size_t MortarSizeHash<Dim>::operator()(
    const std::array<Spectral::SegmentSize, Dim - 1>& mortar_size) {
  return hash(mortar_size);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(r, data)                                                  \
  template bool needs_projection(                                             \
      const Mesh<DIM(data)>& mesh1, const Mesh<DIM(data)>& mesh2,             \
      const std::array<SegmentSize, DIM(data)>& child_sizes);                 \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>        \
  projection_matrix_child_to_parent(                                          \
      const Mesh<DIM(data)>& child_mesh, const Mesh<DIM(data)>& parent_mesh,  \
      const std::array<SegmentSize, DIM(data)>& child_sizes,                  \
      bool operand_is_massive);                                               \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>        \
  projection_matrix_parent_to_child(                                          \
      const Mesh<DIM(data)>& parent_mesh, const Mesh<DIM(data)>& child_mesh,  \
      const std::array<SegmentSize, DIM(data)>& child_sizes);                 \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>        \
  projection_matrices(const Mesh<DIM(data)>& source_mesh,                     \
                      const Mesh<DIM(data)>& target_mesh,                     \
                      const std::array<SegmentSize, DIM(data)>& source_sizes, \
                      const std::array<SegmentSize, DIM(data)>& target_sizes, \
                      bool operand_is_massive);                               \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>        \
  p_projection_matrices(const Mesh<DIM(data)>& source_mesh,                   \
                        const Mesh<DIM(data)>& target_mesh);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3))
#undef INSTANTIATE

#define VECTOR(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(r, data)                                                  \
  template void project(                                                      \
      gsl::not_null<VECTOR(data)*> result, const VECTOR(data) & source,       \
      const Mesh<DIM(data)>& source_mesh, const Mesh<DIM(data)>& target_mesh, \
      const std::array<SegmentSize, DIM(data)>& source_sizes,                 \
      const std::array<SegmentSize, DIM(data)>& target_sizes,                 \
      bool operand_is_massive);
GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3),
                        (DataVector, ComplexDataVector))
#undef INSTANTIATE
#undef VECTOR

#define INSTANTIATE(r, data) template class MortarSizeHash<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))
#undef INSTANTIATE

#define INSTANTIATE(r, data)       \
  template size_t hash<DIM(data)>( \
      const std::array<Spectral::SegmentSize, DIM(data)>& mortar_size);
GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))
#undef INSTANTIATE

#undef DIM
}  // namespace Spectral
