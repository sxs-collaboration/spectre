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
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
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
    const std::array<SegmentSize, Dim>& target_sizes) {
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
        gsl::at(projection_matrices, d) = projection_matrix_child_to_parent(
            source_mesh_slice, target_mesh_slice, SegmentSize::Full);
      }
    } else if (source_size == SegmentSize::Full) {
      gsl::at(projection_matrices, d) = projection_matrix_parent_to_child(
          source_mesh_slice, target_mesh_slice, target_size);
    } else {
      ASSERT(target_size == SegmentSize::Full,
             "Cannot project from one half of segment to the other.");
      gsl::at(projection_matrices, d) = projection_matrix_child_to_parent(
          source_mesh_slice, target_mesh_slice, source_size);
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
#define INSTANTIATE(r, data)                                                   \
  template bool needs_projection(                                              \
      const Mesh<DIM(data)>& mesh1, const Mesh<DIM(data)>& mesh2,              \
      const std::array<SegmentSize, DIM(data)>& child_sizes);                  \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>         \
  projection_matrix_child_to_parent(                                           \
      const Mesh<DIM(data)>& child_mesh, const Mesh<DIM(data)>& parent_mesh,   \
      const std::array<SegmentSize, DIM(data)>& child_sizes,                   \
      bool operand_is_massive);                                                \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>         \
  projection_matrix_parent_to_child(                                           \
      const Mesh<DIM(data)>& parent_mesh, const Mesh<DIM(data)>& child_mesh,   \
      const std::array<SegmentSize, DIM(data)>& child_sizes);                  \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>         \
  projection_matrices(const Mesh<DIM(data)>& source_mesh,                      \
                      const Mesh<DIM(data)>& target_mesh,                      \
                      const std::array<SegmentSize, DIM(data)>& source_sizes,  \
                      const std::array<SegmentSize, DIM(data)>& target_sizes); \
  template std::array<std::reference_wrapper<const Matrix>, DIM(data)>         \
  p_projection_matrices(const Mesh<DIM(data)>& source_mesh,                    \
                        const Mesh<DIM(data)>& target_mesh);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3))
#undef INSTANTIATE

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
