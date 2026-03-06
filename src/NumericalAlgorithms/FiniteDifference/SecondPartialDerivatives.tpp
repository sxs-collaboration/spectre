// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/FiniteDifference/SecondPartialDerivatives.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace fd {
namespace detail {
template <size_t Dim>
void second_logical_partial_derivatives_impl(
    gsl::not_null<std::array<gsl::span<double>, Dim>*>
        pure_second_logical_derivatives,
    gsl::not_null<std::array<gsl::span<double>, Dim>*>
        mixed_second_logical_derivatives,
    gsl::span<double>* buffer, const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, size_t number_of_variables, size_t fd_order);
}  // namespace detail

namespace partial_derivatives_detail {
template <size_t Dim, typename VariableTags, typename DerivativeTags>
struct LogicalImpl;

// Note that the index of ddu is (i, j, k, d1, d2, s)
// where d's are the derivative indices and s is the tensor component and
// i varying the fastest, which is different from any of the logical
// derivatives index (d, i, j, k, s). We assume ddu is symmetric in the
// two derivative indices.
template <typename ResultTags, size_t Dim, typename DerivativeFrame,
          typename ValueType = typename Variables<ResultTags>::value_type,
          typename VectorType = typename Variables<ResultTags>::vector_type>
void second_partial_derivatives_impl(
    const gsl::not_null<Variables<ResultTags>*> ddu,
    const std::array<const ValueType*, Dim>&
        first_logical_partial_derivatives_of_u,
    const std::array<const ValueType*, Dim>&
        pure_second_logical_partial_derivatives_of_u,
    const std::array<const ValueType*, Dim>&
        mixed_second_logical_partial_derivatives_of_u,
    const size_t number_of_independent_components,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const InverseHessian<DataVector, Dim, Frame::ElementLogical,
                         DerivativeFrame>& inverse_hessian) {
  ValueType* ptr_ddu = ddu->data();
  const size_t num_grid_points = ddu->number_of_grid_points();
  VectorType lhs{};
  VectorType first_logical_du{};
  VectorType second_logical_du{};

  // The storage indices into the inverse Jacobian are precomputed to avoid
  // having to recompute them for each tensor component of `u`.
  std::array<std::array<size_t, Dim>, Dim> jacobian_indices{};
  for (size_t deriv_index = 0; deriv_index < Dim; ++deriv_index) {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(gsl::at(jacobian_indices, d), deriv_index) =
          InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>::get_storage_index(d, deriv_index);
    }
  }
  std::array<std::array<std::array<size_t, Dim>, Dim>, Dim> hessian_indices{};
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t first_deriv_index = 0; first_deriv_index < Dim;
         ++first_deriv_index) {
      for (size_t second_deriv_index = 0; second_deriv_index < Dim;
           ++second_deriv_index) {
        gsl::at(gsl::at(gsl::at(hessian_indices, d), first_deriv_index),
                second_deriv_index) =
            InverseHessian<
                DataVector, Dim, Frame::ElementLogical,
                DerivativeFrame>::get_storage_index(d, first_deriv_index,
                                                    second_deriv_index);
      }
    }
  }

  for (size_t component_index = 0;
       component_index < number_of_independent_components; ++component_index) {
    for (size_t first_deriv_index = 0; first_deriv_index < Dim;
         ++first_deriv_index) {
      for (size_t second_deriv_index = first_deriv_index;
           second_deriv_index < Dim; ++second_deriv_index) {
        lhs.set_data_ref(ptr_ddu, num_grid_points);
        ptr_ddu += num_grid_points;

        // clang-tidy: const cast is fine since we won't modify the data and we
        // need it to easily hook into the expression templates.
        second_logical_du.set_data_ref(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            const_cast<ValueType*>(
                gsl::at(pure_second_logical_partial_derivatives_of_u, 0)) +
                component_index * num_grid_points,
            num_grid_points);

        lhs = (*(inverse_jacobian.begin() +
                 gsl::at(gsl::at(jacobian_indices, 0), first_deriv_index))) *
              (*(inverse_jacobian.begin() +
                 gsl::at(gsl::at(jacobian_indices, 0), second_deriv_index))) *
              second_logical_du;

        for (size_t first_logical_deriv_index = 0;
             first_logical_deriv_index < Dim; ++first_logical_deriv_index) {
          for (size_t second_logical_deriv_index = 0;
               second_logical_deriv_index < Dim; ++second_logical_deriv_index) {
            if (first_logical_deriv_index + second_logical_deriv_index != 0) {
              if (first_logical_deriv_index == second_logical_deriv_index) {
                // clang-tidy: const cast is fine since we won't modify the data
                // and we need it to easily hook into the expression templates.
                second_logical_du.set_data_ref(
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                    const_cast<ValueType*>(
                        gsl::at(pure_second_logical_partial_derivatives_of_u,
                                first_logical_deriv_index)) +
                        component_index * num_grid_points,
                    num_grid_points);
              } else {
                // we need to map (first_logical_deriv_index,
                // second_logical_deriv_index) to an index
                // single_mixed_partial_index into the
                // mixed_second_logical_partial_derivatives_of_u. For
                // single_mixed_partial_index, 0 is xy deriv, 1 is yz deriv,
                // and 2 is xz deriv. So we need a symmetric map (0,1)->0,
                // (0,2)->2, (1,2)->1.
                const size_t single_mixed_partial_index =
                    [first_logical_deriv_index,
                     second_logical_deriv_index]() -> size_t {
                  if (first_logical_deriv_index * second_logical_deriv_index !=
                      0) {
                    return 1;
                  } else if (first_logical_deriv_index +
                                 second_logical_deriv_index ==
                             1) {
                    return 0;
                  } else {
                    return 2;
                  }
                }();
                // clang-tidy: const cast is fine since we won't modify the data
                // and we need it to easily hook into the expression templates.
                second_logical_du.set_data_ref(
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                    const_cast<ValueType*>(
                        gsl::at(mixed_second_logical_partial_derivatives_of_u,
                                single_mixed_partial_index)) +
                        component_index * num_grid_points,
                    num_grid_points);
              }

              lhs +=
                  (*(inverse_jacobian.begin() +
                     gsl::at(
                         gsl::at(jacobian_indices, first_logical_deriv_index),
                         first_deriv_index))) *
                  (*(inverse_jacobian.begin() +
                     gsl::at(
                         gsl::at(jacobian_indices, second_logical_deriv_index),
                         second_deriv_index))) *
                  second_logical_du;
            }
          }
        }
        // Now add in the terms with the inverse hessian
        for (size_t d = 0; d < Dim; ++d) {
          // clang-tidy: const cast is fine since we won't modify the data
          // and we need it to easily hook into the expression templates.
          first_logical_du.set_data_ref(
              // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
              const_cast<ValueType*>(
                  gsl::at(first_logical_partial_derivatives_of_u, d)) +
                  component_index * num_grid_points,
              num_grid_points);
          lhs += (*(inverse_hessian.begin() +
                    gsl::at(
                        gsl::at(gsl::at(hessian_indices, d), first_deriv_index),
                        second_deriv_index))) *
                 first_logical_du;
        }
      }
    }
  }
}
}  // namespace partial_derivatives_detail

template <typename DerivativeTags, size_t Dim, typename DerivativeFrame>
void second_partial_derivatives(
    const gsl::not_null<
        Variables<db::wrap_tags_in<::Tags::second_deriv, DerivativeTags,
                                   tmpl::size_t<Dim>, DerivativeFrame>>*>
        second_partial_derivatives,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian,
    const InverseHessian<DataVector, Dim, Frame::ElementLogical,
                         DerivativeFrame>& inverse_hessian) {
  // Note: Tags::second_deriv ensures the first two indices (partial derivative
  // indices) of tensors in second_partial_derivatives are symmetric.
  ASSERT(fd_order == 4, "Only fd_order == 4 is supported at the moment.");
  ASSERT(Dim == 3, "Only 3 dimensions is supported at the moment.");
  ASSERT(second_partial_derivatives->size() ==
             (Dim - 1) * Dim * volume_vars.size(),
         "The second partial derivatives Variables must have size "
             << (Dim - 1) * Dim * volume_vars.size()
             << " ((Dim-1) * Dim * volume_vars.size()) but has size "
             << second_partial_derivatives->size() << " and "
             << second_partial_derivatives->number_of_grid_points()
             << " grid points.");

  // compute the pure and second logical derivatives
  const size_t second_logical_derivs_internal_buffer_size =
      (volume_vars.size() +
       4 * alg::max_element(ghost_cell_vars,
                            [](const auto& a, const auto& b) {
                              return a.second.size() < b.second.size();
                            })
               ->second.size() +
       2 * volume_vars.size());
  DataVector buffer(3 * Dim * volume_vars.size() +
                    second_logical_derivs_internal_buffer_size);

  // The first_logical_partial_derivs is only needed for inverse_hessian
  std::array<gsl::span<double>, Dim> first_logical_partial_derivs{};
  std::array<gsl::span<double>, Dim> pure_second_logical_partial_derivs{};
  std::array<gsl::span<double>, Dim> mixed_second_logical_partial_derivs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(first_logical_partial_derivs, i) =
        gsl::make_span(&buffer[i * volume_vars.size()], volume_vars.size());
  }
  for (size_t i = Dim; i < 2 * Dim; ++i) {
    gsl::at(pure_second_logical_partial_derivs, i - Dim) =
        gsl::make_span(&buffer[i * volume_vars.size()], volume_vars.size());
  }
  for (size_t i = 2 * Dim; i < 3 * Dim; ++i) {
    gsl::at(mixed_second_logical_partial_derivs, i - 2 * Dim) =
        gsl::make_span(&buffer[i * volume_vars.size()], volume_vars.size());
  }
  gsl::span<double> span_buffer =
      gsl::make_span(&buffer[3 * Dim * volume_vars.size()],
                     second_logical_derivs_internal_buffer_size);

  // Compute first logical derivatives (used in the inverse-hessian terms)
  // Potential optimization: since we compute first logical derivatives here,
  // it may be a good idea to have an option to compute the first inertial
  // derivatives as well together with the second inertial derivatives.
  fd::logical_partial_derivatives(make_not_null(&first_logical_partial_derivs),
                                  volume_vars, ghost_cell_vars, volume_mesh,
                                  number_of_variables, fd_order);

  detail::second_logical_partial_derivatives_impl(
      make_not_null(&pure_second_logical_partial_derivs),
      make_not_null(&mixed_second_logical_partial_derivs), &span_buffer,
      volume_vars, ghost_cell_vars, volume_mesh, number_of_variables, fd_order);

  std::array<const double*, Dim> first_logical_partial_derivs_ptrs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(first_logical_partial_derivs_ptrs, i) =
        gsl::at(first_logical_partial_derivs, i).data();
  }

  std::array<const double*, Dim> pure_second_logical_partial_derivs_ptrs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(pure_second_logical_partial_derivs_ptrs, i) =
        gsl::at(pure_second_logical_partial_derivs, i).data();
  }

  std::array<const double*, Dim> mixed_second_logical_partial_derivs_ptrs{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(mixed_second_logical_partial_derivs_ptrs, i) =
        gsl::at(mixed_second_logical_partial_derivs, i).data();
  }

  partial_derivatives_detail::second_partial_derivatives_impl(
      second_partial_derivatives, first_logical_partial_derivs_ptrs,
      pure_second_logical_partial_derivs_ptrs,
      mixed_second_logical_partial_derivs_ptrs,
      Variables<DerivativeTags>::number_of_independent_components,
      inverse_jacobian, inverse_hessian);
}
}  // namespace fd
