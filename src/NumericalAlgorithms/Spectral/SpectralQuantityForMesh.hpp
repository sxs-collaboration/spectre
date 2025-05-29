// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
// clang-tidy: Macro arguments should be in parentheses, but we want to append
// template parameters here.
#define SPECTRAL_QUANTITY_FOR_MESH(function_name, return_type)           \
  const return_type& function_name(const Mesh<1>& mesh) {                \
    return Spectral::detail::get_spectral_quantity_for_mesh(             \
        [](const auto basis, const auto quadrature,                      \
           const size_t num_points) -> const return_type& {              \
          return function_name</* NOLINT */ decltype(basis)::value,      \
                               decltype(quadrature)::value>(num_points); \
        },                                                               \
        mesh);                                                           \
  }
/// \endcond
