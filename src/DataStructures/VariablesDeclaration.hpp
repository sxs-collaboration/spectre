// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace detail {
template <typename TagsList>
struct get_vector_type {
  using type = void;
};

template <typename Tag0, typename... Tags>
struct get_vector_type<tmpl::list<Tag0, Tags...>> {
  using type = typename Tag0::type::type;
};
}  // namespace detail

/// Implementation of the Variables class that can be specialized for different
/// vector types of the underlying tensors, e.g. DataVector or Kokkos::View.
template <typename TagsList,
          typename VectorType =
              typename detail::get_vector_type<TagsList>::type>
class Variables;
/// \endcond
