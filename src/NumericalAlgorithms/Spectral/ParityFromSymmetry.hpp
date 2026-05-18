// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace Spectral {
/*!
 * \brief A compile-time function to determine parity, with respect to the
 * \f$x\f$ coordinate in an axisymmetric spacetime, of tensors in a Variables
 *
 * \details In a \f$d\f$-dimensional space where fields are regular everywhere
 * (particularly at the axis), we utilize the fact that smoothness corresponds
 * to a well-defined Taylor expansion in all \f$d\f$ coordinates. Viewing the
 * components of a rank \f$(p, q)\f$ tensor as smooth functions of \f$(x, y,
 * \ldots, z)\f$, consider a reflection transformation \f$x \to -x\f$. As a
 * coordinate transformation, the Jacobian will look like \f$\mathrm{diag}(-1,
 * 1, 1, \ldots)\f$.
 *
 * If any of the \f$(p, q)\f$ indices correspond to the \f$x\f$-coordinate for
 * this particular component, there is a \f$-1\f$ factor associated with each
 * \f$x\f$-index, i.e., the Jacobian will look like \f$(-1)^{n_x}\f$, where
 * \f$n_x\f$ is the number of \f$x\f$-indices of the component.
 *
 * If this space is symmetric under this reflection, as it is for axisymmetry
 * about the \f$y\f$-axis in 3 dimensions, components can only represent
 * functions invariant under this transformation.
 *
 * This means for an even number of \f$x\f$-indices, the Taylor expansion must
 * only have even powers of \f$x\f$, and for an odd number of \f$x\f$-indices,
 * the expansion must have purely odd powers of \f$x\f$.
 *
 * The primary use-case for this is axisymmetric Cartoon simulations, where
 * elements touching the symmetry axis require ZernikeB1 bases for numerical
 * stability. These bases require knowledge of component parity for certain
 * operations (differentiation, interpolation, etc.).
 *
 * The returned array, alternating values for even/odd, stores the number of
 * next components with the same parity. When there are any neighboring parities
 * (i.e. not the worst-case scenario) the array will be padded with zeros to
 * be the correct size. The first index may be 0 if the first component is
 * odd, but any following zeros are guaranteed to only have zeros following it.
 * The first returned `size_t` is the number of even components, while the
 * second is the number of odd components.
 */
template <typename VariablesTags>
constexpr std::tuple<
    std::array<size_t,
               Variables<VariablesTags>::number_of_independent_components + 1>,
    size_t, size_t>
compute_parity_list() {
  const size_t N =
      Variables<VariablesTags>::number_of_independent_components + 1;
  std::array<size_t, N> parity_run_lengths{};

  const auto is_x_coordinate_index = [](const IndexType index_type,
                                        const size_t index_value) {
    return (index_type == IndexType::Spacetime and index_value == 1) or
           (index_type == IndexType::Spatial and index_value == 0);
  };

  size_t run_index = 0;
  bool current_parity_is_even = true;
  tmpl::for_each<VariablesTags>([&parity_run_lengths, &run_index,
                                 &current_parity_is_even,
                                 &is_x_coordinate_index]<typename TensorTag>(
                                    tmpl::type_<TensorTag> /*meta*/) {
    using tensor_type = typename TensorTag::type;
    constexpr auto index_types = tensor_type::index_types();
    constexpr size_t tensor_size = tensor_type::size();

    for (size_t component_index = 0; component_index < tensor_size;
         ++component_index) {
      const auto tensor_index = tensor_type::get_tensor_index(component_index);
      size_t x_coordinate_count = 0;
      for (size_t index_position = 0; index_position < index_types.size();
           ++index_position) {
        if (is_x_coordinate_index(gsl::at(index_types, index_position),
                                  gsl::at(tensor_index, index_position))) {
          ++x_coordinate_count;
        }
      }
      // If current parity doesn't match last
      const bool component_is_even = (x_coordinate_count % 2 == 0);
      if (component_is_even != current_parity_is_even) {
        ++run_index;
        current_parity_is_even = !current_parity_is_even;
      }
      gsl::at(parity_run_lengths, run_index) += 1;
    }
  });

  const auto [num_even, num_odd] = [&parity_run_lengths]<size_t... Is>(
                                       std::index_sequence<Is...> /*meta*/) {
    std::size_t even_count = ((Is % 2 == 0 ? parity_run_lengths[Is] : 0) + ...);
    std::size_t odd_count = ((Is % 2 == 1 ? parity_run_lengths[Is] : 0) + ...);
    return std::pair{even_count, odd_count};
  }(std::make_index_sequence<N>{});

  return {parity_run_lengths, num_even, num_odd};
}

/*!
 * \brief A compile-time function to determine parity, with respect to the
 * \f$x\f$ coordinate in an axisymmetric spacetime, of a tensor.
 *
 * \see `compute_parity_list(Variables)`
 */
template <typename TensorType>
  requires(tt::is_a_v<Tensor, TensorType>)
constexpr std::tuple<std::array<size_t, TensorType::size() + 1>, size_t, size_t>
compute_parity_list() {
  using tensor_type = Tensor<DataVector, typename TensorType::symmetry,
                             typename TensorType::index_list>;
  using vars_list = ::Tags::convert_to_temp_tensors<tmpl::list<tensor_type>, 0>;
  return compute_parity_list<vars_list>();
}
/*!
 * \brief Returns a compile-time array mapping each component index of
 * `TensorType` to its `Parity` (Even or Odd) based on the number of
 * \f$x\f$-coordinate indices, for use in an axisymmetric spacetime with the
 * Cartoon method.
 *
 * A component is `Parity::Even` if it has an even number of
 * \f$x\f$-coordinate indices, or `Parity::Odd` otherwise.
 *
 * \see `compute_parity_list`
 */
template <typename TensorType>
  requires(tt::is_a_v<Tensor, TensorType>)
constexpr std::array<Parity, TensorType::size()> make_component_parity_array() {
  constexpr auto parity_info = compute_parity_list<TensorType>();
  constexpr auto parity_list = std::get<0>(parity_info);
  constexpr size_t N = TensorType::size();
  std::array<Parity, N> result{};
  size_t component = 0;
  bool is_even = true;
  for (size_t i = 0; component < N; ++i) {
    const size_t seg_size = parity_list[i];
    if (seg_size == 0) {
      if (is_even) {
        is_even = false;
        continue;
      } else {
        break;
      }
    }
    for (size_t k = 0; k < seg_size; ++k, ++component) {
      result[component] = is_even ? Parity::Even : Parity::Odd;
    }
    is_even = not is_even;
  }
  return result;
}
}  // namespace Spectral
