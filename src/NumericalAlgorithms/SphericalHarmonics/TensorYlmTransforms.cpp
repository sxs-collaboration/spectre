// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmTransforms.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmCartToSphere.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmSphereToCart.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace ylm::TensorYlm {

namespace {

template <typename TensorType>
void check_tensor_ylm_transform_sizes(const TensorType& coefficients,
                                      const size_t spectral_size,
                                      const size_t number_of_offsets) {
  for (size_t component = 0; component < coefficients.size(); ++component) {
    ASSERT(coefficients[component].size() == spectral_size * number_of_offsets,
           "Expected each tensor component to have size "
               << spectral_size * number_of_offsets << ", but component "
               << component << " has size " << coefficients[component].size()
               << ".");
  }
}

template <typename TensorType>
void apply_transform_matrix(const gsl::not_null<TensorType*> result,
                            const TensorType& coefficients, const size_t l_max,
                            const size_t number_of_offsets,
                            const SimpleSparseMatrix& transform_matrix) {
  if constexpr (TensorType::rank() == 0) {
    *result = coefficients;
    return;
  }

  const SpherepackIterator iter{l_max, l_max, 1, false};
  const size_t spectral_size = iter.spherepack_array_size();
  check_tensor_ylm_transform_sizes(coefficients, spectral_size,
                                   number_of_offsets);

  set_number_of_grid_points(result, coefficients);

  DataVector source{coefficients.size() * spectral_size};
  DataVector destination{result->size() * spectral_size};
  for (size_t offset = 0; offset < number_of_offsets; ++offset) {
    for (size_t component = 0; component < coefficients.size(); ++component) {
      for (size_t coefficient_index = 0; coefficient_index < spectral_size;
           ++coefficient_index) {
        source[component * spectral_size + coefficient_index] =
            coefficients[component]
                        [coefficient_index * number_of_offsets + offset];
      }
    }
    destination = 0.0;
    const gsl::span<double> source_span{source.data(), source.size()};
    gsl::span<double> destination_span{destination.data(), destination.size()};
    transform_matrix.increment_multiply_on_right(
        make_not_null(&destination_span), 0, 1, source_span, 0, 1);

    for (size_t component = 0; component < result->size(); ++component) {
      for (size_t coefficient_index = 0; coefficient_index < spectral_size;
           ++coefficient_index) {
        (*result)[component][coefficient_index * number_of_offsets + offset] =
            destination[component * spectral_size + coefficient_index];
      }
    }
  }
}

}  // namespace

template <typename TensorType>
void scalar_to_tensor_ylm_coefficients(
    const gsl::not_null<TensorType*> result,
    const TensorType& scalar_ylm_coefficients, const size_t l_max,
    const size_t number_of_offsets,
    const CoefficientNormalization coefficient_normalization) {
  ASSERT(result.get() != &scalar_ylm_coefficients,
         "The output tensor must not alias the input scalar-Ylm coefficients.");
  if constexpr (TensorType::rank() == 0) {
    *result = scalar_ylm_coefficients;
  } else {
    SimpleSparseMatrix transform_matrix{};
    fill_cart_to_sphere<typename TensorType::structure>(
        make_not_null(&transform_matrix), l_max, coefficient_normalization);
    apply_transform_matrix(result, scalar_ylm_coefficients, l_max,
                           number_of_offsets, transform_matrix);
  }
}

template <typename TensorType>
TensorType scalar_to_tensor_ylm_coefficients(
    const TensorType& scalar_ylm_coefficients, const size_t l_max,
    const size_t number_of_offsets,
    const CoefficientNormalization coefficient_normalization) {
  auto result = TensorType{};
  set_number_of_grid_points(make_not_null(&result), scalar_ylm_coefficients);
  scalar_to_tensor_ylm_coefficients(
      make_not_null(&result), scalar_ylm_coefficients, l_max, number_of_offsets,
      coefficient_normalization);
  return result;
}

template <typename TensorType>
void tensor_to_scalar_ylm_coefficients(
    const gsl::not_null<TensorType*> result,
    const TensorType& tensor_ylm_coefficients, const size_t l_max,
    const size_t number_of_offsets,
    const CoefficientNormalization coefficient_normalization) {
  ASSERT(result.get() != &tensor_ylm_coefficients,
         "The output tensor must not alias the input tensor-Ylm coefficients.");
  if constexpr (TensorType::rank() == 0) {
    *result = tensor_ylm_coefficients;
  } else {
    SimpleSparseMatrix transform_matrix{};
    fill_sphere_to_cart<typename TensorType::structure>(
        make_not_null(&transform_matrix), l_max, coefficient_normalization);
    apply_transform_matrix(result, tensor_ylm_coefficients, l_max,
                           number_of_offsets, transform_matrix);
  }
}

template <typename TensorType>
TensorType tensor_to_scalar_ylm_coefficients(
    const TensorType& tensor_ylm_coefficients, const size_t l_max,
    const size_t number_of_offsets,
    const CoefficientNormalization coefficient_normalization) {
  auto result = TensorType{};
  set_number_of_grid_points(make_not_null(&result), tensor_ylm_coefficients);
  tensor_to_scalar_ylm_coefficients(
      make_not_null(&result), tensor_ylm_coefficients, l_max, number_of_offsets,
      coefficient_normalization);
  return result;
}

#define TENSOR(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TENSOR_TYPE(data) TENSOR(data)<DataVector, 3>

#define INSTANTIATE(_, data)                                    \
  template void scalar_to_tensor_ylm_coefficients(              \
      gsl::not_null<TENSOR_TYPE(data)*> result,                 \
      const TENSOR_TYPE(data) & coefficients, size_t l_max,     \
      size_t number_of_offsets,                                 \
      CoefficientNormalization coefficient_normalization);      \
  template TENSOR_TYPE(data) scalar_to_tensor_ylm_coefficients( \
      const TENSOR_TYPE(data) & coefficients, size_t l_max,     \
      size_t number_of_offsets,                                 \
      CoefficientNormalization coefficient_normalization);      \
  template void tensor_to_scalar_ylm_coefficients(              \
      gsl::not_null<TENSOR_TYPE(data)*> result,                 \
      const TENSOR_TYPE(data) & coefficients, size_t l_max,     \
      size_t number_of_offsets,                                 \
      CoefficientNormalization coefficient_normalization);      \
  template TENSOR_TYPE(data) tensor_to_scalar_ylm_coefficients( \
      const TENSOR_TYPE(data) & coefficients, size_t l_max,     \
      size_t number_of_offsets,                                 \
      CoefficientNormalization coefficient_normalization);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (tnsr::i, tnsr::ii, tnsr::ij, tnsr::ijj, tnsr::ijk))

#undef INSTANTIATE
#undef TENSOR_TYPE
#undef TENSOR

template void scalar_to_tensor_ylm_coefficients(
    gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& coefficients, size_t l_max,
    size_t number_of_offsets,
    CoefficientNormalization coefficient_normalization);
template Scalar<DataVector> scalar_to_tensor_ylm_coefficients(
    const Scalar<DataVector>& coefficients, size_t l_max,
    size_t number_of_offsets,
    CoefficientNormalization coefficient_normalization);
template void tensor_to_scalar_ylm_coefficients(
    gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& coefficients, size_t l_max,
    size_t number_of_offsets,
    CoefficientNormalization coefficient_normalization);
template Scalar<DataVector> tensor_to_scalar_ylm_coefficients(
    const Scalar<DataVector>& coefficients, size_t l_max,
    size_t number_of_offsets,
    CoefficientNormalization coefficient_normalization);

}  // namespace ylm::TensorYlm
