// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {

using namespace std::complex_literals;

template <typename DataType>
struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct TensorTag : db::SimpleTag {
  using type = tnsr::ii<DataType, 2>;
};

template <typename ValueType, typename... Structure,
          typename VectorType = std::conditional_t<
              std::is_same_v<ValueType, std::complex<double>>,
              ComplexDataVector, DataVector>>
Tensor<VectorType, Structure...> promote_to_data_vectors(
    const Tensor<ValueType, Structure...>& tensor) {
  Tensor<VectorType, Structure...> result(1_st);
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = tensor[i];
  }
  return result;
}

template <typename VectorType,
          typename ValueType = typename VectorType::value_type>
void test_extract_point() {
  // Test with owning Tensors as well as Variables to make sure there
  // are no assumptions about memory layout in the tensor versions.
  Scalar<VectorType> scalar(2_st);
  tnsr::ii<VectorType, 2> tensor(2_st);
  if constexpr (std::is_same_v<VectorType, ComplexDataVector>) {
    get(scalar) = ComplexDataVector{3.0 + 5.0i, 4.0 + 6.0i};
    get<0, 0>(tensor) = ComplexDataVector{1.0 + 3.0i, 2.0 + 4.0i};
    get<0, 1>(tensor) = ComplexDataVector{3.0 + 5.0i, 4.0 + 6.0i};
    get<1, 1>(tensor) = ComplexDataVector{5.0 + 7.0i, 6.0 + 8.0i};
  } else {
    get(scalar) = DataVector{3.0, 4.0};
    get<0, 0>(tensor) = DataVector{1.0, 2.0};
    get<0, 1>(tensor) = DataVector{3.0, 4.0};
    get<1, 1>(tensor) = DataVector{5.0, 6.0};
  }
  Variables<tmpl::list<ScalarTag<VectorType>, TensorTag<VectorType>>> variables(
      2);
  get<ScalarTag<VectorType>>(variables) = scalar;
  get<TensorTag<VectorType>>(variables) = tensor;

  Scalar<VectorType> reconstructed_scalar{VectorType(2)};
  tnsr::ii<VectorType, 2> reconstructed_tensor(VectorType(2));
  Scalar<VectorType> reconstructed_scalar_from_dv{VectorType(2)};
  tnsr::ii<VectorType, 2> reconstructed_tensor_from_dv(VectorType(2));
  Variables<tmpl::list<ScalarTag<VectorType>, TensorTag<VectorType>>>
      reconstructed_variables(2);

  const auto check_point = [&](const size_t index,
                               const ValueType scalar_component,
                               const std::array<ValueType, 3>&
                                   tensor_components) {
    const Scalar<ValueType> expected_scalar{scalar_component};
    tnsr::ii<ValueType, 2> expected_tensor;
    get<0, 0>(expected_tensor) = tensor_components[0];
    get<0, 1>(expected_tensor) = tensor_components[1];
    get<1, 1>(expected_tensor) = tensor_components[2];
    const Scalar<VectorType> expected_scalar_dv =
        promote_to_data_vectors(expected_scalar);
    const tnsr::ii<VectorType, 2> expected_tensor_dv =
        promote_to_data_vectors(expected_tensor);
    Variables<tmpl::list<ScalarTag<VectorType>, TensorTag<VectorType>>>
        expected_variables(1);
    get<ScalarTag<VectorType>>(expected_variables) = expected_scalar_dv;
    get<TensorTag<VectorType>>(expected_variables) = expected_tensor_dv;

    {
      Scalar<ValueType> result{};
      extract_point(make_not_null(&result), scalar, index);
      CHECK(result == expected_scalar);
    }
    {
      tnsr::ii<ValueType, 2> result{};
      extract_point(make_not_null(&result), tensor, index);
      CHECK(result == expected_tensor);
    }
    CHECK(extract_point(scalar, index) == expected_scalar);
    CHECK(extract_point(tensor, index) == expected_tensor);
    {
      Scalar<VectorType> result(1_st);
      extract_point(make_not_null(&result), scalar, index);
      CHECK(result == expected_scalar_dv);
    }
    {
      tnsr::ii<VectorType, 2> result(1_st);
      extract_point(make_not_null(&result), tensor, index);
      CHECK(result == expected_tensor_dv);
    }
    {
      Variables<tmpl::list<ScalarTag<VectorType>, TensorTag<VectorType>>>
          result(1);
      extract_point(make_not_null(&result), variables, index);
      CHECK(result == expected_variables);
    }
    CHECK(extract_point(variables, index) == expected_variables);

    overwrite_point(make_not_null(&reconstructed_scalar), expected_scalar,
                    index);
    overwrite_point(make_not_null(&reconstructed_tensor), expected_tensor,
                    index);
    overwrite_point(make_not_null(&reconstructed_scalar_from_dv),
                    expected_scalar_dv, index);
    overwrite_point(make_not_null(&reconstructed_tensor_from_dv),
                    expected_tensor_dv, index);
    overwrite_point(make_not_null(&reconstructed_variables), expected_variables,
                    index);
  };
  if constexpr (std::is_same_v<VectorType, ComplexDataVector>) {
    check_point(0, 3.0 + 5.0i, {{1.0 + 3.0i, 3.0 + 5.0i, 5.0 + 7.0i}});
    check_point(1, 4.0 + 6.0i, {{2.0 + 4.0i, 4.0 + 6.0i, 6.0 + 8.0i}});
  } else {
    check_point(0, 3.0, {{1.0, 3.0, 5.0}});
    check_point(1, 4.0, {{2.0, 4.0, 6.0}});
  }

  CHECK(reconstructed_scalar == scalar);
  CHECK(reconstructed_tensor == tensor);
  CHECK(reconstructed_scalar_from_dv == scalar);
  CHECK(reconstructed_tensor_from_dv == tensor);
  CHECK(reconstructed_variables == variables);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.ExtractPoint",
                  "[DataStructures][Unit]") {
  test_extract_point<DataVector>();
  test_extract_point<ComplexDataVector>();
}
