// Distributed under the MIT License.
// See LICENSE.txt for details.

// \file
// Tests data-type specific properties and configuration within
// `TensorExpression`s

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ArbitraryType {};

template <typename ValueType>
using number_expression = tenex::NumberAsExpression<ValueType>;
template <typename ValueType>
using tensor_expression =
    tenex::TensorAsExpression<Scalar<ValueType>, tmpl::list<>>;

template <bool Expected, typename DataTypes>
void test_is_supported_number_datatype() {
  tmpl::for_each<DataTypes>([](auto datatype_value) {
    using datatype = tmpl::type_from<decltype(datatype_value)>;
    // Tested at compile time so other tests can use this
    static_assert(
        tenex::detail::is_supported_number_datatype_v<datatype> == Expected,
        "Test for tenex::detail::test_is_supported_number_datatype failed.");
  });
}

template <bool Expected, typename DataTypes>
void test_is_supported_tensor_datatype() {
  tmpl::for_each<DataTypes>([](auto datatype_value) {
    using datatype = tmpl::type_from<decltype(datatype_value)>;
    // Tested at compile time so other tests can use this
    static_assert(
        tenex::detail::is_supported_tensor_datatype_v<datatype> == Expected,
        "Test for tenex::detail::test_is_supported_tensor_datatype failed.");
  });
}

template <typename T, typename Expected>
void test_upcast_if_derived_vector_type() {
  // Tested at compile time so other tests can use this
  static_assert(
      std::is_same_v<
          typename tenex::detail::upcast_if_derived_vector_type<T>::type,
          Expected>,
      "Test for tenex::detail::upcast_if_derived_vector_type failed.");
}

template <typename DataType, typename ExpectedComplexDataType>
void test_get_complex_datatype() {
  CHECK(std::is_same_v<
        typename tenex::detail::get_complex_datatype<DataType>::type,
        ExpectedComplexDataType>);
}

template <typename MaybeComplexDataType, typename OtherDataType>
void test_is_complex_datatype_of(const bool expected) {
  CHECK(tenex::detail::is_complex_datatype_of_v<MaybeComplexDataType,
                                                OtherDataType> == expected);
}

template <typename LhsDataType, typename RhsDataType>
void test_is_assignable(const bool support_expected) {
  CHECK(tenex::detail::is_assignable_v<LhsDataType, RhsDataType> ==
        support_expected);
}

template <typename X1, typename X2, typename ExpectedBinOpDataType>
void test_binop_datatype_support() {
  if constexpr (not std::is_same_v<ExpectedBinOpDataType, NoSuchType>) {
    CHECK(tenex::detail::binop_datatypes_are_supported_v<X1, X2> == true);
  } else {
    CHECK(tenex::detail::binop_datatypes_are_supported_v<X1, X2> == false);
  }

  CHECK(std::is_same_v<
        typename tenex::detail::get_binop_datatype_impl<
            typename tenex::detail::upcast_if_derived_vector_type<X1>::type,
            typename tenex::detail::upcast_if_derived_vector_type<X2>::type>::
            type,
        ExpectedBinOpDataType>);
}

template <typename X1, typename X2>
void test_tensor_binop_datatypes_are_supported(const bool support_expected) {
  CHECK(
      tenex::detail::tensor_binop_datatypes_are_supported_impl<X1, X2>::value ==
      support_expected);
}

template <typename T1, typename T2>
void test_tensorexpression_binop_datatypes_are_supported(
    const bool support_expected) {
  CHECK(tenex::detail::tensorexpression_binop_datatypes_are_supported_impl<
            T1, T2>::value == support_expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.DataTypeSupport",
                  "[DataStructures][Unit]") {
  // Test which numeric types can and can't appear as terms in
  // `TensorExpression`s

  test_is_supported_number_datatype<true,
                                    tmpl::list<double, std::complex<double>>>();
  test_is_supported_number_datatype<
      false, tmpl::list<int, float, std::complex<int>, std::complex<float>,
                        DataVector, ComplexDataVector, ModalVector,
                        ComplexModalVector, ArbitraryType>>();

  // Test which types can and can't appear as a `Tensor`s data type in a
  // `TensorExpression`

  test_is_supported_tensor_datatype<
      true, tmpl::list<double, std::complex<double>, DataVector,
                       ComplexDataVector>>();
  test_is_supported_tensor_datatype<
      false, tmpl::list<int, float, std::complex<int>, std::complex<float>,
                        ModalVector, ComplexModalVector, ArbitraryType>>();

  // Test helper function that upcasts derived `VectorImpl` types to their
  // base `VectorImpl` types

  test_upcast_if_derived_vector_type<double, double>();
  test_upcast_if_derived_vector_type<int, int>();
  test_upcast_if_derived_vector_type<float, float>();
  test_upcast_if_derived_vector_type<std::complex<double>,
                                     std::complex<double>>();
  test_upcast_if_derived_vector_type<std::complex<int>, std::complex<int>>();
  test_upcast_if_derived_vector_type<std::complex<float>,
                                     std::complex<float>>();
  test_upcast_if_derived_vector_type<DataVector,
                                     VectorImpl<double, DataVector>>();
  test_upcast_if_derived_vector_type<VectorImpl<double, DataVector>,
                                     VectorImpl<double, DataVector>>();
  test_upcast_if_derived_vector_type<
      ComplexDataVector, VectorImpl<std::complex<double>, ComplexDataVector>>();
  test_upcast_if_derived_vector_type<
      VectorImpl<std::complex<double>, ComplexDataVector>,
      VectorImpl<std::complex<double>, ComplexDataVector>>();
  test_upcast_if_derived_vector_type<ModalVector,
                                     VectorImpl<double, ModalVector>>();
  test_upcast_if_derived_vector_type<VectorImpl<double, ModalVector>,
                                     VectorImpl<double, ModalVector>>();
  test_upcast_if_derived_vector_type<
      ComplexModalVector,
      VectorImpl<std::complex<double>, ComplexModalVector>>();
  test_upcast_if_derived_vector_type<
      VectorImpl<std::complex<double>, ComplexModalVector>,
      VectorImpl<std::complex<double>, ComplexModalVector>>();
  test_upcast_if_derived_vector_type<ArbitraryType, ArbitraryType>();

  // Test that we correctly get the complex-valued partner type to another type

  test_get_complex_datatype<double, std::complex<double>>();
  test_get_complex_datatype<int, std::complex<int>>();
  test_get_complex_datatype<float, std::complex<float>>();
  test_get_complex_datatype<std::complex<double>, NoSuchType>();
  test_get_complex_datatype<std::complex<int>, NoSuchType>();
  test_get_complex_datatype<std::complex<float>, NoSuchType>();
  test_get_complex_datatype<DataVector, ComplexDataVector>();
  test_get_complex_datatype<ComplexDataVector, NoSuchType>();
  test_get_complex_datatype<ArbitraryType, NoSuchType>();
  test_get_complex_datatype<NoSuchType, NoSuchType>();

  // Test whether the first data type is known to be the complex partner to
  // the second data type

  test_is_complex_datatype_of<double, double>(false);
  test_is_complex_datatype_of<double, float>(false);
  test_is_complex_datatype_of<double, std::complex<double>>(false);
  test_is_complex_datatype_of<double, std::complex<float>>(false);
  test_is_complex_datatype_of<double, DataVector>(false);
  test_is_complex_datatype_of<double, ComplexDataVector>(false);
  test_is_complex_datatype_of<double, ArbitraryType>(false);
  test_is_complex_datatype_of<double, NoSuchType>(false);

  test_is_complex_datatype_of<std::complex<double>, double>(true);
  test_is_complex_datatype_of<std::complex<double>, float>(false);
  test_is_complex_datatype_of<std::complex<double>, std::complex<double>>(
      false);
  test_is_complex_datatype_of<std::complex<double>, std::complex<float>>(false);
  test_is_complex_datatype_of<std::complex<double>, DataVector>(false);
  test_is_complex_datatype_of<std::complex<double>, ComplexDataVector>(false);
  test_is_complex_datatype_of<std::complex<double>, ArbitraryType>(false);
  test_is_complex_datatype_of<std::complex<double>, NoSuchType>(false);

  test_is_complex_datatype_of<DataVector, double>(false);
  test_is_complex_datatype_of<DataVector, float>(false);
  test_is_complex_datatype_of<DataVector, std::complex<double>>(false);
  test_is_complex_datatype_of<DataVector, std::complex<float>>(false);
  test_is_complex_datatype_of<DataVector, DataVector>(false);
  test_is_complex_datatype_of<DataVector, ComplexDataVector>(false);
  test_is_complex_datatype_of<DataVector, ArbitraryType>(false);
  test_is_complex_datatype_of<DataVector, NoSuchType>(false);

  test_is_complex_datatype_of<ComplexDataVector, double>(false);
  test_is_complex_datatype_of<ComplexDataVector, float>(false);
  test_is_complex_datatype_of<ComplexDataVector, std::complex<double>>(false);
  test_is_complex_datatype_of<ComplexDataVector, std::complex<float>>(false);
  test_is_complex_datatype_of<ComplexDataVector, DataVector>(true);
  test_is_complex_datatype_of<ComplexDataVector, ComplexDataVector>(false);
  test_is_complex_datatype_of<ComplexDataVector, ArbitraryType>(false);
  test_is_complex_datatype_of<ComplexDataVector, NoSuchType>(false);

  test_is_complex_datatype_of<NoSuchType, double>(false);
  test_is_complex_datatype_of<NoSuchType, float>(false);
  test_is_complex_datatype_of<NoSuchType, std::complex<double>>(false);
  test_is_complex_datatype_of<NoSuchType, std::complex<float>>(false);
  test_is_complex_datatype_of<NoSuchType, DataVector>(false);
  test_is_complex_datatype_of<NoSuchType, ComplexDataVector>(false);
  test_is_complex_datatype_of<NoSuchType, ArbitraryType>(false);
  test_is_complex_datatype_of<NoSuchType, NoSuchType>(false);

  // Test whether the first data type is assignable to the second data type

  test_is_assignable<double, double>(true);
  test_is_assignable<double, std::complex<double>>(false);
  test_is_assignable<double, DataVector>(false);
  test_is_assignable<double, ComplexDataVector>(false);
  test_is_assignable<double, ArbitraryType>(false);

  test_is_assignable<std::complex<double>, double>(true);
  test_is_assignable<std::complex<double>, std::complex<double>>(true);
  test_is_assignable<std::complex<double>, DataVector>(false);
  test_is_assignable<std::complex<double>, ComplexDataVector>(false);
  test_is_assignable<std::complex<double>, ArbitraryType>(false);

  test_is_assignable<DataVector, double>(true);
  test_is_assignable<DataVector, std::complex<double>>(false);
  test_is_assignable<DataVector, DataVector>(true);
  test_is_assignable<DataVector, ComplexDataVector>(false);
  test_is_assignable<DataVector, ArbitraryType>(false);

  test_is_assignable<ComplexDataVector, double>(true);
  test_is_assignable<ComplexDataVector, std::complex<double>>(true);
  test_is_assignable<ComplexDataVector, DataVector>(true);
  test_is_assignable<ComplexDataVector, ComplexDataVector>(true);
  test_is_assignable<ComplexDataVector, ArbitraryType>(false);

  test_is_assignable<ArbitraryType, double>(false);
  test_is_assignable<ArbitraryType, std::complex<double>>(false);
  test_is_assignable<ArbitraryType, DataVector>(false);
  test_is_assignable<ArbitraryType, ComplexDataVector>(false);
  // true because is_assignable does not check if the types are supported types
  test_is_assignable<ArbitraryType, ArbitraryType>(true);

  // Test the type resulting from performing a binary arithmetic operation
  // between two types

  test_binop_datatype_support<double, double, double>();
  test_binop_datatype_support<double, std::complex<double>,
                              std::complex<double>>();
  test_binop_datatype_support<double, DataVector, DataVector>();
  test_binop_datatype_support<double, ComplexDataVector, ComplexDataVector>();
  test_binop_datatype_support<double, ArbitraryType, NoSuchType>();
  test_binop_datatype_support<double, NoSuchType, NoSuchType>();

  test_binop_datatype_support<std::complex<double>, double,
                              std::complex<double>>();
  test_binop_datatype_support<std::complex<double>, std::complex<double>,
                              std::complex<double>>();
  test_binop_datatype_support<std::complex<double>, DataVector,
                              ComplexDataVector>();
  test_binop_datatype_support<std::complex<double>, ComplexDataVector,
                              ComplexDataVector>();
  test_binop_datatype_support<std::complex<double>, ArbitraryType,
                              NoSuchType>();
  test_binop_datatype_support<std::complex<double>, NoSuchType, NoSuchType>();

  test_binop_datatype_support<DataVector, double, DataVector>();
  test_binop_datatype_support<DataVector, std::complex<double>,
                              ComplexDataVector>();
  test_binop_datatype_support<DataVector, DataVector, DataVector>();
  test_binop_datatype_support<DataVector, ComplexDataVector,
                              ComplexDataVector>();
  test_binop_datatype_support<DataVector, ArbitraryType, NoSuchType>();
  test_binop_datatype_support<DataVector, NoSuchType, NoSuchType>();

  test_binop_datatype_support<ComplexDataVector, double, ComplexDataVector>();
  test_binop_datatype_support<ComplexDataVector, std::complex<double>,
                              ComplexDataVector>();
  test_binop_datatype_support<ComplexDataVector, DataVector,
                              ComplexDataVector>();
  test_binop_datatype_support<ComplexDataVector, ComplexDataVector,
                              ComplexDataVector>();
  test_binop_datatype_support<ComplexDataVector, ArbitraryType, NoSuchType>();
  test_binop_datatype_support<ComplexDataVector, NoSuchType, NoSuchType>();

  test_binop_datatype_support<ArbitraryType, double, NoSuchType>();
  test_binop_datatype_support<ArbitraryType, std::complex<double>,
                              NoSuchType>();
  test_binop_datatype_support<ArbitraryType, DataVector, NoSuchType>();
  test_binop_datatype_support<ArbitraryType, ComplexDataVector, NoSuchType>();
  // ArbitraryType is result because get_binop_datatype_impl does not check if
  // the types are supported types
  test_binop_datatype_support<ArbitraryType, ArbitraryType, ArbitraryType>();
  test_binop_datatype_support<ArbitraryType, NoSuchType, NoSuchType>();

  test_binop_datatype_support<NoSuchType, double, NoSuchType>();
  test_binop_datatype_support<NoSuchType, std::complex<double>, NoSuchType>();
  test_binop_datatype_support<NoSuchType, DataVector, NoSuchType>();
  test_binop_datatype_support<NoSuchType, ComplexDataVector, NoSuchType>();
  test_binop_datatype_support<NoSuchType, ArbitraryType, NoSuchType>();
  test_binop_datatype_support<NoSuchType, NoSuchType, NoSuchType>();

  // Test whether binary operations can be performed between two `Tensor`s with
  // the given data types

  test_tensor_binop_datatypes_are_supported<double, double>(true);
  test_tensor_binop_datatypes_are_supported<double, std::complex<double>>(true);
  test_tensor_binop_datatypes_are_supported<double, DataVector>(false);
  test_tensor_binop_datatypes_are_supported<double, ComplexDataVector>(false);
  test_tensor_binop_datatypes_are_supported<double, ArbitraryType>(false);

  test_tensor_binop_datatypes_are_supported<std::complex<double>, double>(true);
  test_tensor_binop_datatypes_are_supported<std::complex<double>,
                                            std::complex<double>>(true);
  test_tensor_binop_datatypes_are_supported<std::complex<double>, DataVector>(
      false);
  test_tensor_binop_datatypes_are_supported<std::complex<double>,
                                            ComplexDataVector>(false);
  test_tensor_binop_datatypes_are_supported<std::complex<double>,
                                            ArbitraryType>(false);

  test_tensor_binop_datatypes_are_supported<DataVector, double>(false);
  test_tensor_binop_datatypes_are_supported<DataVector, std::complex<double>>(
      false);
  test_tensor_binop_datatypes_are_supported<DataVector, DataVector>(true);
  test_tensor_binop_datatypes_are_supported<DataVector, ComplexDataVector>(
      true);
  test_tensor_binop_datatypes_are_supported<DataVector, ArbitraryType>(false);

  test_tensor_binop_datatypes_are_supported<ComplexDataVector, double>(false);
  test_tensor_binop_datatypes_are_supported<ComplexDataVector,
                                            std::complex<double>>(false);
  test_tensor_binop_datatypes_are_supported<ComplexDataVector, DataVector>(
      true);
  test_tensor_binop_datatypes_are_supported<ComplexDataVector,
                                            ComplexDataVector>(true);
  test_tensor_binop_datatypes_are_supported<ComplexDataVector, ArbitraryType>(
      false);

  test_tensor_binop_datatypes_are_supported<ArbitraryType, double>(false);
  test_tensor_binop_datatypes_are_supported<ArbitraryType,
                                            std::complex<double>>(false);
  test_tensor_binop_datatypes_are_supported<ArbitraryType, DataVector>(false);
  test_tensor_binop_datatypes_are_supported<ArbitraryType, ComplexDataVector>(
      false);
  // true because tensor_binop_datatypes_are_supported_impl does not check if
  // the types are supported types
  test_tensor_binop_datatypes_are_supported<ArbitraryType, ArbitraryType>(true);

  // Test whether binary operations can be performed between two
  // `TensorExpression`s with the given data types

  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<double>, tensor_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<double>, tensor_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<double>, tensor_expression<DataVector>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<double>, tensor_expression<ComplexDataVector>>(true);

  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<std::complex<double>>, tensor_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<std::complex<double>>,
      tensor_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<std::complex<double>>, tensor_expression<DataVector>>(
      true);
  test_tensorexpression_binop_datatypes_are_supported<
      number_expression<std::complex<double>>,
      tensor_expression<ComplexDataVector>>(true);

  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, number_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, number_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, tensor_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, tensor_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, tensor_expression<DataVector>>(false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<double>, tensor_expression<ComplexDataVector>>(false);

  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>, number_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>,
      number_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>, tensor_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>,
      tensor_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>, tensor_expression<DataVector>>(
      false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<std::complex<double>>,
      tensor_expression<ComplexDataVector>>(false);

  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, number_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, number_expression<std::complex<double>>>(
      true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, tensor_expression<double>>(false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, tensor_expression<std::complex<double>>>(
      false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, tensor_expression<DataVector>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<DataVector>, tensor_expression<ComplexDataVector>>(
      true);

  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<ComplexDataVector>, number_expression<double>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<ComplexDataVector>,
      number_expression<std::complex<double>>>(true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<ComplexDataVector>, tensor_expression<double>>(false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<ComplexDataVector>,
      tensor_expression<std::complex<double>>>(false);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<ComplexDataVector>, tensor_expression<DataVector>>(
      true);
  test_tensorexpression_binop_datatypes_are_supported<
      tensor_expression<ComplexDataVector>,
      tensor_expression<ComplexDataVector>>(true);
}
