// Distributed under the MIT License.
// See LICENSE.txt for detai

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
// [convert_arbitrary_a]
struct ClassForConversionTest {
  double a_;
  double b_;
};

struct ConvertClassForConservionTestA {
  using unpacked_container = double;
  using packed_container = ClassForConversionTest;
  using packed_type = double;

  static inline unpacked_container unpack(const packed_container t,
                                          const size_t /*grid_point_index*/) {
    return t.a_;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed_t,
                          const unpacked_container t,
                          const size_t /*grid_point_index*/) {
    packed_t->a_ = t;
  }

  static inline size_t get_size(const packed_container& /*t*/) { return 1; }
};
// [convert_arbitrary_a]

// [convert_arbitrary_b]
struct ConvertClassForConservionTestB {
  using unpacked_container = double;
  using packed_container = ClassForConversionTest;
  using packed_type = double;

  static inline unpacked_container unpack(const packed_container t,
                                          const size_t /*grid_point_index*/) {
    return t.b_;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed_t,
                          const unpacked_container t,
                          const size_t /*grid_point_index*/) {
    packed_t->b_ = t;
  }

  static inline size_t get_size(const packed_container& /*t*/) { return 1; }
};
// [convert_arbitrary_b]

void test_none() {
  pypp::call<pypp::None>("PyppPyTests", "test_none");
  CHECK_THROWS_WITH(pypp::call<pypp::None>("PyppPyTests", "test_numeric", 1, 2),
                    "Cannot convert non-None type to void.");
  CHECK_THROWS_WITH(pypp::call<std::string>("PyppPyTests", "test_none"),
                    "Cannot convert non-string type to string.");
}

void test_std_string() {
  const auto ret = pypp::call<std::string>("PyppPyTests", "test_string",
                                           std::string("test string"));
  CHECK(ret == std::string("back test string"));
  // Returns "Function returned null"
  // CHECK_THROWS_WITH(pypp::call<std::string>("PyppPyTests", "test_string",
  //                                           std::string("test string_")),
  //                   "Failed string test");
  CHECK_THROWS_WITH(pypp::call<double>("PyppPyTests", "test_string",
                                       std::string("test string")),
                    "Cannot convert non-double type to double.");
}

void test_int() {
  // [pypp_int_test]
  const auto ret = pypp::call<long>("PyppPyTests", "test_numeric", 3, 4);
  CHECK(ret == 3 * 4);
  // [pypp_int_test]
  CHECK_THROWS_WITH(pypp::call<double>("PyppPyTests", "test_numeric", 3, 4),
                    "Cannot convert non-double type to double.");
  CHECK_THROWS_WITH(pypp::call<void*>("PyppPyTests", "test_numeric", 3, 4),
                    "Cannot convert non-None type to void.");
  CHECK_THROWS_WITH(pypp::call<long>("PyppPyTests", "test_none"),
                    "Cannot convert non-long/int type to long.");
}

void test_long() {
  const auto ret = pypp::call<long>("PyppPyTests", "test_numeric", 3L, 4L);
  CHECK(ret == 3L * 4L);
  CHECK_THROWS_WITH(pypp::call<double>("PyppPyTests", "test_numeric", 3L, 4L),
                    "Cannot convert non-double type to double.");
  CHECK_THROWS_WITH(pypp::call<long>("PyppPyTests", "test_numeric", 3.0, 3.74),
                    "Cannot convert non-long/int type to long.");
}

void test_unsigned_long() {
  const auto ret =
      pypp::call<unsigned long>("PyppPyTests", "test_numeric", 3ul, 4ul);
  CHECK(ret == 3ul * 4ul);
  CHECK_THROWS_WITH(pypp::call<double>("PyppPyTests", "test_numeric", 3ul, 4ul),
                    "Cannot convert non-double type to double.");
  CHECK_THROWS_WITH(
      pypp::call<unsigned long>("PyppPyTests", "test_numeric", 3.0, 3.74),
      "Cannot convert non-long/int type to long.");
}

void test_double() {
  const auto ret =
      pypp::call<double>("PyppPyTests", "test_numeric", 3.49582, 3);
  CHECK(ret == 3.0 * 3.49582);
  CHECK_THROWS_WITH(pypp::call<long>("PyppPyTests", "test_numeric", 3.8, 3.9),
                    "Cannot convert non-long/int type to long.");
  CHECK_THROWS_WITH(pypp::call<double>("PyppPyTests", "test_numeric", 3ul, 3ul),
                    "Cannot convert non-double type to double.");
}

void test_std_vector() {
  // [pypp_vector_test]
  const auto ret = pypp::call<std::vector<double>>(
      "PyppPyTests", "test_vector", std::vector<double>{1.3, 4.9},
      std::vector<double>{4.2, 6.8});
  CHECK(approx(ret[0]) == 1.3 * 4.2);
  CHECK(approx(ret[1]) == 4.9 * 6.8);
  // [pypp_vector_test]
  CHECK_THROWS_WITH(pypp::call<std::string>("PyppPyTests", "test_vector",
                                            std::vector<double>{1.3, 4.9},
                                            std::vector<double>{4.2, 6.8}),
                    "Cannot convert non-string type to string.");
}

void test_std_array() {
  // std::arrays and std::vectors should both convert to lists in python so this
  // test calls the same python function as the vector test
  const auto ret = pypp::call<std::array<double, 2>>(
      "PyppPyTests", "test_vector", std::array<double, 2>{{1.3, 4.9}},
      std::array<double, 2>{{4.2, 6.8}});
  CHECK(approx(ret[0]) == 1.3 * 4.2);
  CHECK(approx(ret[1]) == 4.9 * 6.8);
  CHECK_THROWS_WITH(pypp::call<double>("PyppPyTests", "test_vector",
                                       std::array<double, 2>{{1.3, 4.9}},
                                       std::array<double, 2>{{4.2, 6.8}}),
                    "Cannot convert non-double type to double.");

  std::array<DataVector, 3> expected_array{
      {DataVector{2, 3.}, DataVector{2, 1.}, DataVector{2, 2.}}};

  CHECK(expected_array ==
        (pypp::call<std::array<DataVector, 3>>(
            "PyppPyTests", "permute_array",
            tnsr::i<DataVector, 3>{
                {{DataVector{2, 1.}, DataVector{2, 2.}, DataVector{2, 3.}}}})));

  CHECK(expected_array ==
        (pypp::call<std::array<DataVector, 3>>(
            "PyppPyTests", "permute_array",
            std::array<DataVector, 3>{
                {DataVector{2, 1.}, DataVector{2, 2.}, DataVector{2, 3.}}})));
}

void test_datavector() {
  const auto ret = pypp::call<DataVector>(
      "numpy", "multiply", DataVector{1.3, 4.9}, DataVector{4.2, 6.8});
  CHECK(approx(ret[0]) == 1.3 * 4.2);
  CHECK(approx(ret[1]) == 4.9 * 6.8);
  CHECK_THROWS_WITH(
      pypp::call<std::string>("numpy", "multiply", DataVector{1.3, 4.9},
                              DataVector{4.2, 6.8}),
      "Cannot convert non-string type to string.");
  CHECK_THROWS_WITH(pypp::call<DataVector>("PyppPyTests", "two_dim_ndarray"),
                    "Cannot convert array of ndim != 1 to DataVector.");
  CHECK_THROWS_WITH(pypp::call<DataVector>("PyppPyTests", "ndarray_of_floats"),
                    "Cannot convert array of non-double type to DataVector.");
}

void test_complex_datavector() {
  const std::complex<double> test_value_0{1.3, 2.2};
  const std::complex<double> test_value_1{4.0, 3.1};
  const std::complex<double> test_value_2{4.2, 5.7};
  const std::complex<double> test_value_3{6.8, 7.3};
  const auto ret = pypp::call<ComplexDataVector>(
      "numpy", "multiply", ComplexDataVector{test_value_0, test_value_1},
      ComplexDataVector{test_value_2, test_value_3});
  CHECK_ITERABLE_APPROX(ret[0], test_value_0 * test_value_2);
  CHECK_ITERABLE_APPROX(ret[1], test_value_1 * test_value_3);
  CHECK_THROWS_WITH(
      pypp::call<std::string>("numpy", "multiply",
                              ComplexDataVector{test_value_0, test_value_1},
                              ComplexDataVector{test_value_2, test_value_3}),
      "Cannot convert non-string type to string.");
  CHECK_THROWS_WITH(
      pypp::call<ComplexDataVector>("PyppPyTests", "two_dim_ndarray"),
      "Cannot convert array of non-complex type to ComplexDataVector.");
  CHECK_THROWS_WITH(
      pypp::call<ComplexDataVector>("PyppPyTests", "ndarray_of_floats"),
      "Cannot convert array of non-complex type to ComplexDataVector.");

  // test functionality of mixed complex and real values in tensors
  const size_t vector_size = 1;
  Scalar<SpinWeighted<ComplexDataVector, 1>> spin_weighted_argument{
      vector_size};
  get(spin_weighted_argument).data()[0] = test_value_1;
  tnsr::i<ComplexDataVector, 2> complex_tensor_argument{vector_size};
  get<0>(complex_tensor_argument)[0] = test_value_2;
  get<1>(complex_tensor_argument)[0] = test_value_3;
  tnsr::i<DataVector, 2> real_tensor_argument{vector_size};
  get<0>(real_tensor_argument)[0] = 1.2;
  get<1>(real_tensor_argument)[0] = 3.4;

  const auto mixed_return_1 =
      pypp::call<Scalar<SpinWeighted<ComplexDataVector, 1>>>(
          "PyppPyTests", "mixed_complex_real_function_1",
          spin_weighted_argument, complex_tensor_argument,
          real_tensor_argument);
  const auto mixed_return_2 = pypp::call<tnsr::i<DataVector, 2>>(
      "PyppPyTests", "mixed_complex_real_function_2", spin_weighted_argument,
      complex_tensor_argument, real_tensor_argument);
  CHECK(real(get(mixed_return_1).data()[0]) ==
        approx(real(test_value_1 * test_value_2 / 3.4)));
  CHECK(imag(get(mixed_return_1).data()[0]) ==
        approx(imag(test_value_1 * test_value_2 / 3.4)));

  CHECK(get<0>(mixed_return_2)[0] ==
        approx(real(test_value_1 * test_value_2 * 3.4)));
  CHECK(get<1>(mixed_return_2)[0] ==
        approx(imag(test_value_3 * 1.2 / test_value_1)));
}

void test_tensor_double() {
  const Scalar<double> scalar{0.8};
  const tnsr::A<double, 3> vector{{{3., 4., 5., 6.}}};
  const auto tnsr_ia = []() {
    tnsr::ia<double, 3> tnsr{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        tnsr.get(i, j) = i + 2. * j + 1.;
      }
    }
    return tnsr;
  }();
  const auto tnsr_AA = []() {
    tnsr::AA<double, 3> tnsr{};
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        tnsr.get(i, j) = static_cast<double>(i) + j + 1.;
      }
    }
    return tnsr;
  }();
  const auto tnsr_iaa = []() {
    tnsr::iaa<double, 3> tnsr{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        for (size_t k = 0; k < 4; ++k) {
          tnsr.get(i, j, k) = 2. * (k + 1.) * (j + 1.) + i + 1.;
        }
      }
    }
    return tnsr;
  }();
  const auto tnsr_aia = []() {
    tnsr::aia<double, 3> tnsr{};
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 4; ++k) {
          tnsr.get(i, j, k) = 2. * (k + 1.) * (i + 1.) + j + 1.5;
        }
      }
    }
    return tnsr;
  }();
  const auto tnsr_aBcc = []() {
    tnsr::aBcc<double, 3> tnsr{};
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        for (size_t k = 0; k < 4; ++k) {
          for (size_t l = 0; l < 4; ++l) {
            tnsr.get(i, j, k, l) = 3. * i + j + (k + 1.) * (l + 1.) + 1.;
          }
        }
      }
    }
    return tnsr;
  }();

  // Check converting from Tensor to ndarray
  CHECK(pypp::call<bool>("PyppPyTests", "convert_scalar_to_ndarray_successful",
                         scalar));
  CHECK(pypp::call<bool>("PyppPyTests", "convert_scalar_to_double_unsuccessful",
                         scalar));
  CHECK(pypp::call<bool>("PyppPyTests", "convert_vector_successful", vector));
  CHECK(pypp::call<bool>("PyppPyTests", "convert_tnsr_ia_successful", tnsr_ia));
  CHECK(pypp::call<bool>("PyppPyTests", "convert_tnsr_AA_successful", tnsr_AA));
  CHECK(
      pypp::call<bool>("PyppPyTests", "convert_tnsr_iaa_successful", tnsr_iaa));
  CHECK(
      pypp::call<bool>("PyppPyTests", "convert_tnsr_aia_successful", tnsr_aia));
  CHECK(pypp::call<bool>("PyppPyTests", "convert_tnsr_aBcc_successful",
                         tnsr_aBcc));

  // Check converting from ndarray to Tensor
  CHECK(scalar ==
        (pypp::call<Scalar<double>>("PyppPyTests", "scalar_from_double")));
  CHECK(scalar ==
        (pypp::call<Scalar<double>>("PyppPyTests", "scalar_from_ndarray")));
  CHECK(vector == (pypp::call<tnsr::A<double, 3>>("PyppPyTests", "vector")));
  CHECK(tnsr_ia == (pypp::call<tnsr::ia<double, 3>>("PyppPyTests", "tnsr_ia")));
  CHECK(tnsr_AA == (pypp::call<tnsr::AA<double, 3>>("PyppPyTests", "tnsr_AA")));
  CHECK(tnsr_iaa ==
        (pypp::call<tnsr::iaa<double, 3>>("PyppPyTests", "tnsr_iaa")));
  CHECK(tnsr_aia ==
        (pypp::call<tnsr::aia<double, 3>>("PyppPyTests", "tnsr_aia")));
  CHECK(tnsr_aBcc ==
        (pypp::call<tnsr::aBcc<double, 3>>("PyppPyTests", "tnsr_aBcc")));

  // Check conversion throws with incorrect rank
  CHECK_THROWS_WITH(
      (pypp::call<tnsr::i<double, 3>>("PyppPyTests", "scalar_from_ndarray")),
      "Mismatch between ndim of numpy ndarray (0) and rank of Tensor (1)");
  CHECK_THROWS_WITH(
      (pypp::call<tnsr::ij<double, 3>>("PyppPyTests", "vector")),
      "Mismatch between ndim of numpy ndarray (1) and rank of Tensor (2)");
  CHECK_THROWS_WITH(
      (pypp::call<Scalar<double>>("PyppPyTests", "tnsr_AA")),
      "Mismatch between ndim of numpy ndarray (2) and rank of Tensor (0)");
  // Check conversion throws with correct rank but incorrect dimension
  CHECK_THROWS_WITH((pypp::call<tnsr::i<double, 3>>("PyppPyTests", "vector")),
                    "Mismatch between number of components of ndarray (4) and "
                    "Tensor of rank 1 in 0'th index with dimension 3");
  CHECK_THROWS_WITH(
      (pypp::call<tnsr::iaa<double, 3>>("PyppPyTests", "tnsr_aia")),
      "Mismatch between number of components of ndarray (4) and Tensor of rank "
      "3 in 0'th index with dimension 3");
}

void test_tensor_datavector() {
  const size_t npts = 5;
  const Scalar<DataVector> scalar{DataVector(npts, 0.8)};

  const tnsr::A<DataVector, 3> vector{
      {{DataVector(npts, 3.), DataVector(npts, 4.), DataVector(npts, 5.),
        DataVector(npts, 6.)}}};

  const auto tnsr_ia = []() {
    tnsr::ia<DataVector, 3> tnsr{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        tnsr.get(i, j) = DataVector(npts, i + 2. * j + 1.);
      }
    }
    return tnsr;
  }();
  const auto tnsr_AA = []() {
    tnsr::AA<DataVector, 3> tnsr{};
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        tnsr.get(i, j) = DataVector(npts, static_cast<double>(i) + j + 1.);
      }
    }
    return tnsr;
  }();
  const auto tnsr_iaa = []() {
    tnsr::iaa<DataVector, 3> tnsr{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        for (size_t k = 0; k < 4; ++k) {
          tnsr.get(i, j, k) =
              DataVector(npts, 2. * (k + 1.) * (j + 1.) + i + 1.);
        }
      }
    }
    return tnsr;
  }();
  const auto tnsr_aia = []() {
    tnsr::aia<DataVector, 3> tnsr{};
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 4; ++k) {
          tnsr.get(i, j, k) =
              DataVector(npts, 2. * (k + 1.) * (i + 1.) + j + 1.5);
        }
      }
    }
    return tnsr;
  }();
  const auto tnsr_aBcc = []() {
    tnsr::aBcc<DataVector, 3> tnsr{};
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        for (size_t k = 0; k < 4; ++k) {
          for (size_t l = 0; l < 4; ++l) {
            tnsr.get(i, j, k, l) =
                DataVector(npts, 3. * i + j + (k + 1.) * (l + 1.) + 1.);
          }
        }
      }
    }
    return tnsr;
  }();

  CHECK(scalar ==
        (pypp::call<Scalar<DataVector>>("PyppPyTests", "identity", scalar)));
  CHECK(vector == (pypp::call<tnsr::A<DataVector, 3>>("PyppPyTests", "identity",
                                                      vector)));
  CHECK(tnsr_ia == (pypp::call<tnsr::ia<DataVector, 3>>("PyppPyTests",
                                                        "identity", tnsr_ia)));
  CHECK(tnsr_AA == (pypp::call<tnsr::AA<DataVector, 3>>("PyppPyTests",
                                                        "identity", tnsr_AA)));
  CHECK(tnsr_iaa == (pypp::call<tnsr::iaa<DataVector, 3>>(
                        "PyppPyTests", "identity", tnsr_iaa)));
  CHECK(tnsr_aia == (pypp::call<tnsr::aia<DataVector, 3>>(
                        "PyppPyTests", "identity", tnsr_aia)));
  CHECK(tnsr_aBcc == (pypp::call<tnsr::aBcc<DataVector, 3>>(
                         "PyppPyTests", "identity", tnsr_aBcc)));
}

template <typename T>
void test_einsum(const T& used_for_size) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-10., 10.);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const auto scalar =
      make_with_random_values<Scalar<T>>(nn_generator, nn_dist, used_for_size);
  const auto vector = make_with_random_values<tnsr::A<T, 3>>(
      nn_generator, nn_dist, used_for_size);
  const auto tnsr_ia = make_with_random_values<tnsr::ia<T, 3>>(
      nn_generator, nn_dist, used_for_size);
  const auto tnsr_AA = make_with_random_values<tnsr::AA<T, 3>>(
      nn_generator, nn_dist, used_for_size);
  const auto tnsr_iaa = make_with_random_values<tnsr::iaa<T, 3>>(
      nn_generator, nn_dist, used_for_size);

  const auto expected = [&scalar, &vector, &tnsr_ia, &tnsr_AA, &tnsr_iaa,
                         &used_for_size]() {
    auto tnsr = make_with_value<tnsr::i<T, 3>>(used_for_size, 0.);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < 4; ++a) {
        tnsr.get(i) += scalar.get() * vector.get(a) * tnsr_ia.get(i, a);
        for (size_t b = 0; b < 4; ++b) {
          tnsr.get(i) += tnsr_AA.get(a, b) * tnsr_iaa.get(i, a, b);
        }
      }
    }
    return tnsr;
  }();
  // [einsum_example]
  const auto tensor_from_python = pypp::call<tnsr::i<T, 3>>(
      "PyppPyTests", "test_einsum", scalar, vector, tnsr_ia, tnsr_AA, tnsr_iaa);
  // [einsum_example]
  CHECK_ITERABLE_CUSTOM_APPROX(expected, tensor_from_python,
                               approx.epsilon(2.0e-13));
}

void test_function_of_time() {
  const tnsr::i<double, 3> x_d{{{3.4, 4.2, 5.8}}};
  const tnsr::i<DataVector, 3> x_dv{
      {{DataVector(8, 3.4), DataVector(8, 4.2), DataVector(8, 5.8)}}};
  const double t = -9.2;
  const auto check = [](const auto& x, const double time) {
    CHECK((2 * x.get(0) + x.get(1) - x.get(2) - time) ==
          (pypp::call<Scalar<typename std::decay_t<decltype(x)>::value_type>>(
               "PyppPyTests", "test_function_of_time", x, time)
               .get()));
  };
  check(x_d, t);
  check(x_dv, t);
}

template <template <class> class Optional>
void test_optional() {
  const Optional<Scalar<double>> scalar_double_a{Scalar<double>{0.8}};
  const Optional<Scalar<double>> scalar_double_b{Scalar<double>{1.8}};

  const Optional<Scalar<DataVector>> scalar_datavector_a{
      Scalar<DataVector>{{{{0.8, 0.7, 0.5}}}}};
  const Optional<Scalar<DataVector>> scalar_datavector_b{
      Scalar<DataVector>{{{{1.8, 2.3, 4.2}}}}};

  const auto impl = [](const auto& a, const auto& b) {
    using T = typename std::decay_t<decltype(a)>::value_type;
    const auto result_double_double =
        pypp::call<T>("PyppPyTests", "add_scalars", a, b);
    CHECK_ITERABLE_APPROX(get(result_double_double), get(T{get(*a) + get(*b)}));

    const auto result_double_none =
        pypp::call<T>("PyppPyTests", "add_scalars", a, Optional<T>{});
    CHECK_ITERABLE_APPROX(get(result_double_none), get(*a));

    const auto result_none_double =
        pypp::call<T>("PyppPyTests", "add_scalars", Optional<T>{}, b);
    CHECK_ITERABLE_APPROX(get(result_none_double), get(*b));
  };

  impl(scalar_double_a, scalar_double_b);
  impl(scalar_datavector_a, scalar_datavector_b);
}

void test_custom_conversion() {
  const Scalar<DataVector> t{DataVector{5, 2.5}};
  {
    // [convert_arbitrary_a_call]
    const auto result = pypp::call<Scalar<DataVector>,
                                   tmpl::list<ConvertClassForConservionTestA>>(
        "PyppPyTests", "custom_conversion", t,
        ClassForConversionTest{2.0, 3.0});
    // [convert_arbitrary_a_call]
    CHECK(DataVector{5, 5.0} == get(result));
  }
  {
    const auto result = pypp::call<Scalar<DataVector>,
                                   tmpl::list<ConvertClassForConservionTestB>>(
        "PyppPyTests", "custom_conversion", t,
        ClassForConversionTest{2.0, 3.0});
    CHECK(DataVector{5, 7.5} == get(result));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Pypp", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Framework/Tests/"};
  {
    INFO("Testing scipy support");
    CHECK((pypp::call<size_t>("scipy", "ndim", tnsr::abcc<double, 3>{})) == 4);
  }

  test_none();
  test_std_string();
  test_int();
  test_long();
  test_unsigned_long();
  test_double();
  test_std_vector();
  test_std_array();
  test_datavector();
  test_complex_datavector();
  test_tensor_double();
  test_tensor_datavector();
  test_einsum<double>(0.);
  test_einsum<DataVector>(DataVector(5));
  test_function_of_time();
  test_optional<std::optional>();
  test_custom_conversion();
}
