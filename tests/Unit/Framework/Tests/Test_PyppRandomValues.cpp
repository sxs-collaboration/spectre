// Distributed under the MIT License.
// See LICENSE.txt for detai

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace {
template <typename T>
void check_single_not_null0(const gsl::not_null<T*> result, const T& t0) {
  if constexpr (tt::is_a_v<Tensor, T>) {
    get(*result) = get(t0) + 5.0;
  } else {
    *result = t0 + 5.0;
  }
}

template <typename T>
void check_single_not_null1(const gsl::not_null<T*> result, const T& t0,
                            const T& t1) {
  if constexpr (tt::is_a_v<Tensor, T>) {
    get(*result) = get(t0) + get(t1);
  } else {
    *result = t0 + t1;
  }
}

template <typename T>
void check_single_not_null2(const gsl::not_null<T*> result, const T& t0,
                            const T& t1) {
  if constexpr (tt::is_a_v<Tensor, T>) {
    get(*result) = sqrt(get(t0)) + 1.0 / sqrt(-get(t1));
  } else {
    *result = sqrt(t0) + 1.0 / sqrt(-t1);
  }
}

template <typename T>
void check_single_not_null3(const gsl::not_null<T*> result, const T& t0) {
  // This function _adds_ to the result tensor, instead of assigning to it
  if constexpr (tt::is_a_v<Tensor, T>) {
    get(*result) += get(t0) + 3.0;
  } else {
    *result += t0 + 3.0;
  }
}

template <typename T>
void check_double_not_null0(const gsl::not_null<T*> result0,
                            const gsl::not_null<T*> result1, const T& t0) {
  if constexpr (tt::is_a_v<Tensor, T>) {
    get(*result0) = get(t0) + 5.0;
    get(*result1) = 2.0 * get(t0) + 5.0;
  } else {
    *result0 = t0 + 5.0;
    *result1 = 2.0 * t0 + 5.0;
  }
}

template <typename T>
void check_double_not_null1(const gsl::not_null<T*> result0,
                            const gsl::not_null<T*> result1, const T& t0,
                            const T& t1) {
  if constexpr (tt::is_a_v<Tensor, T>) {
    get(*result0) = get(t0) + get(t1);
    get(*result1) = 2.0 * get(t0) + get(t1);
  } else {
    *result0 = t0 + t1;
    *result1 = 2.0 * t0 + t1;
  }
}

template <typename T>
void check_double_not_null2(const gsl::not_null<T*> result0,
                            const gsl::not_null<T*> result1, const T& t0,
                            const T& t1) {
  if constexpr (tt::is_a_v<Tensor, T>) {
    get(*result0) = sqrt(get(t0)) + 1.0 / sqrt(-get(t1));
    get(*result1) = 2.0 * get(t0) + get(t1);
  } else {
    *result0 = sqrt(t0) + 1.0 / sqrt(-t1);
    *result1 = 2.0 * t0 + t1;
  }
}

template <typename T>
T check_by_value0(const T& t0) {
  if constexpr (tt::is_a_v<Tensor, T>) {
    return T{get(t0) + 5.0};
  } else {
    return t0 + 5.0;
  }
}

template <typename T>
T check_by_value1(const T& t0, const T& t1) {
  if constexpr (tt::is_a_v<Tensor, T>) {
    return T{get(t0) + get(t1)};
  } else {
    return t0 + t1;
  }
}

template <typename T>
T check_by_value2(const T& t0, const T& t1) {
  if constexpr (tt::is_a_v<Tensor, T>) {
    return T{sqrt(get(t0)) + 1.0 / sqrt(-get(t1))};
  } else {
    return sqrt(t0) + 1.0 / sqrt(-t1);
  }
}

class RandomValuesTests {
 public:
  RandomValuesTests(const double a, const double b,
                    const std::array<double, 3>& c)
      : a_(a), b_(b), c_(c) {}

  // by value, single argument
  template <typename T>
  T check_by_value0(const T& t0) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      return T{get(t0) + 5.0};
    } else {
      return t0 + 5.0;
    }
  }

  template <typename T>
  T check_by_value1(const T& t0) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      return T{get(t0) + 5.0 * a_};
    } else {
      return t0 + 5.0 * a_;
    }
  }

  template <typename T>
  T check_by_value2(const T& t0) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      return T{get(t0) + 5.0 * a_ + b_};
    } else {
      return t0 + 5.0 * a_ + b_;
    }
  }

  template <typename T>
  T check_by_value3(const T& t0) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      return T{get(t0) + 5.0 * a_ + b_ + c_[0] - 2.0 * c_[1] - c_[2]};
    } else {
      return t0 + 5.0 * a_ + b_ + c_[0] - 2.0 * c_[1] - c_[2];
    }
  }

  // by value, two arguments
  template <typename T>
  T check2_by_value0(const T& t0, const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      return T{get(t0) + get(t1)};
    } else {
      return t0 + t1;
    }
  }

  template <typename T>
  T check2_by_value1(const T& t0, const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      return T{get(t0) + get(t1) + 5.0 * a_};
    } else {
      return t0 + t1 + 5.0 * a_;
    }
  }

  template <typename T>
  T check2_by_value2(const T& t0, const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      return T{get(t0) + 5.0 * a_ + b_ * get(t1)};
    } else {
      return t0 + 5.0 * a_ + t1 * b_;
    }
  }

  template <typename T>
  T check2_by_value3(const T& t0, const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      return T{get(t0) * c_[0] + 5.0 * a_ + b_ * get(t1) + c_[1] - c_[2]};
    } else {
      return t0 * c_[0] + 5.0 * a_ + t1 * b_ + c_[1] - c_[2];
    }
  }

  // single not_null, single argument
  template <typename T>
  void check_by_not_null0(const gsl::not_null<T*> result0, const T& t0) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      get(*result0) = get(t0) + 5.0;
    } else {
      *result0 = t0 + 5.0;
    }
  }

  template <typename T>
  void check_by_not_null1(const gsl::not_null<T*> result0, const T& t0) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      get(*result0) = get(t0) + 5.0 * a_;
    } else {
      *result0 = t0 + 5.0 * a_;
    }
  }

  template <typename T>
  void check_by_not_null2(const gsl::not_null<T*> result0, const T& t0) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      get(*result0) = get(t0) + 5.0 * a_ + b_;
    } else {
      *result0 = t0 + 5.0 * a_ + b_;
    }
  }

  template <typename T>
  void check_by_not_null3(const gsl::not_null<T*> result0, const T& t0) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      get(*result0) = get(t0) + 5.0 * a_ + b_ + c_[0] - 2.0 * c_[1] - c_[2];
    } else {
      *result0 = t0 + 5.0 * a_ + b_ + c_[0] - 2.0 * c_[1] - c_[2];
    }
  }

  // by value, two arguments
  template <typename T>
  void check2_by_not_null0(const gsl::not_null<T*> result0, const T& t0,
                           const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      *result0 = T{get(t0) + get(t1)};
    } else {
      *result0 = t0 + t1;
    }
  }

  template <typename T>
  void check2_by_not_null1(const gsl::not_null<T*> result0, const T& t0,
                           const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      *result0 = T{get(t0) + get(t1) + 5.0 * a_};
    } else {
      *result0 = t0 + t1 + 5.0 * a_;
    }
  }

  template <typename T>
  void check2_by_not_null2(const gsl::not_null<T*> result0, const T& t0,
                           const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      *result0 = T{get(t0) + 5.0 * a_ + b_ * get(t1)};
    } else {
      *result0 = t0 + 5.0 * a_ + t1 * b_;
    }
  }

  template <typename T>
  void check2_by_not_null3(const gsl::not_null<T*> result0, const T& t0,
                           const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      *result0 = T{get(t0) * c_[0] + 5.0 * a_ + b_ * get(t1) + c_[1] - c_[2]};
    } else {
      *result0 = t0 * c_[0] + 5.0 * a_ + t1 * b_ + c_[1] - c_[2];
    }
  }

  // by value, two arguments
  template <typename T>
  void check3_by_not_null0(const gsl::not_null<T*> result0,
                           const gsl::not_null<T*> result1, const T& t0,
                           const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      *result0 = T{get(t0) + get(t1)};
      *result1 = T{2.0 * get(t0) + get(t1)};
    } else {
      *result0 = t0 + t1;
      *result1 = 2.0 * t0 + t1;
    }
  }

  template <typename T>
  void check3_by_not_null1(const gsl::not_null<T*> result0,
                           const gsl::not_null<T*> result1, const T& t0,
                           const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      *result0 = T{get(t0) + get(t1) + 5.0 * a_};
      *result1 = T{2.0 * get(t0) + get(t1) + 5.0 * a_};
    } else {
      *result0 = t0 + t1 + 5.0 * a_;
      *result1 = 2.0 * t0 + t1 + 5.0 * a_;
    }
  }

  template <typename T>
  void check3_by_not_null2(const gsl::not_null<T*> result0,
                           const gsl::not_null<T*> result1, const T& t0,
                           const T& t1) const {
    if constexpr (tt::is_a_v<Tensor, T>) {
      *result0 = T{get(t0) + 5.0 * a_ + b_ * get(t1)};
      *result1 = T{2.0 * get(t0) + 5.0 * a_ + b_ * get(t1)};
    } else {
      *result0 = t0 + 5.0 * a_ + t1 * b_;
      *result1 = 2.0 * t0 + 5.0 * a_ + t1 * b_;
    }
  }

 private:
  double a_ = 0;
  double b_ = 0;
  std::array<double, 3> c_{};
};

template <typename T>
void test_free(const T& value) {
  pypp::check_with_random_values<1>(&check_single_not_null0<T>, "PyppPyTests",
                                    {"check_single_not_null0"},
                                    {{{-10.0, 10.0}}}, value);
  pypp::check_with_random_values<1>(&check_single_not_null1<T>, "PyppPyTests",
                                    {"check_single_not_null1"},
                                    {{{-10.0, 10.0}}}, value);
  pypp::check_with_random_values<2>(&check_single_not_null2<T>, "PyppPyTests",
                                    {"check_single_not_null2"},
                                    {{{0.0, 10.0}, {-10.0, 0.0}}}, value);
  pypp::check_with_random_values<1>(&check_single_not_null3<T>, "PyppPyTests",
                                    {"check_single_not_null0"},
                                    {{{-10.0, 10.0}}}, value, 1.e-12, {}, 2.);

  pypp::check_with_random_values<1>(
      &check_double_not_null0<T>, "PyppPyTests",
      {"check_double_not_null0_result0", "check_double_not_null0_result1"},
      {{{-10.0, 10.0}}}, value);
  pypp::check_with_random_values<1>(
      &check_double_not_null1<T>, "PyppPyTests",
      {"check_double_not_null1_result0", "check_double_not_null1_result1"},
      {{{-10.0, 10.0}}}, value);
  // [cxx_two_not_null]
  pypp::check_with_random_values<2>(
      &check_double_not_null2<T>, "PyppPyTests",
      {"check_double_not_null2_result0", "check_double_not_null2_result1"},
      {{{0.0, 10.0}, {-10.0, 0.0}}}, value);
  // [cxx_two_not_null]

  pypp::check_with_random_values<1>(&check_by_value0<T>, "PyppPyTests",
                                    "check_by_value0", {{{-10.0, 10.0}}},
                                    value);
  pypp::check_with_random_values<1>(&check_by_value1<T>, "PyppPyTests",
                                    "check_by_value1", {{{-10.0, 10.0}}},
                                    value);
  pypp::check_with_random_values<2>(&check_by_value2<T>, "PyppPyTests",
                                    "check_by_value2",
                                    {{{0.0, 10.0}, {-10.0, 0.0}}}, value);
}

template <typename T>
void test_member_functions(const T& value) {
  const double a = 3.1;
  const double b = 7.24;
  const std::array<double, 3> c{{4.23, -8.3, 5.4}};
  const RandomValuesTests test_class{a, b, c};

  // by value, single argument
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check_by_value0<T>, test_class, "PyppPyTests",
      "check_by_value0", {{{-10.0, 10.0}}}, std::make_tuple(), value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check_by_value1<T>, test_class, "PyppPyTests",
      "check_by_value1_class", {{{-10.0, 10.0}}}, std::make_tuple(a), value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check_by_value2<T>, test_class, "PyppPyTests",
      "check_by_value2_class", {{{-10.0, 10.0}}}, std::make_tuple(a, b), value);
  pypp::check_with_random_values<1>(&RandomValuesTests::check_by_value3<T>,
                                    test_class, "PyppPyTests",
                                    "check_by_value3_class", {{{-10.0, 10.0}}},
                                    std::make_tuple(a, b, c), value);

  // by value, two arguments
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check2_by_value0<T>, test_class, "PyppPyTests",
      "check_by_value1", {{{-10.0, 10.0}}}, std::make_tuple(), value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check2_by_value1<T>, test_class, "PyppPyTests",
      "check2_by_value1_class", {{{-10.0, 10.0}}}, std::make_tuple(a), value);
  pypp::check_with_random_values<1>(&RandomValuesTests::check2_by_value2<T>,
                                    test_class, "PyppPyTests",
                                    "check2_by_value2_class", {{{-10.0, 10.0}}},
                                    std::make_tuple(a, b), value);
  pypp::check_with_random_values<1>(&RandomValuesTests::check2_by_value3<T>,
                                    test_class, "PyppPyTests",
                                    "check2_by_value3_class", {{{-10.0, 10.0}}},
                                    std::make_tuple(a, b, c), value);

  // Single not_null, single argument
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check_by_not_null0<T>, test_class, "PyppPyTests"s,
      {"check_by_value0"s}, {{{-10.0, 10.0}}}, std::make_tuple(), value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check_by_not_null1<T>, test_class, "PyppPyTests",
      {"check_by_value1_class"}, {{{-10.0, 10.0}}}, std::make_tuple(a), value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check_by_not_null2<T>, test_class, "PyppPyTests",
      {"check_by_value2_class"}, {{{-10.0, 10.0}}}, std::make_tuple(a, b),
      value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check_by_not_null3<T>, test_class, "PyppPyTests",
      {"check_by_value3_class"}, {{{-10.0, 10.0}}}, std::make_tuple(a, b, c),
      value);

  // Single not_null, two arguments
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check2_by_not_null0<T>, test_class, "PyppPyTests",
      {"check_by_value1"}, {{{-10.0, 10.0}}}, std::make_tuple(), value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check2_by_not_null1<T>, test_class, {"PyppPyTests"},
      {"check2_by_value1_class"}, {{{-10.0, 10.0}}}, std::make_tuple(a), value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check2_by_not_null2<T>, test_class, {"PyppPyTests"},
      {"check2_by_value2_class"}, {{{-10.0, 10.0}}}, std::make_tuple(a, b),
      value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check2_by_not_null3<T>, test_class, {"PyppPyTests"},
      {"check2_by_value3_class"}, {{{-10.0, 10.0}}}, std::make_tuple(a, b, c),
      value);

  // Double not_null, two arguments
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check3_by_not_null0<T>, test_class, "PyppPyTests",
      {"check_double_not_null1_result0", "check_double_not_null1_result1"},
      {{{-10.0, 10.0}}}, std::make_tuple(), value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check3_by_not_null1<T>, test_class, {"PyppPyTests"},
      {"check2_by_value1_class", "check2_by_value1_class1"}, {{{-10.0, 10.0}}},
      std::make_tuple(a), value);
  pypp::check_with_random_values<1>(
      &RandomValuesTests::check3_by_not_null2<T>, test_class, {"PyppPyTests"},
      {"check2_by_value2_class", "check2_by_value2_class1"}, {{{-10.0, 10.0}}},
      std::make_tuple(a, b), value);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Pypp.CheckRandomValues", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Framework/Tests/"};
  constexpr size_t size = 5;
  const DataVector dv(size);
  const double doub(0.);
  const Scalar<double> scalar_double{5.0};
  const Scalar<DataVector> scalar_dv{size};
  test_free(doub);
  test_free(dv);
  test_free(scalar_double);
  test_free(scalar_dv);

  // Test member functions
  test_member_functions(doub);
  test_member_functions(dv);
  test_member_functions(scalar_double);
  test_member_functions(scalar_dv);
}
