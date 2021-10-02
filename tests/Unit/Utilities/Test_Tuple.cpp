// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>

#include "Utilities/Gsl.hpp"
#include "Utilities/Tuple.hpp"

namespace {
// [tuple_fold_struct_defn]
template <typename T>
struct tuple_fold_plus {
  T value = 0.0;
  template <typename S>
  void operator()(const S& element) {
    value += element;
  }
};
// [tuple_fold_struct_defn]
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.tuple_fold", "[Utilities][Unit]") {
  {
    // [tuple_fold_lambda]
    const auto my_tupull = std::make_tuple(2, 7, -3.8, 20.9);
    double sum_value = 0.0;
    tuple_fold(my_tupull,
               [](const auto& element, double& state) { state += element; },
               sum_value);
    CHECK(sum_value == approx(26.1));
    // [tuple_fold_lambda]

    // [tuple_counted_fold_lambda]
    sum_value = 0.0;
    tuple_counted_fold(my_tupull,
                       [](const auto& element, size_t index, double& state) {
                         if (index != 1) {
                           state += element;
                         }
                       },
                       sum_value);
    CHECK(sum_value == approx(19.1));
    // [tuple_counted_fold_lambda]
  }
  {
    // [tuple_fold_struct]
    const auto my_tupull = std::make_tuple(2, 7, -3.8, 20.9);
    tuple_fold_plus<double> sum_value{};
    tuple_fold(my_tupull, sum_value);
    CHECK(sum_value.value == approx(26.1));
    // [tuple_fold_struct]
  }
  {
    // Check passing rvalue to tuple_fold and tuple_counted_fold works
    const auto my_tupull = std::make_tuple(2, 7, -3.8, 20.9);
    double sum_value = 0.0;
    tuple_fold(my_tupull,
               [](const auto& element, double& state, const std::string& word) {
                 state += element;
                 CHECK(word == "test sentence");
               },
               sum_value, std::string("test sentence"));
    CHECK(sum_value == approx(26.1));
    sum_value = 0.0;
    tuple_counted_fold(my_tupull,
                       [](const auto& element, const size_t /*index*/,
                          double& state, const std::string& word) {
                         state += element;
                         CHECK(word == "test sentence");
                       },
                       sum_value, std::string("test sentence"));
    CHECK(sum_value == approx(26.1));
  }
  {
    // Check passing const lvalue to tuple_fold and tuple_counted_fold works
    const auto my_tupull = std::make_tuple(2, 7, -3.8, 20.9);
    const std::string test_sentence("test sentence");
    double sum_value = 0.0;
    tuple_fold(my_tupull,
               [](const auto& element, double& state, const std::string& word) {
                 state += element;
                 CHECK(word == "test sentence");
               },
               sum_value, test_sentence);
    CHECK(sum_value == approx(26.1));
    sum_value = 0.0;
    tuple_counted_fold(my_tupull,
                       [](const auto& element, const size_t /*index*/,
                          double& state, const std::string& word) {
                         state += element;
                         CHECK(word == "test sentence");
                       },
                       sum_value, test_sentence);
    CHECK(sum_value == approx(26.1));
  }
  {
    // Check left and right fold work as expected
    const auto my_tupull = std::make_tuple(2, 7, -3.8, 20.9);
    double sum_value = 0.0;
    tuple_fold(my_tupull,
               [](const auto& element, double& state) {
                 if (state < 8.0) {
                   state += element;
                 }
               },
               sum_value);
    CHECK(sum_value == approx(9.0));
    sum_value = 0.0;
    tuple_counted_fold(
        my_tupull,
        [](const auto& element, const size_t /*index*/, double& state) {
          if (state < 8.0) {
            state += element;
          }
        },
        sum_value);
    CHECK(sum_value == approx(9.0));

    sum_value = 0.0;
    tuple_fold<true>(my_tupull,
                     [](const auto& element, double& state) {
                       if (state < 8.0) {
                         state += element;
                       }
                     },
                     sum_value);
    CHECK(sum_value == approx(20.9));
    sum_value = 0.0;
    tuple_counted_fold<true>(
        my_tupull,
        [](const auto& element, const size_t /*index*/, double& state) {
          if (state < 8.0) {
            state += element;
          }
        },
        sum_value);
    CHECK(sum_value == approx(20.9));
  }
}

namespace {
// [tuple_transform_negate]
struct negate {
  template <typename T, typename Index, typename S>
  void operator()(const T& element, Index /*index*/,
                  S& second_tuple_element) const {
    std::get<Index::value>(second_tuple_element) = -element;
  }
};
// [tuple_transform_negate]
struct negate_if_sum_less {
  template <typename T, typename Index, typename S>
  void operator()(const T& element, Index /*index*/, S& second_tuple_element,
                  const gsl::not_null<double*> state,
                  const std::string& test_sentence,
                  const std::string& test_sentence2) const {
    CHECK(test_sentence == "test sentence");
    CHECK(test_sentence2 == "test sentence2");
    if (*state < 8.0) {
      *state += element;
      std::get<Index::value>(second_tuple_element) = -element;
    } else {
      std::get<Index::value>(second_tuple_element) = element;
    }
  }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.tuple_transform", "[Utilities][Unit]") {
  // [tuple_transform]
  const auto my_tupull = std::make_tuple(2, 7, -3.8, 20.9);
  std::decay_t<decltype(my_tupull)> out_tupull;
  tuple_transform(my_tupull,
                  [](const auto& element, auto index, auto& out_tuple) {
                    constexpr size_t index_v = decltype(index)::value;
                    std::get<index_v>(out_tuple) = -element;
                  },
                  out_tupull);

  CHECK(std::get<0>(out_tupull) == -2);
  CHECK(std::get<1>(out_tupull) == -7);
  CHECK(std::get<2>(out_tupull) == 3.8);
  CHECK(std::get<3>(out_tupull) == -20.9);
  // [tuple_transform]

  // Check iterating left-to-right and right-to-left, and also passing in rvalue
  // and const lvalue references
  const std::string test_sentence2("test sentence2");
  double sum_value = 0.0;
  tuple_transform(my_tupull, negate_if_sum_less{}, out_tupull, &sum_value,
                  std::string("test sentence"), test_sentence2);
  CHECK(std::get<0>(out_tupull) == -2);
  CHECK(std::get<1>(out_tupull) == -7);
  CHECK(std::get<2>(out_tupull) == -3.8);
  CHECK(std::get<3>(out_tupull) == 20.9);
  CHECK(sum_value == approx(9.0));

  sum_value = 0.0;
  tuple_transform<true>(my_tupull, negate_if_sum_less{}, out_tupull, &sum_value,
                        std::string("test sentence"), test_sentence2);
  CHECK(std::get<0>(out_tupull) == 2);
  CHECK(std::get<1>(out_tupull) == 7);
  CHECK(std::get<2>(out_tupull) == -3.8);
  CHECK(std::get<3>(out_tupull) == -20.9);
  CHECK(sum_value == approx(20.9));
}
