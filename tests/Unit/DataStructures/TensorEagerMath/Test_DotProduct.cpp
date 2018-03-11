// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestingFramework.hpp"

namespace {
template <typename T>
void check_dot_product(const T& used_for_size) noexcept {
  const auto one = make_with_value<T>(used_for_size, 1.0);
  const auto two = make_with_value<T>(used_for_size, 2.0);
  const auto minus_three = make_with_value<T>(used_for_size, -3.0);
  const auto four = make_with_value<T>(used_for_size, 4.0);
  const auto minus_five = make_with_value<T>(used_for_size, -5.0);
  const auto eight = make_with_value<T>(used_for_size, 8.0);
  const auto twelve = make_with_value<T>(used_for_size, 12.0);
  const auto thirteen = make_with_value<T>(used_for_size, 13.0);

  // Check non-metric version
  {
    const tnsr::i<T, 1, Frame::Grid> one_d_covector{{{two}}};
    const tnsr::I<T, 1, Frame::Grid> one_d_vector{{{four}}};
    CHECK_ITERABLE_APPROX(get(dot_product(one_d_covector, one_d_vector)),
                          eight);
    CHECK_ITERABLE_APPROX(get(dot_product(one_d_vector, one_d_covector)),
                          eight);

    const tnsr::i<T, 1, Frame::Grid> negative_one_d_covector{{{minus_three}}};
    const tnsr::I<T, 1, Frame::Grid> negative_one_d_vector{{{minus_three}}};
    CHECK_ITERABLE_APPROX(
        get(dot_product(negative_one_d_covector, negative_one_d_vector)),
        make_with_value<T>(used_for_size, 9.0));
    CHECK_ITERABLE_APPROX(get(dot_product(negative_one_d_covector,
                                          one_d_vector)),
                          make_with_value<T>(used_for_size, -12.0));

    const tnsr::A<T, 1, Frame::Grid> one_plus_one_d_vector{
        {{minus_three, four}}};
    const tnsr::a<T, 1, Frame::Grid> one_plus_one_d_covector{{{two, twelve}}};
    CHECK_ITERABLE_APPROX(get(dot_product(one_plus_one_d_vector,
                                          one_plus_one_d_covector)),
                          make_with_value<T>(used_for_size, 42.0));

    const tnsr::I<T, 2, Frame::Grid> two_d_vector{
        {{minus_five, twelve}}};
    const tnsr::i<T, 2, Frame::Grid> two_d_covector{{{two, four}}};
    CHECK_ITERABLE_APPROX(get(dot_product(two_d_vector, two_d_covector)),
                          make_with_value<T>(used_for_size, 38.0));

    const tnsr::i<T, 3, Frame::Grid> three_d_covector{
        {{minus_three, twelve, four}}};
    const tnsr::I<T, 3, Frame::Grid> three_d_vector{
        {{four, minus_five, two}}};
    CHECK_ITERABLE_APPROX(get(dot_product(three_d_covector, three_d_vector)),
                          make_with_value<T>(used_for_size, -64.0));

    // 5D example
    const tnsr::a<T, 4, Frame::Grid> five_d_covector{
        {{two, twelve, four, one, two}}};
    const tnsr::A<T, 4, Frame::Grid> five_d_vector{
        {{minus_five, minus_three, two, four, one}}};
    CHECK_ITERABLE_APPROX(get(dot_product(five_d_covector, five_d_vector)),
                          make_with_value<T>(used_for_size, -32.0));
  }

  // Check metric versions
  {
    const tnsr::i<T, 1, Frame::Grid> one_d_covector_a{{{two}}};
    const tnsr::i<T, 1, Frame::Grid> one_d_covector_b{{{four}}};
    const tnsr::II<T, 1, Frame::Grid> inv_h = [&four]() {
      tnsr::II<T, 1, Frame::Grid> tensor;
      get<0, 0>(tensor) = four;
      return tensor;
    }();
    CHECK_ITERABLE_APPROX(
        get(dot_product(one_d_covector_a, one_d_covector_b, inv_h)),
        make_with_value<T>(used_for_size, 32.0));

    const tnsr::i<T, 3, Frame::Grid> three_d_covector_a{
        {{minus_three, twelve, four}}};
    const tnsr::i<T, 3, Frame::Grid> three_d_covector_b{
        {{minus_five, four, two}}};
    const tnsr::II<T, 3, Frame::Grid> inv_g =
        [&two, &minus_three, &four, &minus_five, &twelve, &thirteen]() {
          tnsr::II<T, 3, Frame::Grid> tensor;
          get<0, 0>(tensor) = two;
          get<0, 1>(tensor) = minus_three;
          get<0, 2>(tensor) = four;
          get<1, 1>(tensor) = minus_five;
          get<1, 2>(tensor) = twelve;
          get<2, 2>(tensor) = thirteen;
          return tensor;
        }();
    CHECK_ITERABLE_APPROX(
        get(dot_product(three_d_covector_a, three_d_covector_b, inv_g)),
        make_with_value<T>(used_for_size, 486.0));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.DotProduct",
                  "[DataStructures][Unit]") {
  check_dot_product(double{});
  check_dot_product(DataVector(5));
}
