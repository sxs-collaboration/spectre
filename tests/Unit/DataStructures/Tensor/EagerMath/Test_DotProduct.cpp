// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace {
// All overloads give the same results in a Kokkos kernel as on the host
void test_on_device() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto vector_a = make_with_random_values<tnsr::I<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto vector_b = make_with_random_values<tnsr::I<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto one_form = make_with_random_values<tnsr::i<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto metric = make_with_random_values<tnsr::ii<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    Scalar<double> euclidean{};
    dot_product(make_not_null(&euclidean), vector_a, vector_b);
    Scalar<double> vector_and_one_form{};
    dot_product(make_not_null(&vector_and_one_form), vector_a, one_form);
    Scalar<double> with_metric{};
    dot_product(make_not_null(&with_metric), vector_a, vector_b, metric);
    // Passing the same vector twice takes a different code path
    Scalar<double> squared{};
    dot_product(make_not_null(&squared), vector_a, vector_a, metric);
    return Kokkos::Array<double, 8>{
        get(euclidean),
        get(dot_product(vector_a, vector_b)),
        get(vector_and_one_form),
        get(dot_product(vector_a, one_form)),
        get(with_metric),
        get(dot_product(vector_a, vector_b, metric)),
        get(squared),
        get(dot_product(vector_a, vector_a, metric))};
  });
}
}  // namespace
#endif  // SPECTRE_KOKKOS

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.EuclideanDotProduct",
                  "[DataStructures][Unit]") {
  // Check for DataVectors
  {
    const size_t npts = 5;
    const DataVector one(npts, 1.0);
    const DataVector two(npts, 2.0);
    const DataVector minus_three(npts, -3.0);
    const DataVector four(npts, 4.0);
    const DataVector minus_five(npts, -5.0);
    const DataVector twelve(npts, 12.0);

    const tnsr::i<DataVector, 1, Frame::Grid> one_d_covector_a{{{two}}};
    const tnsr::i<DataVector, 1, Frame::Grid> one_d_covector_b{{{four}}};
    const tnsr::I<DataVector, 1, Frame::Grid> one_d_vector_c{{{four}}};
    CHECK_ITERABLE_APPROX(get(dot_product(one_d_covector_a, one_d_covector_b)),
                          DataVector(npts, 8.0));
    CHECK_ITERABLE_APPROX(get(dot_product(one_d_covector_a, one_d_vector_c)),
                          DataVector(npts, 8.0));

    const tnsr::i<DataVector, 1, Frame::Grid> negative_one_d_covector{
        {{minus_three}}};
    const tnsr::I<DataVector, 1, Frame::Grid> negative_one_d_vector{
        {{minus_three}}};
    CHECK_ITERABLE_APPROX(
        get(dot_product(negative_one_d_covector, one_d_covector_b)),
        DataVector(npts, -12.0));
    CHECK_ITERABLE_APPROX(
        get(dot_product(negative_one_d_covector, one_d_vector_c)),
        DataVector(npts, -12.0));

    const tnsr::A<DataVector, 1, Frame::Grid> one_d_vector_a{
        {{minus_three, four}}};
    const tnsr::A<DataVector, 1, Frame::Grid> one_d_vector_b{{{two, twelve}}};
    const tnsr::a<DataVector, 1, Frame::Grid> one_d_covector_c{{{two, twelve}}};
    CHECK_ITERABLE_APPROX(get(dot_product(one_d_vector_a, one_d_vector_b)),
                          DataVector(npts, 42.0));
    CHECK_ITERABLE_APPROX(get(dot_product(one_d_vector_a, one_d_covector_c)),
                          DataVector(npts, 42.0));

    const tnsr::I<DataVector, 2, Frame::Grid> two_d_vector_a{
        {{minus_five, twelve}}};
    const tnsr::I<DataVector, 2, Frame::Grid> two_d_vector_b{{{two, four}}};
    const tnsr::i<DataVector, 2, Frame::Grid> two_d_covector_b{{{two, four}}};
    CHECK_ITERABLE_APPROX(get(dot_product(two_d_vector_a, two_d_vector_b)),
                          DataVector(npts, 38.0));
    CHECK_ITERABLE_APPROX(get(dot_product(two_d_vector_a, two_d_covector_b)),
                          DataVector(npts, 38.0));

    const tnsr::i<DataVector, 3, Frame::Grid> three_d_covector_a{
        {{minus_three, twelve, four}}};
    const tnsr::i<DataVector, 3, Frame::Grid> three_d_covector_b{
        {{four, minus_five, two}}};
    const tnsr::I<DataVector, 3, Frame::Grid> three_d_vector_b{
        {{four, minus_five, two}}};
    CHECK_ITERABLE_APPROX(
        get(dot_product(three_d_covector_a, three_d_covector_b)),
        DataVector(npts, -64.0));
    CHECK_ITERABLE_APPROX(
        get(dot_product(three_d_covector_a, three_d_vector_b)),
        DataVector(npts, -64.0));

    // 5D example
    const tnsr::a<DataVector, 4, Frame::Grid> five_d_covector_a{
        {{two, twelve, four, one, two}}};
    const tnsr::a<DataVector, 4, Frame::Grid> five_d_covector_b{
        {{minus_five, minus_three, two, four, one}}};
    const tnsr::A<DataVector, 4, Frame::Grid> five_d_vector_b{
        {{minus_five, minus_three, two, four, one}}};
    CHECK_ITERABLE_APPROX(
        get(dot_product(five_d_covector_a, five_d_covector_b)),
        DataVector(npts, -32.0));
    CHECK_ITERABLE_APPROX(
        get(dot_product(five_d_covector_a, five_d_vector_b)),
        DataVector(npts, -32.0));
  }
  // Check case for doubles
  {
    const tnsr::i<double, 1, Frame::Grid> one_d_covector_double_a{{{2.}}};
    const tnsr::i<double, 1, Frame::Grid> one_d_covector_double_b{{{4.}}};
    const tnsr::I<double, 1, Frame::Grid> one_d_vector_double_b{{{4.}}};
    CHECK(get(dot_product(one_d_covector_double_a, one_d_covector_double_b)) ==
          8.0);
    CHECK(get(dot_product(one_d_covector_double_a, one_d_vector_double_b)) ==
          8.0);

    const tnsr::a<double, 4, Frame::Grid> five_d_covector_double_a{
        {{2.0, 12.0, 4.0, 1.0, 2.0}}};
    const tnsr::a<double, 4, Frame::Grid> five_d_covector_double_b{
        {{4.0, 2.0, -4.0, 3.0, 5.0}}};
    const tnsr::A<double, 4, Frame::Grid> five_d_vector_double_b{
        {{4.0, 2.0, -4.0, 3.0, 5.0}}};
    CHECK(get(dot_product(five_d_covector_double_a,
                          five_d_covector_double_b)) == 29.0);
    CHECK(get(dot_product(five_d_covector_double_a,
                          five_d_vector_double_b)) == 29.0);
  }
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.DotProduct",
                  "[DataStructures][Unit]") {
  // Check for DataVectors
  {
    const size_t npts = 5;
    const DataVector one(npts, 1.0);
    const DataVector two(npts, 2.0);
    const DataVector minus_three(npts, -3.0);
    const DataVector four(npts, 4.0);
    const DataVector minus_five(npts, -5.0);
    const DataVector twelve(npts, 12.0);
    const DataVector thirteen(npts, 13.0);

    const tnsr::i<DataVector, 1, Frame::Grid> one_d_covector_a{{{two}}};
    const tnsr::i<DataVector, 1, Frame::Grid> one_d_covector_b{{{four}}};
    const tnsr::II<DataVector, 1, Frame::Grid> inv_h = [&four]() {
      tnsr::II<DataVector, 1, Frame::Grid> tensor;
      get<0, 0>(tensor) = four;
      return tensor;
    }();
    CHECK_ITERABLE_APPROX(
        get(dot_product(one_d_covector_a, one_d_covector_b, inv_h)),
        DataVector(npts, 32.0));

    const tnsr::i<DataVector, 3, Frame::Grid> three_d_covector_a{
        {{minus_three, twelve, four}}};
    const tnsr::i<DataVector, 3, Frame::Grid> three_d_covector_b{
        {{minus_five, four, two}}};
    const tnsr::II<DataVector, 3, Frame::Grid> inv_g =
        [&two, &minus_three, &four, &minus_five, &twelve, &thirteen]() {
          tnsr::II<DataVector, 3, Frame::Grid> tensor;
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
        DataVector(npts, 486.0));
    CHECK_ITERABLE_APPROX(
        get(dot_product(three_d_covector_a, three_d_covector_a, inv_g)),
        DataVector(npts, 778.0));
  }

  {
    // Check for doubles
    const tnsr::i<double, 1, Frame::Grid> one_d_covector_a{2.0};
    const tnsr::i<double, 1, Frame::Grid> one_d_covector_b{4.0};
    const tnsr::II<double, 1, Frame::Grid> inv_h = []() {
      tnsr::II<double, 1, Frame::Grid> tensor{};
      get<0, 0>(tensor) = 4.0;
      return tensor;
    }();

    CHECK(get(dot_product(one_d_covector_a, one_d_covector_b, inv_h)) == 32.0);

    const tnsr::i<double, 3, Frame::Grid> three_d_covector_a{
        {{-3.0, 12.0, 4.0}}};
    const tnsr::i<double, 3, Frame::Grid> three_d_covector_b{
        {{-5.0, 4.0, 2.0}}};
    const tnsr::II<double, 3, Frame::Grid> inv_g = []() {
      tnsr::II<double, 3, Frame::Grid> tensor{};
      get<0, 0>(tensor) = 2.0;
      get<0, 1>(tensor) = -3.0;
      get<0, 2>(tensor) = 4.0;
      get<1, 1>(tensor) = -5.0;
      get<1, 2>(tensor) = 12.0;
      get<2, 2>(tensor) = 13.0;
      return tensor;
    }();
    CHECK(get(dot_product(three_d_covector_a, three_d_covector_b, inv_g)) ==
          486.0);
    CHECK(get(dot_product(three_d_covector_a, three_d_covector_a, inv_g)) ==
          778.0);
  }
#ifdef SPECTRE_KOKKOS
  test_on_device();
#endif  // SPECTRE_KOKKOS
}
