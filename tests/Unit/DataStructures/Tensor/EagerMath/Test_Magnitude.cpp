// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
void test_euclidean_magnitude() {
  // Check for DataVectors
  {
    const size_t npts = 5;
    const DataVector one(npts, 1.0);
    const DataVector two(npts, 2.0);
    const DataVector minus_three(npts, -3.0);
    const DataVector four(npts, 4.0);
    const DataVector minus_five(npts, -5.0);
    const DataVector twelve(npts, 12.0);

    const tnsr::i<DataVector, 1, Frame::Grid> one_d_covector{{{two}}};
    CHECK_ITERABLE_APPROX(get(magnitude(one_d_covector)), two);

    const tnsr::i<DataVector, 1, Frame::Grid> negative_one_d_covector{
        {{minus_three}}};
    CHECK_ITERABLE_APPROX(get(magnitude(negative_one_d_covector)),
                          (DataVector{npts, 3.0}));

    const tnsr::A<DataVector, 1, Frame::Grid> one_d_vector{
        {{minus_three, four}}};
    CHECK_ITERABLE_APPROX(get(magnitude(one_d_vector)),
                          (DataVector{npts, 5.0}));

    const tnsr::I<DataVector, 2, Frame::Grid> two_d_vector{
        {{minus_five, twelve}}};
    CHECK_ITERABLE_APPROX(get(magnitude(two_d_vector)),
                          (DataVector{npts, 13.0}));

    const tnsr::i<DataVector, 3, Frame::Grid> three_d_covector{
        {{minus_three, twelve, four}}};
    CHECK_ITERABLE_APPROX(get(magnitude(three_d_covector)),
                          (DataVector{npts, 13.0}));

    const tnsr::a<DataVector, 4, Frame::Grid> five_d_covector{
        {{two, twelve, four, one, two}}};
    CHECK_ITERABLE_APPROX(get(magnitude(five_d_covector)),
                          (DataVector{npts, 13.0}));
  }
  // Check case for doubles
  {
    const tnsr::i<double, 1, Frame::Grid> one_d_covector_double{{{2.}}};
    CHECK(get(magnitude(one_d_covector_double)) == 2.);

    const tnsr::a<double, 4, Frame::Grid> five_d_covector_double{
        {{2, 12, 4, 1, 2}}};
    CHECK(get(magnitude(five_d_covector_double)) == 13.);
  }
}

void test_magnitude() {
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

    const tnsr::i<DataVector, 1, Frame::Grid> one_d_covector{{{two}}};
    const tnsr::II<DataVector, 1, Frame::Grid> inv_h = [&four]() {
      tnsr::II<DataVector, 1, Frame::Grid> tensor;
      get<0, 0>(tensor) = four;
      return tensor;
    }();

    CHECK_ITERABLE_APPROX(get(magnitude(one_d_covector, inv_h)),
                          (DataVector{npts, 4.0}));
    const tnsr::i<DataVector, 3, Frame::Grid> three_d_covector{
        {{minus_three, twelve, four}}};
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
    CHECK_ITERABLE_APPROX(get(magnitude(three_d_covector, inv_g)),
                          (DataVector{npts, sqrt(778.0)}));
  }

  {
    // Check for doubles
    const tnsr::i<double, 1, Frame::Grid> one_d_covector{2.0};
    const tnsr::II<double, 1, Frame::Grid> inv_h = []() {
      tnsr::II<double, 1, Frame::Grid> tensor{};
      get<0, 0>(tensor) = 4.0;
      return tensor;
    }();

    CHECK(get(magnitude(one_d_covector, inv_h)) == 4.0);

    const tnsr::i<double, 3, Frame::Grid> three_d_covector{{{-3.0, 12.0, 4.0}}};
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
    CHECK(get(magnitude(three_d_covector, inv_g)) == sqrt(778.0));
  }
}

struct Vector : db::SimpleTag {
  static std::string name() noexcept { return "Vector"; }
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
};
template <size_t Dim>
struct Covector : db::SimpleTag {
  static std::string name() noexcept { return "Covector"; }
  using type = tnsr::i<DataVector, Dim, Frame::Grid>;
};
struct Metric : db::SimpleTag {
  static std::string name() noexcept { return "Metric"; }
  using type = tnsr::ii<DataVector, 3, Frame::Grid>;
};
struct InverseMetric : db::SimpleTag {
  static std::string name() noexcept { return "InverseMetric"; }
  using type = tnsr::II<DataVector, 3, Frame::Grid>;
};
void test_magnitude_tags() {
  const auto box =
      db::create<db::AddSimpleTags<Vector, Covector<2>>,
                 db::AddComputeTags<Tags::EuclideanMagnitude<Vector>,
                                    Tags::EuclideanMagnitude<Covector<2>>,
                                    Tags::NormalizedCompute<Vector>,
                                    Tags::NormalizedCompute<Covector<2>>>>(
          db::item_type<Vector>({{{1., 2.}, {2., 3.}, {2., 6.}}}),
          db::item_type<Covector<2>>({{{3., 5.}, {4., 12.}}}));

  CHECK(db::get<Tags::EuclideanMagnitude<Vector>>(box) ==
        Scalar<DataVector>({{{3., 7.}}}));
  CHECK(db::get<Tags::EuclideanMagnitude<Covector<2>>>(box) ==
        Scalar<DataVector>({{{5., 13.}}}));
  CHECK(db::get<Tags::Normalized<Vector>>(box) ==
        db::item_type<Vector>(
            {{{1. / 3., 2. / 7.}, {2. / 3., 3. / 7.}, {2. / 3., 6. / 7.}}}));
  CHECK(db::get<Tags::Normalized<Covector<2>>>(box) ==
        db::item_type<Covector<2>>(
            {{{3. / 5., 5. / 13.}, {4. / 5., 12. / 13.}}}));

  using Tag = Vector;
  /// [magnitude_name]
  CHECK(Tags::Magnitude<Tag>::name() == "Magnitude(" + Tag::name() + ")");
  /// [magnitude_name]
  /// [normalized_name]
  CHECK(Tags::Normalized<Tag>::name() == "Normalized(" + Tag::name() + ")");
  /// [normalized_name]
}

void test_general_magnitude_tags() {
  constexpr size_t npts = 5;
  const tnsr::i<DataVector, 3, Frame::Grid> covector{
      {{DataVector{npts, -3.0}, DataVector{npts, 12.0},
        DataVector{npts, 4.0}}}};
  const tnsr::II<DataVector, 3, Frame::Grid> inv_metric = []() noexcept {
    auto tensor = make_with_value<tnsr::II<DataVector, 3, Frame::Grid>>(
        DataVector{npts}, 0.0);
    get<0, 0>(tensor) = 2.0;
    get<0, 1>(tensor) = -3.0;
    get<0, 2>(tensor) = 4.0;
    get<1, 1>(tensor) = -5.0;
    get<1, 2>(tensor) = 12.0;
    get<2, 2>(tensor) = 13.0;
    return tensor;
  }
  ();

  const tnsr::I<DataVector, 3, Frame::Grid> vector{
      {{DataVector{npts, -3.0}, DataVector{npts, 12.0},
        DataVector{npts, 4.0}}}};
  const tnsr::ii<DataVector, 3, Frame::Grid> metric = []() noexcept {
    auto tensor = make_with_value<tnsr::ii<DataVector, 3, Frame::Grid>>(
        DataVector{npts}, 0.0);
    get<0, 0>(tensor) = 2.0;
    get<0, 1>(tensor) = -3.0;
    get<0, 2>(tensor) = 4.0;
    get<1, 1>(tensor) = -5.0;
    get<1, 2>(tensor) = 12.0;
    get<2, 2>(tensor) = 13.0;
    return tensor;
  }
  ();

  const auto box =
      db::create<db::AddSimpleTags<Vector, Covector<3>, Metric, InverseMetric>,
                 db::AddComputeTags<
                     Tags::NonEuclideanMagnitude<Vector, Metric>,
                     Tags::NonEuclideanMagnitude<Covector<3>, InverseMetric>,
                     Tags::NormalizedCompute<Vector>,
                     Tags::NormalizedCompute<Covector<3>>>>(vector, covector,
                                                            metric, inv_metric);

  CHECK_ITERABLE_APPROX(get(db::get<Tags::Magnitude<Vector>>(box)),
                        (DataVector{npts, sqrt(778.0)}));
  CHECK_ITERABLE_APPROX(get<0>(db::get<Tags::Normalized<Vector>>(box)),
                        get<0>(vector) / sqrt(778.0));
  CHECK_ITERABLE_APPROX(get<1>(db::get<Tags::Normalized<Vector>>(box)),
                        get<1>(vector) / sqrt(778.0));
  CHECK_ITERABLE_APPROX(get<2>(db::get<Tags::Normalized<Vector>>(box)),
                        get<2>(vector) / sqrt(778.0));

  CHECK_ITERABLE_APPROX(get(db::get<Tags::Magnitude<Covector<3>>>(box)),
                        (DataVector{npts, sqrt(778.0)}));
  CHECK_ITERABLE_APPROX(get<0>(db::get<Tags::Normalized<Covector<3>>>(box)),
                        get<0>(covector) / sqrt(778.0));
  CHECK_ITERABLE_APPROX(get<1>(db::get<Tags::Normalized<Covector<3>>>(box)),
                        get<1>(covector) / sqrt(778.0));
  CHECK_ITERABLE_APPROX(get<2>(db::get<Tags::Normalized<Covector<3>>>(box)),
                        get<2>(covector) / sqrt(778.0));
}

struct MyScalar : db::SimpleTag {
  static std::string name() noexcept { return "MyScalar"; }
  using type = Scalar<DataVector>;
};
void test_root_tags() {
  constexpr size_t npts = 5;
  const Scalar<DataVector> my_scalar{DataVector{npts, -3.0}};

  const auto box =
      db::create<db::AddSimpleTags<MyScalar>,
                 db::AddComputeTags<Tags::Sqrt<MyScalar>>>(my_scalar);

  CHECK(get(db::get<Tags::Sqrt<MyScalar>>(box)) == DataVector{npts, sqrt(3.0)});

  using Tag = MyScalar;
  /// [sqrt_name]
  CHECK(Tags::Sqrt<Tag>::name() == "Sqrt(" + Tag::name() + ")");
  /// [sqrt_name]
}

void test_l2_norm_tag() {
  constexpr size_t npts = 5;
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist{-1., 1.};
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const Scalar<DataVector> my_scalar{DataVector{npts, -3.0}};
  const auto vec = make_with_random_values<tnsr::I<DataVector, 3, Frame::Grid>>(
      nn_generator, nn_dist, my_scalar);
  const auto covec =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Grid>>(
          nn_generator, nn_dist, my_scalar);
  const auto psi =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Grid>>(
          nn_generator, nn_dist, my_scalar);
  const auto invpsi =
      make_with_random_values<tnsr::II<DataVector, 3, Frame::Grid>>(
          nn_generator, nn_dist, my_scalar);

  const auto box = db::create<
      db::AddSimpleTags<MyScalar, Vector, Covector<3>, Metric, InverseMetric>,
      db::AddComputeTags<
          Tags::L2Norm<MyScalar>, Tags::L2Norm<Vector>,
          Tags::L2Norm<Covector<3>>, Tags::L2Norm<Metric>,
          Tags::L2Norm<InverseMetric>, Tags::L2Norm<MyScalar, true>,
          Tags::L2Norm<Vector, true>, Tags::L2Norm<Covector<3>, true>,
          Tags::L2Norm<Metric, true>, Tags::L2Norm<InverseMetric, true>>>(
      my_scalar, vec, covec, psi, invpsi);

  const auto local_norm_vec =
      sqrt(square(get<0>(vec)) + square(get<1>(vec)) + square(get<2>(vec)));
  const auto local_norm_covec = sqrt(
      square(get<0>(covec)) + square(get<1>(covec)) + square(get<2>(covec)));
  const auto local_norm_psi = sqrt(
      square(get<0, 0>(psi)) + square(get<0, 1>(psi)) + square(get<0, 2>(psi)) +
      square(get<1, 0>(psi)) + square(get<1, 1>(psi)) + square(get<1, 2>(psi)) +
      square(get<2, 0>(psi)) + square(get<2, 1>(psi)) + square(get<2, 2>(psi)));
  const auto local_norm_invpsi =
      sqrt(square(get<0, 0>(invpsi)) + square(get<0, 1>(invpsi)) +
           square(get<0, 2>(invpsi)) + square(get<1, 0>(invpsi)) +
           square(get<1, 1>(invpsi)) + square(get<1, 2>(invpsi)) +
           square(get<2, 0>(invpsi)) + square(get<2, 1>(invpsi)) +
           square(get<2, 2>(invpsi)));

  CHECK(get(db::get<Tags::L2Norm<MyScalar>>(box)) == DataVector{npts, 3.0});
  CHECK(get(db::get<Tags::L2Norm<Vector>>(box)) == local_norm_vec);
  CHECK(get(db::get<Tags::L2Norm<Covector<3>>>(box)) == local_norm_covec);
  CHECK(get(db::get<Tags::L2Norm<Metric>>(box)) == local_norm_psi);
  CHECK(get(db::get<Tags::L2Norm<InverseMetric>>(box)) == local_norm_invpsi);

  using PlusSquare = funcl::Plus<funcl::Identity, funcl::Square<>>;
  const double local_acc_norm_scalar =
      alg::accumulate(DataVector{npts, 3.0}, 0., PlusSquare{});
  const double local_acc_norm_vec =
      alg::accumulate(local_norm_vec, 0., PlusSquare{});
  const double local_acc_norm_covec =
      alg::accumulate(local_norm_covec, 0., PlusSquare{});
  const double local_acc_norm_psi =
      alg::accumulate(local_norm_psi, 0., PlusSquare{});
  const double local_acc_norm_invpsi =
      alg::accumulate(local_norm_invpsi, 0., PlusSquare{});

  CHECK(db::get<Tags::L2Norm<MyScalar, true>>(box) == local_acc_norm_scalar);
  CHECK(db::get<Tags::L2Norm<Vector, true>>(box) == local_acc_norm_vec);
  CHECK(db::get<Tags::L2Norm<Covector<3>, true>>(box) == local_acc_norm_covec);
  CHECK(db::get<Tags::L2Norm<Metric, true>>(box) == local_acc_norm_psi);
  CHECK(db::get<Tags::L2Norm<InverseMetric, true>>(box) ==
        local_acc_norm_invpsi);

  using Tag = MyScalar;
  /// [l2norm_name]
  CHECK(Tags::L2Norm<Tag>::name() == "L2Norm(" + Tag::name() + ")");
  /// [l2norm_name]
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.Magnitude",
                  "[DataStructures][Unit]") {
  test_euclidean_magnitude();
  test_magnitude();
  test_magnitude_tags();
  test_general_magnitude_tags();
  test_root_tags();
  test_l2_norm_tag();
}
