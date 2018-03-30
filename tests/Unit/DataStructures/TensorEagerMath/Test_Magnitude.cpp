// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.EuclideanMagnitude",
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

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.Magnitude",
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

namespace {
struct Vector : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Vector";
  using type = tnsr::I<DataVector, 3, Frame::Grid>;
};
struct Covector : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Covector";
  using type = tnsr::i<DataVector, 2, Frame::Grid>;
};
}  // namespace
SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.Magnitude.Tags",
                  "[DataStructures][Unit]") {
  const auto box = db::create<
      db::AddSimpleTags<Vector, Covector>,
      db::AddComputeTags<
          Tags::EuclideanMagnitude<Vector>, Tags::EuclideanMagnitude<Covector>,
          Tags::Normalized<Vector, Tags::EuclideanMagnitude<Vector>>,
          Tags::Normalized<Covector, Tags::EuclideanMagnitude<Covector>>>>(
      db::item_type<Vector>({{{1., 2.}, {2., 3.}, {2., 6.}}}),
      db::item_type<Covector>({{{3., 5.}, {4., 12.}}}));

  CHECK(db::get<Tags::EuclideanMagnitude<Vector>>(box) ==
        Scalar<DataVector>({{{3., 7.}}}));
  CHECK(db::get<Tags::EuclideanMagnitude<Covector>>(box) ==
        Scalar<DataVector>({{{5., 13.}}}));
  CHECK(db::get<Tags::Normalized<Vector, Tags::EuclideanMagnitude<Vector>>>(
            box) ==
        db::item_type<Vector>(
            {{{1. / 3., 2. / 7.}, {2. / 3., 3. / 7.}, {2. / 3., 6. / 7.}}}));
  CHECK(db::get<Tags::Normalized<Covector, Tags::EuclideanMagnitude<Covector>>>(
            box) ==
        db::item_type<Covector>({{{3. / 5., 5. / 13.}, {4. / 5., 12. / 13.}}}));
}
