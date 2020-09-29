// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
struct MyScalar : db::SimpleTag {
  using type = Scalar<DataVector>;
};
template <size_t Dim, typename Frame>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame>;
};
template <size_t Dim, typename Frame>
struct Covector : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame>;
};
template <size_t Dim, typename Frame>
struct Metric : db::SimpleTag {
  using type = tnsr::ii<DataVector, Dim, Frame>;
};
template <size_t Dim, typename Frame>
struct InverseMetric : db::SimpleTag {
  using type = tnsr::II<DataVector, Dim, Frame>;
};

template <typename Frame>
void test_l2_norm_tag() {
  constexpr size_t npts = 5;

  const DataVector one(npts, 1.);
  const DataVector two(npts, 2.);
  const DataVector minus_three(npts, -3.);
  const DataVector four(npts, 4.);
  const DataVector minus_five(npts, -5.);
  const DataVector twelve(npts, 12.);
  const auto mixed = []() {
    DataVector tmp(npts, 0.);
    for (size_t i = 0; i < npts; ++i) {
      if (i % 2) {
        tmp[i] = 3.5;
      } else {
        tmp[i] = 4.5;
      }
    }
    return tmp;
  }();

  // create test tensors
  const auto psi_1d = [&npts, &minus_five]() {
    tnsr::ii<DataVector, 1, Frame> tmp{npts};
    get<0, 0>(tmp) = minus_five;
    return tmp;
  }();
  const auto psi_2d = [&npts, &one, &two]() {
    tnsr::ii<DataVector, 2, Frame> tmp{npts};
    get<0, 0>(tmp) = two;
    get<0, 1>(tmp) = two;
    get<1, 0>(tmp) = two;
    get<1, 1>(tmp) = one;
    return tmp;
  }();
  const auto psi_3d = [&npts, &one, &two, &minus_three]() {
    tnsr::ii<DataVector, 3, Frame> tmp{npts};
    get<0, 0>(tmp) = one;
    get<0, 1>(tmp) = two;
    get<0, 2>(tmp) = minus_three;
    get<1, 1>(tmp) = two;
    get<1, 2>(tmp) = two;
    get<2, 2>(tmp) = one;
    return tmp;
  }();

  const auto box = db::create<
      db::AddSimpleTags<MyScalar, Vector<1, Frame>, Vector<2, Frame>,
                        Vector<3, Frame>, Covector<1, Frame>,
                        Covector<2, Frame>, Covector<3, Frame>,
                        Metric<1, Frame>, Metric<2, Frame>, Metric<3, Frame>,
                        InverseMetric<1, Frame>, InverseMetric<2, Frame>,
                        InverseMetric<3, Frame>>,
      db::AddComputeTags<Tags::PointwiseL2NormCompute<MyScalar>,
                         Tags::PointwiseL2NormCompute<Vector<1, Frame>>,
                         Tags::PointwiseL2NormCompute<Vector<2, Frame>>,
                         Tags::PointwiseL2NormCompute<Vector<3, Frame>>,
                         Tags::PointwiseL2NormCompute<Covector<1, Frame>>,
                         Tags::PointwiseL2NormCompute<Covector<2, Frame>>,
                         Tags::PointwiseL2NormCompute<Covector<3, Frame>>,
                         Tags::PointwiseL2NormCompute<Metric<1, Frame>>,
                         Tags::PointwiseL2NormCompute<Metric<2, Frame>>,
                         Tags::PointwiseL2NormCompute<Metric<3, Frame>>,
                         Tags::PointwiseL2NormCompute<InverseMetric<1, Frame>>,
                         Tags::PointwiseL2NormCompute<InverseMetric<2, Frame>>,
                         Tags::PointwiseL2NormCompute<InverseMetric<3, Frame>>,
                         Tags::L2NormCompute<MyScalar>,
                         Tags::L2NormCompute<Vector<1, Frame>>,
                         Tags::L2NormCompute<Vector<2, Frame>>,
                         Tags::L2NormCompute<Vector<3, Frame>>,
                         Tags::L2NormCompute<Covector<1, Frame>>,
                         Tags::L2NormCompute<Covector<2, Frame>>,
                         Tags::L2NormCompute<Covector<3, Frame>>,
                         Tags::L2NormCompute<Metric<1, Frame>>,
                         Tags::L2NormCompute<Metric<2, Frame>>,
                         Tags::L2NormCompute<Metric<3, Frame>>,
                         Tags::L2NormCompute<InverseMetric<1, Frame>>,
                         Tags::L2NormCompute<InverseMetric<2, Frame>>,
                         Tags::L2NormCompute<InverseMetric<3, Frame>>>>(
      Scalar<DataVector>{{{minus_three}}},
      tnsr::I<DataVector, 1, Frame>{{{mixed}}},
      tnsr::I<DataVector, 2, Frame>{{{minus_three, mixed}}},
      tnsr::I<DataVector, 3, Frame>{{{minus_three, mixed, four}}},
      tnsr::i<DataVector, 1, Frame>{{{four}}},
      tnsr::i<DataVector, 2, Frame>{{{four, two}}},
      tnsr::i<DataVector, 3, Frame>{{{four, two, twelve}}}, psi_1d, psi_2d,
      psi_3d, determinant_and_inverse(psi_1d).second,
      determinant_and_inverse(psi_2d).second,
      determinant_and_inverse(psi_3d).second);

  // Test point-wise L2-norm against precomputed values
  // rank 0
  CHECK_ITERABLE_APPROX(get(db::get<Tags::PointwiseL2Norm<MyScalar>>(box)),
                        DataVector(npts, 3.));
  // rank (1, 0)
  const auto verification_vec_1d = []() {
    DataVector tmp(npts, 0.);
    for (size_t i = 0; i < npts; ++i) {
      if (i % 2) {
        tmp[i] = 3.5;
      } else {
        tmp[i] = 4.5;
      }
    }
    return tmp;
  }();
  const auto verification_vec_2d = []() {
    DataVector tmp(npts, 0.);
    for (size_t i = 0; i < npts; ++i) {
      if (i % 2) {
        tmp[i] = 4.6097722286464435;
      } else {
        tmp[i] = 5.408326913195984;
      }
    }
    return tmp;
  }();
  const auto verification_vec_3d = []() {
    DataVector tmp(npts, 0.);
    for (size_t i = 0; i < npts; ++i) {
      if (i % 2) {
        tmp[i] = 6.103277807866851;
      } else {
        tmp[i] = 6.726812023536855;
      }
    }
    return tmp;
  }();
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Vector<1, Frame>>>(box)),
      verification_vec_1d);
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Vector<2, Frame>>>(box)),
      verification_vec_2d);
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Vector<3, Frame>>>(box)),
      verification_vec_3d);
  // rank (0, 1)
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Covector<1, Frame>>>(box)),
      DataVector(npts, 4.));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Covector<2, Frame>>>(box)),
      DataVector(npts, 4.47213595499958));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Covector<3, Frame>>>(box)),
      DataVector(npts, 12.806248474865697));
  // rank (0, 2)
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Metric<1, Frame>>>(box)),
      DataVector(npts, 5.));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Metric<2, Frame>>>(box)),
      DataVector(npts, 3.605551275463989));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<Metric<3, Frame>>>(box)),
      DataVector(npts, 6.324555320336759));
  // rank (2, 0)
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<InverseMetric<1, Frame>>>(box)),
      DataVector(npts, 0.2));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<InverseMetric<2, Frame>>>(box)),
      DataVector(npts, 1.8027756377319946));
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::PointwiseL2Norm<InverseMetric<3, Frame>>>(box)),
      DataVector(npts, 0.4787135538781691));

  // Test L2-norm reduced over domain, against precomputed values
  // rank 0
  CHECK(db::get<Tags::L2Norm<MyScalar>>(box) == approx(3.));
  // rank (1, 0)
  CHECK(db::get<Tags::L2Norm<Vector<1, Frame>>>(box) ==
        approx(4.129164564412516));
  CHECK(db::get<Tags::L2Norm<Vector<2, Frame>>>(box) ==
        approx(5.103920062069938));
  CHECK(db::get<Tags::L2Norm<Vector<3, Frame>>>(box) ==
        approx(6.48459713474939));
  // rank (0, 1)
  CHECK(db::get<Tags::L2Norm<Covector<1, Frame>>>(box) == approx(4.));
  CHECK(db::get<Tags::L2Norm<Covector<2, Frame>>>(box) ==
        approx(4.47213595499958));
  CHECK(db::get<Tags::L2Norm<Covector<3, Frame>>>(box) ==
        approx(12.806248474865697));
  // rank (0, 2)
  CHECK(db::get<Tags::L2Norm<Metric<1, Frame>>>(box) == approx(5.));
  CHECK(db::get<Tags::L2Norm<Metric<2, Frame>>>(box) ==
        approx(3.605551275463989));
  CHECK(db::get<Tags::L2Norm<Metric<3, Frame>>>(box) ==
        approx(6.324555320336759));
  // rank (2, 0)
  CHECK(db::get<Tags::L2Norm<InverseMetric<1, Frame>>>(box) == approx(0.2));
  CHECK(db::get<Tags::L2Norm<InverseMetric<2, Frame>>>(box) ==
        approx(1.8027756377319946));
  CHECK(db::get<Tags::L2Norm<InverseMetric<3, Frame>>>(box) ==
        approx(0.4787135538781691));

  // Check tag names
  using Tag = MyScalar;
  TestHelpers::db::test_simple_tag<Tags::PointwiseL2Norm<Tag>>(
      "PointwiseL2Norm(MyScalar)");
  TestHelpers::db::test_compute_tag<Tags::PointwiseL2NormCompute<Tag>>(
      "PointwiseL2Norm(MyScalar)");
  TestHelpers::db::test_simple_tag<Tags::L2Norm<Tag>>("L2Norm(MyScalar)");
  TestHelpers::db::test_compute_tag<Tags::L2NormCompute<Tag>>(
      "L2Norm(MyScalar)");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.Norms",
                  "[DataStructures][Unit]") {
  test_l2_norm_tag<Frame::Grid>();
  test_l2_norm_tag<Frame::Inertial>();
}
