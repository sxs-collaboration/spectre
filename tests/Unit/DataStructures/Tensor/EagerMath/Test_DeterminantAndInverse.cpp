// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
// In the spirit of the Tensor type aliases, but for a rank-2 Tensor with each
// index in a different frame. If Fr1 == Fr2, then this reduces to tnsr::iJ.
template <typename DataType, size_t Dim, typename Fr1, typename Fr2>
using tnsr_iJ = Tensor<DataType, tmpl::integral_list<int32_t, 2, 1>,
                       index_list<SpatialIndex<Dim, UpLo::Lo, Fr1>,
                                  SpatialIndex<Dim, UpLo::Up, Fr2>>>;

template <typename TensorType>
void verify_det_and_inv_1d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == 2.0);
  CHECK((get<0, 0>(det_inv.second)) == 0.5);
}

template <typename TensorType>
void verify_det_and_inv_generic_2d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<1, 0>(t) = 4.0;
  get<1, 1>(t) = 5.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -2.0);
  CHECK((get<0, 0>(det_inv.second)) == -2.5);
  CHECK((get<0, 1>(det_inv.second)) == 1.5);
  CHECK((get<1, 0>(det_inv.second)) == 2.0);
  CHECK((get<1, 1>(det_inv.second)) == -1.0);
}

template <typename TensorType>
void verify_det_and_inv_generic_3d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<0, 2>(t) = 6.0;
  get<1, 0>(t) = 4.0;
  get<1, 1>(t) = 5.0;
  get<1, 2>(t) = 7.0;
  get<2, 0>(t) = 8.0;
  get<2, 1>(t) = 9.0;
  get<2, 2>(t) = 10.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -2.0);
  CHECK((get<0, 0>(det_inv.second)) == 6.5);
  CHECK((get<0, 1>(det_inv.second)) == -12.0);
  CHECK((get<0, 2>(det_inv.second)) == 4.5);
  CHECK((get<1, 0>(det_inv.second)) == -8.0);
  CHECK((get<1, 1>(det_inv.second)) == 14.0);
  CHECK((get<1, 2>(det_inv.second)) == -5.0);
  CHECK((get<2, 0>(det_inv.second)) == 2.0);
  CHECK((get<2, 1>(det_inv.second)) == -3.0);
  CHECK((get<2, 2>(det_inv.second)) == 1.0);
}

template <typename TensorType>
void verify_det_and_inv_generic_4d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<0, 2>(t) = 6.0;
  get<0, 3>(t) = 11.0;
  get<1, 0>(t) = 4.0;
  get<1, 1>(t) = 5.0;
  get<1, 2>(t) = 7.0;
  get<1, 3>(t) = 12.0;
  get<2, 0>(t) = 8.0;
  get<2, 1>(t) = 9.0;
  get<2, 2>(t) = 10.0;
  get<2, 3>(t) = 13.0;
  get<3, 0>(t) = 14.0;
  get<3, 1>(t) = 15.0;
  get<3, 2>(t) = 16.0;
  get<3, 3>(t) = 17.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -8.0);
  CHECK((get<0, 0>(det_inv.second)) == -4.0);
  CHECK((get<0, 1>(det_inv.second)) == 9.0);
  CHECK((get<0, 2>(det_inv.second)) == -9.5);
  CHECK((get<0, 3>(det_inv.second)) == 3.5);
  CHECK((get<1, 0>(det_inv.second)) == 3.25);
  CHECK((get<1, 1>(det_inv.second)) == -8.5);
  CHECK((get<1, 2>(det_inv.second)) == 10.0);
  CHECK((get<1, 3>(det_inv.second)) == -3.75);
  CHECK((get<2, 0>(det_inv.second)) == 1.25);
  CHECK((get<2, 1>(det_inv.second)) == -1.5);
  CHECK((get<2, 2>(det_inv.second)) == 0.0);
  CHECK((get<2, 3>(det_inv.second)) == 0.25);
  CHECK((get<3, 0>(det_inv.second)) == -0.75);
  CHECK((get<3, 1>(det_inv.second)) == 1.5);
  CHECK((get<3, 2>(det_inv.second)) == -1.0);
  CHECK((get<3, 3>(det_inv.second)) == 0.25);
}

template <typename TensorType>
void verify_det_and_inv_symmetric_2d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<1, 1>(t) = 5.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == 1.0);
  CHECK((get<0, 0>(det_inv.second)) == 5.0);
  CHECK((get<0, 1>(det_inv.second)) == -3.0);
  CHECK((get<1, 0>(det_inv.second)) == -3.0);
  CHECK((get<1, 1>(det_inv.second)) == 2.0);
}

template <typename TensorType>
void verify_det_and_inv_symmetric_3d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<0, 2>(t) = 6.0;
  get<1, 1>(t) = 5.0;
  get<1, 2>(t) = 7.0;
  get<2, 2>(t) = 10.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -16.0);
  CHECK((get<0, 0>(det_inv.second)) == -0.0625);
  CHECK((get<0, 1>(det_inv.second)) == -0.75);
  CHECK((get<0, 2>(det_inv.second)) == 0.5625);
  CHECK((get<1, 0>(det_inv.second)) == -0.75);
  CHECK((get<1, 1>(det_inv.second)) == 1.0);
  CHECK((get<1, 2>(det_inv.second)) == -0.25);
  CHECK((get<2, 0>(det_inv.second)) == 0.5625);
  CHECK((get<2, 1>(det_inv.second)) == -0.25);
  CHECK((get<2, 2>(det_inv.second)) == -0.0625);
}

template <typename TensorType>
void verify_det_and_inv_symmetric_4d() {
  TensorType t{};
  get<0, 0>(t) = 2.0;
  get<0, 1>(t) = 3.0;
  get<0, 2>(t) = 6.0;
  get<0, 3>(t) = 11.0;
  get<1, 1>(t) = 5.0;
  get<1, 2>(t) = 7.0;
  get<1, 3>(t) = 12.0;
  get<2, 2>(t) = 10.0;
  get<2, 3>(t) = 13.0;
  get<3, 3>(t) = 17.0;
  const auto det_inv = determinant_and_inverse(t);
  CHECK(det_inv.first.get() == -100.0);
  CHECK((get<0, 0>(det_inv.second)) == approx(0.84));
  CHECK((get<0, 1>(det_inv.second)) == approx(-0.94));
  CHECK((get<0, 2>(det_inv.second)) == approx(-0.34));
  CHECK((get<0, 3>(det_inv.second)) == approx(0.38));
  CHECK((get<1, 0>(det_inv.second)) == approx(-0.94));
  CHECK((get<1, 1>(det_inv.second)) == approx(1.04));
  CHECK((get<1, 2>(det_inv.second)) == approx(-0.06));
  CHECK((get<1, 3>(det_inv.second)) == approx(-0.08));
  CHECK((get<2, 0>(det_inv.second)) == approx(-0.34));
  CHECK((get<2, 1>(det_inv.second)) == approx(-0.06));
  CHECK((get<2, 2>(det_inv.second)) == approx(0.84));
  CHECK((get<2, 3>(det_inv.second)) == approx(-0.38));
  CHECK((get<3, 0>(det_inv.second)) == approx(0.38));
  CHECK((get<3, 1>(det_inv.second)) == approx(-0.08));
  CHECK((get<3, 2>(det_inv.second)) == approx(-0.38));
  CHECK((get<3, 3>(det_inv.second)) == approx(0.16));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.DeterminantAndInverse",
                  "[DataStructures][Unit]") {
  // Check that the inverse tensor has the expected index structure -- for an
  // input Tensor of type T^a_b, the inverse should have type T^b_a.
  {
    static_assert(
        cpp17::is_same_v<
            Tensor<double, tmpl::integral_list<int32_t, 2, 1>,
                   index_list<SpatialIndex<2, UpLo::Up, Frame::Inertial>,
                              SpatialIndex<2, UpLo::Lo, Frame::Grid>>>,
            decltype(determinant_and_inverse(
                std::declval<
                    Tensor<double, tmpl::integral_list<int32_t, 2, 1>,
                           index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                                      SpatialIndex<2, UpLo::Lo,
                                                   Frame::Inertial>>>>()))::
                second_type>,
        "Inverse tensor has incorrect index structure.");
  }

  // Check paired determinant and inverse for 1x1 through 4x4 tensors, both
  // generic and symmetric, with both spatial and spacetime indices.
  {
    verify_det_and_inv_1d<tnsr::ii<double, 1, Frame::Grid>>();
    verify_det_and_inv_1d<tnsr::ij<double, 1, Frame::Grid>>();
    verify_det_and_inv_1d<tnsr_iJ<double, 1, Frame::Grid, Frame::Inertial>>();

    verify_det_and_inv_symmetric_2d<tnsr::ii<double, 2, Frame::Grid>>();
    verify_det_and_inv_generic_2d<tnsr::ij<double, 2, Frame::Grid>>();
    verify_det_and_inv_generic_2d<
        tnsr_iJ<double, 2, Frame::Grid, Frame::Inertial>>();

    verify_det_and_inv_symmetric_3d<tnsr::ii<double, 3, Frame::Grid>>();
    verify_det_and_inv_generic_3d<tnsr::ij<double, 3, Frame::Grid>>();
    verify_det_and_inv_generic_3d<
        tnsr_iJ<double, 3, Frame::Grid, Frame::Inertial>>();

    verify_det_and_inv_symmetric_4d<tnsr::ii<double, 4, Frame::Grid>>();
    verify_det_and_inv_generic_4d<tnsr::ij<double, 4, Frame::Grid>>();
    verify_det_and_inv_generic_4d<
        tnsr_iJ<double, 4, Frame::Grid, Frame::Inertial>>();

    verify_det_and_inv_symmetric_2d<tnsr::aa<double, 1, Frame::Grid>>();
    verify_det_and_inv_generic_2d<tnsr::ab<double, 1, Frame::Grid>>();

    verify_det_and_inv_symmetric_3d<tnsr::aa<double, 2, Frame::Grid>>();
    verify_det_and_inv_generic_3d<tnsr::ab<double, 2, Frame::Grid>>();

    verify_det_and_inv_symmetric_4d<tnsr::aa<double, 3, Frame::Grid>>();
    verify_det_and_inv_generic_4d<tnsr::ab<double, 3, Frame::Grid>>();
  }

  // Check paired determinant and inverse for a Tensor<DataVector>.
  {
    tnsr::ij<DataVector, 2, Frame::Grid> t{};
    get<0, 0>(t) = DataVector({2.0, 3.0, 5.0, 1.0});
    get<0, 1>(t) = DataVector({3.0, 5.0, 4.0, -1.0});
    get<1, 0>(t) = DataVector({4.0, 2.0, 2.0, 1.0});
    get<1, 1>(t) = DataVector({5.0, 3.0, 2.0, 1.0});
    const auto det_inv = determinant_and_inverse(t);
    CHECK(det_inv.first.get() == DataVector({-2.0, -1.0, 2.0, 2.0}));
    CHECK((get<0, 0>(det_inv.second)) == DataVector({-2.5, -3.0, 1.0, 0.5}));
    CHECK((get<0, 1>(det_inv.second)) == DataVector({1.5, 5.0, -2.0, 0.5}));
    CHECK((get<1, 0>(det_inv.second)) == DataVector({2.0, 2.0, -1.0, -0.5}));
    CHECK((get<1, 1>(det_inv.second)) == DataVector({-1.0, -3.0, 2.5, 0.5}));
  }
}
