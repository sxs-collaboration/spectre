// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Side.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/Domain/DomainTestHelpers.hpp"

namespace {
void test_vci_1d() {
  VolumeCornerIterator<1> vci{};
  CHECK(vci);
  CHECK(vci() == std::array<Side, 1>{{Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 1>{{-1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 1>{{Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 1>{{1.0}});
  ++vci;
  CHECK(not vci);
}

void test_vci_2d() {
  VolumeCornerIterator<2> vci{};
  CHECK(vci);
  CHECK(vci() == std::array<Side, 2>{{Side::Lower, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{-1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 2>{{Side::Upper, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 2>{{Side::Lower, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{-1.0, 1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 2>{{Side::Upper, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 2>{{1.0, 1.0}});
  ++vci;
  CHECK(not vci);
}

void test_vci_3d() {
  VolumeCornerIterator<3> vci{};
  CHECK(vci);
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Lower, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, -1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Lower, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, -1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Upper, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, 1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Upper, Side::Lower}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, 1.0, -1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Lower, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, -1.0, 1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Lower, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, -1.0, 1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Lower, Side::Upper, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{-1.0, 1.0, 1.0}});
  ++vci;
  CHECK(vci() == std::array<Side, 3>{{Side::Upper, Side::Upper, Side::Upper}});
  CHECK(vci.coords_of_corner() == std::array<double, 3>{{1.0, 1.0, 1.0}});
  ++vci;
  CHECK(not vci);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.TestHelpers.VolumeCornerIterator",
                  "[Domain][Unit]") {
  test_vci_1d();
  test_vci_2d();
  test_vci_3d();
}

namespace {

template <size_t SpatialDim, typename SpatialFrame>
void test_euclidean_basis_vectors(const DataVector& used_for_size) noexcept {
  for (const auto& direction : Direction<SpatialDim>::all_directions()) {
    auto expected =
        make_with_value<tnsr::I<DataVector, SpatialDim, SpatialFrame>>(
            used_for_size, 0.0);
    expected.get(direction.axis()) =
        make_with_value<DataVector>(used_for_size, direction.sign());

    CHECK_ITERABLE_APPROX((euclidean_basis_vector<SpatialDim, SpatialFrame>(
                              direction, used_for_size)),
                          std::move(expected));
  }
}

}  //  namespace

SPECTRE_TEST_CASE("Unit.Domain.TestHelpers.BasisVector", "[Unit][Domain]") {
  const DataVector dv(5);

  test_euclidean_basis_vectors<1, Frame::Inertial>(dv);
  test_euclidean_basis_vectors<2, Frame::Inertial>(dv);
  test_euclidean_basis_vectors<3, Frame::Inertial>(dv);

  test_euclidean_basis_vectors<1, Frame::Grid>(dv);
  test_euclidean_basis_vectors<2, Frame::Grid>(dv);
  test_euclidean_basis_vectors<3, Frame::Grid>(dv);
}
