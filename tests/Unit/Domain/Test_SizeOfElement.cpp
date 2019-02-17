// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/Direction.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
tnsr::I<DataVector, 2> make_inertial_coords_2d(const Mesh<2>& mesh) noexcept {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  const auto map = domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine2D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2)},
      domain::CoordinateMaps::DiscreteRotation<2>(
          OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}));
  return map(logical_coordinates(mesh));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.SizeOfElement", "[Domain][Unit]") {
  SECTION("1D") {
    const auto mesh = Mesh<1>(4, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);
    const auto coords = [&mesh]() {
      const auto map =
          domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
              domain::CoordinateMaps::Affine(-1.0, 1.0, 0.3, 1.2));
      return map(logical_coordinates(mesh));
    }();
    const auto size = size_of_element(mesh, coords);
    const auto size_expected = make_array<1>(0.9);
    CHECK_ITERABLE_APPROX(size, size_expected);
  }

  SECTION("2D") {
    const auto mesh = Mesh<2>({{4, 5}}, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);
    const auto coords = make_inertial_coords_2d(mesh);
    const auto size = size_of_element(mesh, coords);
    const auto size_expected = make_array(0.1, 1.7);
    CHECK_ITERABLE_APPROX(size, size_expected);
  }

  SECTION("3D") {
    const auto mesh = Mesh<3>({{4, 5, 6}}, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);
    const auto coords = [&mesh]() {
      using Affine = domain::CoordinateMaps::Affine;
      using Affine3D =
          domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
      const auto map =
          domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
              Affine3D{Affine(-1.0, 1.0, 0.3, 0.4),
                       Affine(-1.0, 1.0, -0.5, 1.2),
                       Affine(-1.0, 1.0, 12.0, 12.5)},
              domain::CoordinateMaps::Rotation<3>(0.7, 2.3, -0.4));
      return map(logical_coordinates(mesh));
    }();
    const auto size = size_of_element(mesh, coords);
    const auto size_expected = make_array(0.1, 1.7, 0.5);
    CHECK_ITERABLE_APPROX(size, size_expected);
  }

  SECTION("ComputeTag") {
    auto mesh = Mesh<2>({{4, 5}}, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto);
    auto coords = make_inertial_coords_2d(mesh);
    const auto box = db::create<
        db::AddSimpleTags<Tags::Mesh<2>, Tags::Coordinates<2, Frame::Inertial>>,
        db::AddComputeTags<Tags::SizeOfElement<2>>>(mesh, std::move(coords));

    const auto size_compute_item = db::get<Tags::SizeOfElement<2>>(box);
    const auto size_expected = make_array(0.1, 1.7);
    CHECK_ITERABLE_APPROX(size_compute_item, size_expected);
  }
}
