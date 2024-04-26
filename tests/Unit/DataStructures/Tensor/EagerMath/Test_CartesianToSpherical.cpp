// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/CartesianToSpherical.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.CartesianToSpherical",
                  "[DataStructures][Unit]") {
  {
    tnsr::I<double, 2, Frame::Grid> x{{{1., 1.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(sqrt(2.0)));
    CHECK(get<1>(x_spherical) == approx(M_PI_4));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(1.));
    CHECK(get<1>(x_back) == approx(1.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, 1.}, DataVector{5, 0.}, DataVector{5, 0.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(1.));
    CHECK(get<1>(x_spherical) == approx(M_PI_2));
    CHECK(get<2>(x_spherical) == approx(0.));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(1.));
    CHECK(get<1>(x_back) == approx(0.));
    CHECK(get<2>(x_back) == approx(0.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, 0.}, DataVector{5, 1.}, DataVector{5, 0.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(1.));
    CHECK(get<1>(x_spherical) == approx(M_PI_2));
    CHECK(get<2>(x_spherical) == approx(M_PI_2));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(0.));
    CHECK(get<1>(x_back) == approx(1.));
    CHECK(get<2>(x_back) == approx(0.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, -1.}, DataVector{5, 0.}, DataVector{5, 0.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(1.));
    CHECK(get<1>(x_spherical) == approx(M_PI_2));
    CHECK(get<2>(x_spherical) == approx(M_PI));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(-1.));
    CHECK(get<1>(x_back) == approx(0.));
    CHECK(get<2>(x_back) == approx(0.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, 1.}, DataVector{5, 0.}, DataVector{5, 1.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(sqrt(2.)));
    CHECK(get<1>(x_spherical) == approx(M_PI_4));
    CHECK(get<2>(x_spherical) == approx(0.));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(1.));
    CHECK(get<1>(x_back) == approx(0.));
    CHECK(get<2>(x_back) == approx(1.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, 1.}, DataVector{5, 0.}, DataVector{5, -1.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(sqrt(2.)));
    CHECK(get<1>(x_spherical) == approx(3. * M_PI_4));
    CHECK(get<2>(x_spherical) == approx(0.));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(1.));
    CHECK(get<1>(x_back) == approx(0.));
    CHECK(get<2>(x_back) == approx(-1.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, 0.}, DataVector{5, 0.}, DataVector{5, 1.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(1.));
    CHECK(get<1>(x_spherical) == approx(0.));
    CHECK(get<2>(x_spherical) == approx(0.));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(0.));
    CHECK(get<1>(x_back) == approx(0.));
    CHECK(get<2>(x_back) == approx(1.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, 0.}, DataVector{5, 0.}, DataVector{5, -1.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(1.));
    CHECK(get<1>(x_spherical) == approx(M_PI));
    CHECK(get<2>(x_spherical) == approx(0.));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(0.));
    CHECK(get<1>(x_back) == approx(0.));
    CHECK(get<2>(x_back) == approx(-1.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, 1.}, DataVector{5, 1.}, DataVector{5, 1.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(sqrt(3.)));
    CHECK(get<1>(x_spherical) == approx(acos(1. / sqrt(3.))));
    CHECK(get<2>(x_spherical) == approx(M_PI_4));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(1.));
    CHECK(get<1>(x_back) == approx(1.));
    CHECK(get<2>(x_back) == approx(1.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, -1.}, DataVector{5, 1.}, DataVector{5, 1.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(sqrt(3.)));
    CHECK(get<1>(x_spherical) == approx(acos(1. / sqrt(3.))));
    CHECK(get<2>(x_spherical) == approx(3. * M_PI_4));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(-1.));
    CHECK(get<1>(x_back) == approx(1.));
    CHECK(get<2>(x_back) == approx(1.));
  }
  {
    tnsr::I<DataVector, 3, Frame::Inertial> x{
        {DataVector{5, -1.}, DataVector{5, -1.}, DataVector{5, 1.}}};
    const auto x_spherical = cartesian_to_spherical(x);
    CHECK(get<0>(x_spherical) == approx(sqrt(3.)));
    CHECK(get<1>(x_spherical) == approx(acos(1. / sqrt(3.))));
    CHECK(get<2>(x_spherical) == approx(-3. * M_PI_4));
    const auto x_back = spherical_to_cartesian(x_spherical);
    CHECK(get<0>(x_back) == approx(-1.));
    CHECK(get<1>(x_back) == approx(-1.));
    CHECK(get<2>(x_back) == approx(1.));
  }
}
