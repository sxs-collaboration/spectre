// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace TestHelpers::Xcts::Solutions {

namespace detail {
template <::Xcts::Equations EnabledEquations,
          ::Xcts::Geometry ConformalGeometry, int ConformalMatterScale,
          typename Solution>
void verify_solution_impl(const Solution& solution,
                          const std::array<double, 3>& center,
                          const double inner_radius, const double outer_radius,
                          const double tolerance) {
  CAPTURE(EnabledEquations);
  CAPTURE(ConformalGeometry);
  using system = ::Xcts::FirstOrderSystem<EnabledEquations, ConformalGeometry,
                                          ConformalMatterScale>;
  const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  using AffineMap = domain::CoordinateMaps::Affine;
  using AffineMap3D =
      domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
  const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                              AffineMap3D>
      coord_map{
          {{-1., 1., center[0] + inner_radius, center[0] + outer_radius},
           {-1., 1., center[1] + inner_radius, center[1] + outer_radius},
           {-1., 1., center[2] + inner_radius, center[2] + outer_radius}}};
  FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
      solution, mesh, coord_map, tolerance);
}
}  // namespace detail

/*!
 * \brief Verify the `solution` solves the XCTS equations numerically.
 *
 * \tparam ConformalGeometry Specify `Xcts::Geometry::FlatCartesian` to test
 * both the flat _and_ the curved system, or `Xcts::Geometry::Curved` to only
 * test the curved system (for solutions on a curved conformal background).
 * \tparam Solution The analytic solution to test (inferred)
 * \param solution The analytic solution to test
 * \param center offset for the \p inner_radius and the \p outer_radius
 * \param inner_radius Lower-left corner of a cube on which to test
 * \param outer_radius Upper-right corner of a cube on which to test
 * \param tolerance Requested tolerance
 */
template <::Xcts::Geometry ConformalGeometry, int ConformalMatterScale,
          typename Solution>
void verify_solution(const Solution& solution,
                     const std::array<double, 3>& center,
                     const double inner_radius, const double outer_radius,
                     const double tolerance) {
  if constexpr (ConformalGeometry == ::Xcts::Geometry::FlatCartesian) {
    INVOKE_TEST_FUNCTION(
        detail::verify_solution_impl,
        (solution, center, inner_radius, outer_radius, tolerance),
        (::Xcts::Equations::Hamiltonian, ::Xcts::Equations::HamiltonianAndLapse,
         ::Xcts::Equations::HamiltonianLapseAndShift),
        (::Xcts::Geometry::FlatCartesian, ::Xcts::Geometry::Curved),
        (ConformalMatterScale));
  } else {
    INVOKE_TEST_FUNCTION(
        detail::verify_solution_impl,
        (solution, center, inner_radius, outer_radius, tolerance),
        (::Xcts::Equations::Hamiltonian, ::Xcts::Equations::HamiltonianAndLapse,
         ::Xcts::Equations::HamiltonianLapseAndShift),
        (::Xcts::Geometry::Curved), (ConformalMatterScale));
  }
}

}  // namespace TestHelpers::Xcts::Solutions
