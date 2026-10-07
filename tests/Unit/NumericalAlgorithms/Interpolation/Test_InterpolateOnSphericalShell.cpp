// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolateOnSphericalShell.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarA : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct ScalarB : db::SimpleTag {
  using type = Scalar<DataVector>;
};

void test_interpolate_on_spherical_shell() {
  const Mesh<3> mesh{
      {{6, 8, 15}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};
  const auto logical = logical_coordinates(mesh);
  Variables<tmpl::list<ScalarA, ScalarB>> vars{mesh.number_of_grid_points()};
  get(get<ScalarA>(vars)) = (1.0 + get<0>(logical) + square(get<0>(logical))) *
                            (sin(get<1>(logical)) * cos(get<2>(logical)) +
                             square(cos(get<1>(logical))));
  get(get<ScalarB>(vars)) =
      exp(get<0>(logical)) * sin(get<1>(logical)) * sin(2.0 * get<2>(logical));

  // - Two groups whose radial coordinates differ by roundoff.
  // - A point just beyond the grouping tolerance, which would be off by about
  //   1e-9 if it were interpolated with the preceding group.
  // - Points alternating between two radial coordinates, which aren't grouped.
  // - A single point on the outer face.
  const std::vector<double> xi{
      0.3,  0.3 + 1.0e-15,  0.3 - 2.0e-15, 0.3 + 1.0e-9, -0.7, -0.7 + 1.0e-15,
      -0.7, -0.7 - 1.0e-15, 0.5,           -0.5,         0.5,  1.0};
  const std::vector<double> theta{0.1, 1.2, 3.0, 0.5, 0.0, 0.7,
                                  2.2, 1.6, 0.9, 2.5, 1.1, 0.4};
  const std::vector<double> phi{0.0, 2.0, 6.1, 1.5, 1.0, 4.4,
                                3.3, 0.2, 2.7, 3.9, 5.5, 5.0};
  tnsr::I<DataVector, 3, Frame::ElementLogical> target_points{xi.size()};
  for (size_t s = 0; s < xi.size(); ++s) {
    get<0>(target_points)[s] = xi[s];
    get<1>(target_points)[s] = theta[s];
    get<2>(target_points)[s] = phi[s];
  }
  const auto result =
      intrp::interpolate_on_spherical_shell(vars, mesh, target_points);
  const auto expected =
      intrp::Irregular<3>(mesh, target_points).interpolate(vars);
  CHECK_VARIABLES_APPROX(result, expected);
}

#ifdef SPECTRE_DEBUG
void test_errors() {
  const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Variables<tmpl::list<ScalarA>> vars{mesh.number_of_grid_points(), 0.0};
  const tnsr::I<DataVector, 3, Frame::ElementLogical> target_points{1_st, 0.0};
  CHECK_THROWS_WITH(
      intrp::interpolate_on_spherical_shell(vars, mesh, target_points),
      Catch::Matchers::ContainsSubstring(
          "Expected a spherical-harmonic shell, but the mesh is"));
}
#endif

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.InterpolateOnSphericalShell",
                  "[Unit][NumericalAlgorithms]") {
  test_interpolate_on_spherical_shell();
#ifdef SPECTRE_DEBUG
  test_errors();
#endif
}
}  // namespace
