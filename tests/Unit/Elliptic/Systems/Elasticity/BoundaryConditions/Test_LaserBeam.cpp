// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Elasticity/BoundaryConditions/LaserBeam.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity::BoundaryConditions {

namespace {
std::vector<LaserBeam::ModeInfo> tem00_modes() { return {{0, 0, 1.}}; }

// Must match `HG_MODES` in LaserBeam.py
std::vector<LaserBeam::ModeInfo> hg_modes() {
  return {{0, 2, 1.}, {2, 0, -1.}, {1, 3, 0.5}};
}

// The boundary condition assumes a flat surface, so the (random) unit normal
// is the same at all points. It is parametrized by the angles theta and phi.
template <bool Linearized, bool Tem00>
void apply_boundary_condition(
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
    const tnsr::I<DataVector, 3>& x, const double theta, const double phi,
    const double beam_width) {
  const LaserBeam laser_beam{beam_width, Tem00 ? tem00_modes() : hg_modes()};
  const auto direction = Direction<3>::lower_xi();
  auto face_normal = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
  get<0>(face_normal) = sin(theta) * cos(phi);
  get<1>(face_normal) = sin(theta) * sin(phi);
  get<2>(face_normal) = cos(theta);
  const auto box = db::create<db::AddSimpleTags<
      domain::Tags::Faces<3, domain::Tags::Coordinates<3, Frame::Inertial>>,
      domain::Tags::Faces<3, domain::Tags::FaceNormal<3>>>>(
      DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, x}},
      DirectionMap<3, tnsr::i<DataVector, 3>>{{direction, face_normal}});
  tnsr::I<DataVector, 3> displacement{
      x.begin()->size(), std::numeric_limits<double>::signaling_NaN()};
  tnsr::iJ<DataVector, 3> deriv_displacement{
      x.begin()->size(), std::numeric_limits<double>::signaling_NaN()};
  elliptic::apply_boundary_condition<Linearized, void, tmpl::list<LaserBeam>>(
      laser_beam, box, direction, make_not_null(&displacement),
      n_dot_minus_stress, deriv_displacement);
}

// Pressure p on a flat surface with the constant outward unit normal n,
// extracted from n_i (-T^{ij}) = -p n^j. Also checks that the traction has no
// tangential component. The check is a single whole-vector comparison per
// component to keep the number of assertions small on large grids.
DataVector pressure_on_surface(const LaserBeam& laser_beam,
                               const tnsr::I<DataVector, 3>& x,
                               const std::array<double, 3>& normal) {
  const size_t num_points = get<0>(x).size();
  auto face_normal = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
  for (size_t d = 0; d < 3; ++d) {
    face_normal.get(d) = gsl::at(normal, d);
  }
  tnsr::I<DataVector, 3> displacement{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  tnsr::I<DataVector, 3> n_dot_minus_stress{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  const tnsr::iJ<DataVector, 3> deriv_displacement{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  laser_beam.apply(make_not_null(&displacement),
                   make_not_null(&n_dot_minus_stress), deriv_displacement, x,
                   face_normal);
  DataVector pressure{num_points, 0.};
  for (size_t d = 0; d < 3; ++d) {
    pressure -= gsl::at(normal, d) * n_dot_minus_stress.get(d);
  }
  const double scale = max(abs(pressure));
  for (size_t d = 0; d < 3; ++d) {
    const DataVector residual =
        n_dot_minus_stress.get(d) + gsl::at(normal, d) * pressure;
    CHECK(max(abs(residual)) <= 1.e-14 * scale);
  }
  return pressure;
}

// Pressure on the z = const surface with outward normal (0, 0, -1)
DataVector pressure_on_z_surface(const LaserBeam& laser_beam,
                                 const tnsr::I<DataVector, 3>& x) {
  return pressure_on_surface(laser_beam, x, {{0., 0., -1.}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.BoundaryConditions.LaserBeam",
                  "[Unit][Elliptic]") {
  const double beam_width = 2.;
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>, LaserBeam>(
      "LaserBeam:\n"
      "  BeamWidth: 2.\n"
      "  Modes:\n"
      "    - N: 0\n"
      "      M: 0\n"
      "      Weight: 1.");
  REQUIRE(dynamic_cast<const LaserBeam*>(created.get()) != nullptr);
  const auto& laser_beam = dynamic_cast<const LaserBeam&>(*created);
  const auto created_hg = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>, LaserBeam>(
      "LaserBeam:\n"
      "  BeamWidth: 2.\n"
      "  Modes:\n"
      "    - N: 0\n"
      "      M: 2\n"
      "      Weight: 1.\n"
      "    - N: 2\n"
      "      M: 0\n"
      "      Weight: -1.\n"
      "    - N: 1\n"
      "      M: 3\n"
      "      Weight: 0.5");
  REQUIRE(dynamic_cast<const LaserBeam*>(created_hg.get()) != nullptr);
  const auto& laser_beam_hg = dynamic_cast<const LaserBeam&>(*created_hg);

  {
    INFO("Semantics");
    test_serialization(laser_beam);
    test_copy_semantics(laser_beam);
    auto move_laser_beam = laser_beam;
    test_move_semantics(std::move(move_laser_beam), laser_beam);
    test_serialization(laser_beam_hg);
    CHECK(laser_beam != laser_beam_hg);
    CHECK(laser_beam == LaserBeam{beam_width, tem00_modes()});
    CHECK(laser_beam_hg == LaserBeam{beam_width, hg_modes()});
    CHECK(laser_beam != LaserBeam{3., tem00_modes()});
  }
  {
    INFO("Properties");
    CHECK(laser_beam.beam_width() == beam_width);
    CHECK(laser_beam.modes() == tem00_modes());
    CHECK(laser_beam_hg.modes() == hg_modes());
    CHECK(laser_beam.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              3, elliptic::BoundaryConditionType::Neumann});
  }
  {
    INFO("ModeInfo");
    const auto mode = TestHelpers::test_creation<LaserBeam::ModeInfo>(
        "N: 1\n"
        "M: 3\n"
        "Weight: -0.5");
    CHECK(mode == LaserBeam::ModeInfo{1, 3, -0.5});
    CHECK(mode != LaserBeam::ModeInfo{3, 1, -0.5});
    test_serialization(mode);
  }
  {
    INFO("Mode list order and repetition are preserved");
    // Flow-style entries, as suggested in the help text
    const std::vector<LaserBeam::ModeInfo> repeated{
        {2, 0, -1.}, {0, 2, 1.}, {0, 2, 1.}};
    const auto created_repeated = TestHelpers::test_factory_creation<
        elliptic::BoundaryConditions::BoundaryCondition<3>, LaserBeam>(
        "LaserBeam:\n"
        "  BeamWidth: 2.\n"
        "  Modes:\n"
        "    - {N: 2, M: 0, Weight: -1.}\n"
        "    - {N: 0, M: 2, Weight: 1.}\n"
        "    - {N: 0, M: 2, Weight: 1.}");
    const auto& laser_beam_repeated =
        dynamic_cast<const LaserBeam&>(*created_repeated);
    CHECK(laser_beam_repeated.modes() == repeated);
    // Parsed fields of the first entry, independent of operator==
    REQUIRE(laser_beam_repeated.modes().size() == 3);
    CHECK(laser_beam_repeated.modes()[0].n == 2);
    CHECK(laser_beam_repeated.modes()[0].m == 0);
    CHECK(laser_beam_repeated.modes()[0].weight == -1.);
    CHECK(laser_beam_repeated == LaserBeam{beam_width, repeated});
    // operator== compares the mode lists element-wise, so a reordered list
    // is a different object ...
    const LaserBeam laser_beam_reordered{beam_width,
                                         {{0, 2, 1.}, {0, 2, 1.}, {2, 0, -1.}}};
    CHECK(laser_beam_repeated != laser_beam_reordered);
    // ... but the pressure is invariant under reordering
    tnsr::I<DataVector, 3> x_reorder{size_t{3}, 0.};
    get<0>(x_reorder) = DataVector{0.3, -1.2, 2.1};
    get<1>(x_reorder) = DataVector{-0.7, 0.4, 1.5};
    const auto pressure_repeated =
        pressure_on_z_surface(laser_beam_repeated, x_reorder);
    const auto pressure_reordered =
        pressure_on_z_surface(laser_beam_reordered, x_reorder);
    CHECK_ITERABLE_APPROX(pressure_repeated, pressure_reordered);
    // Analytic reference at (x, y) = (1, 0.5) on the z = const surface,
    // where u = x and v = -y (H_m^2 is even): with r0 = 2 the arguments
    // are u / r0 = 0.5 and v / r0 = 0.25, H_2(0.5) = -1, H_2(0.25) = -1.75,
    // and each TEM20/TEM02 mode is normalized by pi r0^2 2^2 2! = 8 pi r0^2.
    // The bracket is -1 + 2 * 3.0625 = 5.125; an evaluator that discarded
    // the duplicate would give 2.0625 instead.
    tnsr::I<DataVector, 3> x_point{size_t{1}, 0.};
    get<0>(x_point) = DataVector{1.};
    get<1>(x_point) = DataVector{0.5};
    const double pressure_point_expected =
        std::exp(-1.25 / 4.) / (4. * M_PI) * 5.125 / 8.;
    CHECK(pressure_on_z_surface(laser_beam_repeated, x_point)[0] ==
          approx(pressure_point_expected));
    test_serialization(laser_beam_repeated);
  }
  {
    INFO("Invalid options");
    using Catch::Matchers::ContainsSubstring;
    const auto create = [](const std::string& options) {
      return TestHelpers::test_factory_creation<
          elliptic::BoundaryConditions::BoundaryCondition<3>, LaserBeam>(
          options);
    };
    // Modes is required (the option parser has no defaults)
    CHECK_THROWS_WITH(create("LaserBeam:\n"
                             "  BeamWidth: 2."),
                      ContainsSubstring("did not specify the option") and
                          ContainsSubstring("Modes"));
    // The mode indices are unsigned
    CHECK_THROWS_WITH(create("LaserBeam:\n"
                             "  BeamWidth: 2.\n"
                             "  Modes:\n"
                             "    - {N: -1, M: 0, Weight: 1.}"),
                      ContainsSubstring("Failed to convert value"));
    CHECK_THROWS_WITH(create("LaserBeam:\n"
                             "  BeamWidth: 2.\n"
                             "  Modes:\n"
                             "    - {N: 0, M: -1, Weight: 1.}"),
                      ContainsSubstring("Failed to convert value"));
    // Misspelled key
    CHECK_THROWS_WITH(
        create("LaserBeam:\n"
               "  BeamWidth: 2.\n"
               "  Modes:\n"
               "    - {N: 0, M: 0, Weigth: 1.}"),
        ContainsSubstring("Option 'Weigth' is not a valid option"));
    // Zero width passes the (inclusive) lower bound of the option but would
    // divide by zero, so the constructor rejects it
    CHECK_THROWS_WITH(create("LaserBeam:\n"
                             "  BeamWidth: 0.\n"
                             "  Modes:\n"
                             "    - {N: 0, M: 0, Weight: 1.}"),
                      ContainsSubstring("BeamWidth must be positive"));
    CHECK_THROWS_WITH(create("LaserBeam:\n"
                             "  BeamWidth: -1.\n"
                             "  Modes:\n"
                             "    - {N: 0, M: 0, Weight: 1.}"),
                      ContainsSubstring("is below the lower bound"));
    // An empty mode list would give zero traction
    CHECK_THROWS_WITH(create("LaserBeam:\n"
                             "  BeamWidth: 2.\n"
                             "  Modes: []"),
                      ContainsSubstring("at least one mode"));
    // The same checks apply to direct construction
    CHECK_THROWS_WITH(LaserBeam(0., tem00_modes()),
                      ContainsSubstring("BeamWidth must be positive"));
    CHECK_THROWS_WITH(LaserBeam(-1., tem00_modes()),
                      ContainsSubstring("BeamWidth must be positive"));
    // Non-finite widths are rejected by the constructor's std::isfinite
    // check, which does not raise FE_INVALID. A NaN in an input file does
    // not reach the constructor: the option parser's lower-bound comparison
    // with NaN raises FE_INVALID first, which is trapped, so there is no YAML
    // NaN case. Infinity passes that comparison and reaches the constructor.
    CHECK_THROWS_WITH(
        LaserBeam(std::numeric_limits<double>::quiet_NaN(), tem00_modes()),
        ContainsSubstring("BeamWidth must be positive"));
    CHECK_THROWS_WITH(
        LaserBeam(std::numeric_limits<double>::infinity(), tem00_modes()),
        ContainsSubstring("BeamWidth must be positive"));
    CHECK_THROWS_WITH(create("LaserBeam:\n"
                             "  BeamWidth: .inf\n"
                             "  Modes:\n"
                             "    - {N: 0, M: 0, Weight: 1.}"),
                      ContainsSubstring("BeamWidth must be positive"));
    CHECK_THROWS_WITH(LaserBeam(2., {}),
                      ContainsSubstring("at least one mode"));
    // Non-empty lists with zero or cancelling weights remain valid
    const auto created_zero_weight = create(
        "LaserBeam:\n"
        "  BeamWidth: 2.\n"
        "  Modes:\n"
        "    - {N: 0, M: 0, Weight: 0.}");
    CHECK(dynamic_cast<const LaserBeam&>(*created_zero_weight).modes() ==
          std::vector<LaserBeam::ModeInfo>{{0, 0, 0.}});
    CHECK(LaserBeam(2., {{0, 2, 1.}, {2, 0, -1.}}).modes().size() == 2);
    // The same mode twice with opposite weights is also valid and cancels
    // pointwise, so the pressure is zero (the TEM02 - TEM20 pair above only
    // has zero net force)
    const LaserBeam laser_beam_cancelling{2., {{0, 2, 1.}, {0, 2, -1.}}};
    CHECK(laser_beam_cancelling.modes().size() == 2);
    tnsr::I<DataVector, 3> x_cancel{size_t{3}, 0.};
    get<0>(x_cancel) = DataVector{0., 1., -2.5};
    get<1>(x_cancel) = DataVector{0., 0.5, 1.5};
    const DataVector pressure_cancelling =
        pressure_on_z_surface(laser_beam_cancelling, x_cancel);
    const DataVector zero{3, 0.};
    CHECK_ITERABLE_APPROX(pressure_cancelling, zero);
  }

  // Test applying the boundary conditions. The bounds are for x, the polar
  // and azimuthal angles of the face normal, and the beam width.
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Elliptic/Systems/Elasticity/BoundaryConditions/");
  const std::array<std::pair<double, double>, 4> bounds{
      {{-2., 2.}, {0., M_PI}, {-M_PI, M_PI}, {0.5, 2.}}};
  pypp::check_with_random_values<4>(&apply_boundary_condition<false, true>,
                                    "LaserBeam", {"normal_dot_minus_stress"},
                                    bounds, DataVector{3});
  pypp::check_with_random_values<4>(
      &apply_boundary_condition<false, false>, "LaserBeam",
      {"normal_dot_minus_stress_hermite_gauss"}, bounds, DataVector{3});
  pypp::check_with_random_values<4>(
      &apply_boundary_condition<true, true>, "LaserBeam",
      {"normal_dot_minus_stress_linearized"}, bounds, DataVector{3});
  pypp::check_with_random_values<4>(
      &apply_boundary_condition<true, false>, "LaserBeam",
      {"normal_dot_minus_stress_linearized"}, bounds, DataVector{3});
  {
    INFO("Linearized condition overwrites the output");
    // pypp above already fills the result buffer with random values before
    // the call; this direct check makes the requirement explicit.
    tnsr::I<DataVector, 3> displacement{size_t{4}, 1.};
    tnsr::I<DataVector, 3> n_dot_minus_stress{size_t{4}, 1.};
    const tnsr::iJ<DataVector, 3> deriv_displacement{size_t{4}, 1.};
    LaserBeam::apply_linearized(make_not_null(&displacement),
                                make_not_null(&n_dot_minus_stress),
                                deriv_displacement);
    for (size_t d = 0; d < 3; ++d) {
      CHECK(n_dot_minus_stress.get(d) == DataVector(4, 0.));
    }
  }

  {
    INFO("Mode normalization, symmetry and orientation");
    // Uniform grid on the z = 0 plane, wide enough that the profiles vanish
    // at the edges. The trapezoidal rule is then spectrally accurate.
    const double r0 = 1.;
    const size_t num_points_1d = 481;
    const double half_extent = 12. * r0;
    const double spacing =
        2. * half_extent / static_cast<double>(num_points_1d - 1);
    const size_t num_points = square(num_points_1d);
    tnsr::I<DataVector, 3> x{num_points, 0.};
    tnsr::I<DataVector, 3> x_swapped{num_points, 0.};
    for (size_t i = 0; i < num_points_1d; ++i) {
      for (size_t j = 0; j < num_points_1d; ++j) {
        const size_t k = i + num_points_1d * j;
        get<0>(x)[k] = -half_extent + static_cast<double>(i) * spacing;
        get<1>(x)[k] = -half_extent + static_cast<double>(j) * spacing;
        get<0>(x_swapped)[k] = get<1>(x)[k];
        get<1>(x_swapped)[k] = get<0>(x)[k];
      }
    }
    const auto total_force =
        [&x, &r0, &spacing](const std::vector<LaserBeam::ModeInfo>& modes) {
          return sum(pressure_on_z_surface(LaserBeam{r0, modes}, x)) *
                 square(spacing);
        };
    Approx custom_approx = Approx::custom().epsilon(1.e-11).scale(1.);
    // Each mode with unit weight exerts unit total force. The higher orders
    // exercise the Hermite recursion, `pow` and `factorial` at larger n; their
    // profiles still vanish at the grid edges (H_10^2(12) e^{-144} ~ 1e-35).
    for (const auto& [n, m] : std::vector<std::pair<size_t, size_t>>{{0, 0},
                                                                     {0, 2},
                                                                     {2, 0},
                                                                     {1, 3},
                                                                     {3, 3},
                                                                     {4, 1},
                                                                     {10, 0},
                                                                     {0, 10},
                                                                     {6, 6},
                                                                     {8, 3}}) {
      CAPTURE(n);
      CAPTURE(m);
      CHECK(total_force({{n, m, 1.}}) == custom_approx(1.));
    }
    CHECK(total_force({{0, 0, 2.5}}) == custom_approx(2.5));
    CHECK(total_force({{0, 2, 1.}, {2, 0, -1.}}) == custom_approx(0.));
    CHECK(total_force(hg_modes()) == custom_approx(0.5));
    // Overlap integrals of the squared profiles, normalized by TEM00's.
    // Thermal-noise estimates for a thin coating on a half space scale with
    // the integral of the squared profile, so these ratios are physically
    // meaningful. With t = u / r_0 the 1D integrals are
    // int e^{-2t^2} dt = sqrt(pi/2), int H_2^2 e^{-2t^2} dt = 3 sqrt(pi/2)
    // and int H_2^4 e^{-2t^2} dt = 41 sqrt(pi/2), which give
    //   int p_00^2 dA = 1 / (2 pi r_0^2),
    //   int p_20^2 dA / int p_00^2 dA = 41 / 64,
    //   int p_02 p_20 dA / int p_00^2 dA = 9 / 64,
    //   int (p_02 - p_20)^2 dA / int p_00^2 dA = 2 * 41/64 - 2 * 9/64 = 1.
    const auto overlap = [&x, &r0, &spacing](
                             const std::vector<LaserBeam::ModeInfo>& modes_a,
                             const std::vector<LaserBeam::ModeInfo>& modes_b) {
      const DataVector product =
          pressure_on_z_surface(LaserBeam{r0, modes_a}, x) *
          pressure_on_z_surface(LaserBeam{r0, modes_b}, x);
      return sum(product) * square(spacing);
    };
    const double tem00_overlap = overlap(tem00_modes(), tem00_modes());
    CHECK(tem00_overlap == custom_approx(1. / (2. * M_PI * square(r0))));
    CHECK(overlap({{2, 0, 1.}}, {{2, 0, 1.}}) / tem00_overlap ==
          custom_approx(41. / 64.));
    CHECK(overlap({{0, 2, 1.}}, {{2, 0, 1.}}) / tem00_overlap ==
          custom_approx(9. / 64.));
    const std::vector<LaserBeam::ModeInfo> differential{{0, 2, 1.},
                                                        {2, 0, -1.}};
    CHECK(overlap(differential, differential) / tem00_overlap ==
          custom_approx(1.));
    // TEM_nm(x, y) = TEM_mn(y, x)
    CHECK_ITERABLE_APPROX(
        pressure_on_z_surface(LaserBeam{r0, {{0, 2, 1.}}}, x),
        pressure_on_z_surface(LaserBeam{r0, {{2, 0, 1.}}}, x_swapped));
    // For a surface normal to z, N is the Hermite index along x. H_2 has its
    // zeros at +-1/sqrt(2), so TEM20 vanishes at x = r_0 / sqrt(2), y = 0
    // but not at x = 0, y = r_0 / sqrt(2).
    tnsr::I<DataVector, 3> x_nodes{size_t{2}, 0.};
    get<0>(x_nodes)[0] = r0 / sqrt(2.);
    get<1>(x_nodes)[1] = r0 / sqrt(2.);
    const auto pressure_tem20 =
        pressure_on_z_surface(LaserBeam{r0, {{2, 0, 1.}}}, x_nodes);
    CHECK(pressure_tem20[0] == approx(0.));
    // H_2(0)^2 e^{-1/2} / (pi r_0^2 2^2 2!) with r_0 = 1
    CHECK(pressure_tem20[1] == approx(exp(-0.5) / (2. * M_PI)));
  }

  {
    INFO("In-plane basis for normals close to the x-axis");
    // For |n_x| > 0.9 the reference tangent is y instead of x, so u is the
    // normalized projection of y into the surface and v = n x u. The random
    // normals above rarely reach this branch, so check it with deterministic
    // normals and the node of H_2 at r_0 / sqrt(2): TEM20 vanishes on the
    // u-axis there and TEM02 on the v-axis. The third normal is not
    // axis-aligned and exactly normalized: n = (-12, 5, 0) / 13 gives
    // u = (5, 12, 0) / 13 and v = (0, 0, -1).
    const double r0 = 1.5;
    const double node = r0 / sqrt(2.);
    // H_2(0)^2 e^{-1/2} / (pi r_0^2 2^2 2!)
    const double pressure_at_node = exp(-0.5) / (2. * M_PI * square(r0));
    using Vector = std::array<double, 3>;
    for (const auto& [normal, u, v] :
         std::vector<std::tuple<Vector, Vector, Vector>>{
             {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}},
             {{{-1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., -1.}}},
             {{{-12. / 13., 5. / 13., 0.}},
              {{5. / 13., 12. / 13., 0.}},
              {{0., 0., -1.}}}}) {
      INFO("n = (" << gsl::at(normal, 0) << ", " << gsl::at(normal, 1) << ", "
                   << gsl::at(normal, 2) << ")");
      // Points at the node along u, at the node along v, and the latter
      // displaced along the normal (the distance from the surface through
      // the origin must not matter)
      tnsr::I<DataVector, 3> x{size_t{3}, 0.};
      for (size_t d = 0; d < 3; ++d) {
        x.get(d)[0] = node * gsl::at(u, d);
        x.get(d)[1] = node * gsl::at(v, d);
        x.get(d)[2] = node * gsl::at(v, d) + 3. * r0 * gsl::at(normal, d);
      }
      const auto tem20 =
          pressure_on_surface(LaserBeam{r0, {{2, 0, 1.}}}, x, normal);
      CHECK(tem20[0] == approx(0.));
      CHECK(tem20[1] == approx(pressure_at_node));
      CHECK(tem20[2] == approx(pressure_at_node));
      const auto tem02 =
          pressure_on_surface(LaserBeam{r0, {{0, 2, 1.}}}, x, normal);
      CHECK(tem02[0] == approx(pressure_at_node));
      CHECK(tem02[1] == approx(0.));
      CHECK(tem02[2] == approx(0.));
    }
  }

  {
    INFO("A face normal that varies across the face is rejected");
    // Three points with the deviation at the interior one, so that a check
    // comparing only the endpoints would miss it
    tnsr::I<DataVector, 3> x{size_t{3}, 0.};
    auto face_normal = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
    get<2>(face_normal) = -1.;
    get<0>(face_normal)[1] = 1.;
    get<2>(face_normal)[1] = 0.;
    tnsr::I<DataVector, 3> displacement{
        size_t{3}, std::numeric_limits<double>::signaling_NaN()};
    tnsr::I<DataVector, 3> n_dot_minus_stress{
        size_t{3}, std::numeric_limits<double>::signaling_NaN()};
    const tnsr::iJ<DataVector, 3> deriv_displacement{
        size_t{3}, std::numeric_limits<double>::signaling_NaN()};
    CHECK_THROWS_WITH(
        laser_beam.apply(make_not_null(&displacement),
                         make_not_null(&n_dot_minus_stress), deriv_displacement,
                         x, face_normal),
        Catch::Matchers::ContainsSubstring("assumes a flat surface"));
  }

  {
    INFO("Consistency with half-space mirror solution");
    const DataVector used_for_size{5};
    // Choose an arbitrary set of points on the z=0 surface of the mirror
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> dist(-2. * beam_width, 2. * beam_width);
    auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
        make_not_null(&generator), make_not_null(&dist), used_for_size);
    get<2>(x) = 0.;
    // Choose a constitutive relation with arbitrary parameters
    const ConstitutiveRelations::IsotropicHomogeneous<3> constitutive_relation{
        1., 2.};
    // Get the surface stress from the half-space mirror solution.
    // Disable relative tolerance because we compare with an absolute tolerance
    // in the test below.
    const Solutions::HalfSpaceMirror half_space_mirror{
        beam_width, constitutive_relation, 350, 1e-12, 0.};
    const auto minus_stress_solution = get<Tags::MinusStress<3>>(
        half_space_mirror.variables(x, tmpl::list<Tags::MinusStress<3>>{}));
    // Get the stress normal to the surface. The solution assumes the material
    // extends in positive z-direction, so the normal is (0, 0, -1)
    tnsr::I<DataVector, 3> n_dot_minus_stress_solution{used_for_size.size()};
    get<0>(n_dot_minus_stress_solution) = -get<2, 0>(minus_stress_solution);
    get<1>(n_dot_minus_stress_solution) = -get<2, 1>(minus_stress_solution);
    get<2>(n_dot_minus_stress_solution) = -get<2, 2>(minus_stress_solution);
    auto face_normal = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
    get<2>(face_normal) = -1;
    // Shift the plane where we evaluate the boundary condition along the
    // z-direction, just because it shouldn't affect the result and it might
    // catch issues with computing the coordinate distance
    get<2>(x) += 2.;
    // Compare to the boundary conditions
    tnsr::I<DataVector, 3> displacement{
        used_for_size.size(), std::numeric_limits<double>::signaling_NaN()};
    tnsr::I<DataVector, 3> n_dot_minus_stress{
        used_for_size.size(), std::numeric_limits<double>::signaling_NaN()};
    const tnsr::iJ<DataVector, 3> deriv_displacement{
        used_for_size.size(), std::numeric_limits<double>::signaling_NaN()};
    laser_beam.apply(make_not_null(&displacement),
                     make_not_null(&n_dot_minus_stress), deriv_displacement, x,
                     face_normal);
    for (size_t d = 0; d < 3; ++d) {
      CAPTURE(d);
      CHECK_ITERABLE_APPROX(n_dot_minus_stress.get(d),
                            n_dot_minus_stress_solution.get(d));
    }
  }
}

}  // namespace Elasticity::BoundaryConditions
