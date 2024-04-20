// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <string>

#include "Domain/Creators/ShapeMapOptions.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_kerr_schild_boyer_lindquist() {
  const auto kerr_schild_boyer_lindquist = TestHelpers::test_creation<
      domain::creators::time_dependent_options::KerrSchildFromBoyerLindquist>(
      "Mass: 1.7\n"
      "Spin: [0.45, 0.12, 0.34]");
  CHECK(kerr_schild_boyer_lindquist.mass == 1.7);
  CHECK(kerr_schild_boyer_lindquist.spin == std::array{0.45, 0.12, 0.34});
}

void test_shape_map_options() {
  {
    constexpr bool include_transition_ends_at_cube = false;
    constexpr domain::ObjectLabel object_label = domain::ObjectLabel::A;
    CAPTURE(include_transition_ends_at_cube);
    CAPTURE(object_label);
    const auto shape_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::ShapeMapOptions<
            include_transition_ends_at_cube, object_label>>(
        "LMax: 8\n"
        "InitialValues: Spherical\n"
        "SizeInitialValues: [0.5, 1.0, 2.4]");
    CHECK(shape_map_options.name() == "ShapeMapA");
    CHECK(shape_map_options.l_max == 8);
    CHECK_FALSE(shape_map_options.initial_values.has_value());
    CHECK(shape_map_options.initial_size_values.has_value());
    CHECK(shape_map_options.initial_size_values.value() ==
          std::array{0.5, 1.0, 2.4});
    CHECK_FALSE(shape_map_options.transition_ends_at_cube);
  }
  {
    constexpr bool include_transition_ends_at_cube = true;
    constexpr domain::ObjectLabel object_label = domain::ObjectLabel::B;
    CAPTURE(include_transition_ends_at_cube);
    CAPTURE(object_label);
    const auto shape_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::ShapeMapOptions<
            include_transition_ends_at_cube, object_label>>(
        "LMax: 8\n"
        "InitialValues:\n"
        "  Mass: 1.7\n"
        "  Spin: [0.45, 0.12, 0.34]\n"
        "SizeInitialValues: Auto\n"
        "TransitionEndsAtCube: True");
    CHECK(shape_map_options.name() == "ShapeMapB");
    CHECK(shape_map_options.l_max == 8);
    CHECK(shape_map_options.initial_values.has_value());
    CHECK(std::holds_alternative<domain::creators::time_dependent_options::
                                     KerrSchildFromBoyerLindquist>(
        shape_map_options.initial_values.value()));
    CHECK_FALSE(shape_map_options.initial_size_values.has_value());
    CHECK(shape_map_options.transition_ends_at_cube);
  }
}

void test_funcs() {
  const double inner_radius = 0.5;
  const size_t l_max = 8;
  // We choose a Schwarzschild BH so all coefs are zero and it's easy to check
  {
    const auto shape_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::ShapeMapOptions<
            false, domain::ObjectLabel::None>>(
        "LMax: 8\n"
        "InitialValues:\n"
        "  Mass: 1.0\n"
        "  Spin: [0.0, 0.0, 0.0]\n"
        "SizeInitialValues: [0.5, 1.0, 2.4]");

    const auto [shape_funcs, size_funcs] =
        domain::creators::time_dependent_options::initial_shape_and_size_funcs(
            shape_map_options, inner_radius);

    for (size_t i = 0; i < shape_funcs.size(); i++) {
      CHECK(gsl::at(shape_funcs, i) ==
            DataVector{ylm::Spherepack::spectral_size(l_max, l_max), 0.0});
    }
    CHECK(size_funcs == std::array{DataVector{0.5}, DataVector{1.0},
                                   DataVector{2.4}, DataVector{0.0}});
  }
  {
    const auto shape_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::ShapeMapOptions<
            false, domain::ObjectLabel::None>>(
        "LMax: 8\n"
        "InitialValues:\n"
        "  Mass: 1.0\n"
        "  Spin: [0.0, 0.0, 0.0]\n"
        "SizeInitialValues: Auto");

    const auto [shape_funcs, size_funcs] =
        domain::creators::time_dependent_options::initial_shape_and_size_funcs(
            shape_map_options, inner_radius);

    for (size_t i = 0; i < shape_funcs.size(); i++) {
      CHECK(gsl::at(shape_funcs, i) ==
            DataVector{ylm::Spherepack::spectral_size(l_max, l_max), 0.0});
    }
    CHECK(size_funcs == std::array{DataVector{0.0}, DataVector{0.0},
                                   DataVector{0.0}, DataVector{0.0}});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.ShapeMapOptions", "[Domain][Unit]") {
  test_kerr_schild_boyer_lindquist();
  test_shape_map_options();
  test_funcs();
}
