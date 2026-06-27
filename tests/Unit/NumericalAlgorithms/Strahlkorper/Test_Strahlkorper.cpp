// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/StrahlkorperTestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/Strahlkorper/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/Strahlkorper/IO/InitialShapeFromFile.hpp"
#include "NumericalAlgorithms/Strahlkorper/InitialShape.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct Inertial;
}  // namespace Frame

namespace ylm {
namespace {
struct TestCreationMetavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<ylm::InitialShape<Frame::Inertial>,
                   tmpl::list<ylm::InitialShapes::Sphere<Frame::Inertial>,
                              ylm::InitialShapes::FromFile<Frame::Inertial>>>>;
  };
};

void test_invert_spec_phys_transform() {
  const double avg_radius = 1.0;
  const double delta_radius = 0.1;
  const size_t l_grid = 33;
  const auto l_grid_high_res = static_cast<size_t>(l_grid * 1.5);
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};

  // Create radius as a function of angle
  DataVector radius(ylm::Spherepack::physical_size(l_grid, l_grid), avg_radius);
  {
    std::uniform_real_distribution<double> ran(0.0, 1.0);
    MAKE_GENERATOR(gen);
    for (auto& r : radius) {
      r += delta_radius * ran(gen);
    }
  }
  CAPTURE(radius);

  // Initialize a strahlkorper of l_max=l_grid
  const Strahlkorper<Frame::Inertial> sk(l_grid, l_grid, radius, center);

  // Put that Strahlkorper onto a larger grid
  const Strahlkorper<Frame::Inertial> sk_high_res(l_grid_high_res,
                                                  l_grid_high_res, sk);

  // Compare coefficients
  SpherepackIterator iter(sk.l_max(), sk.m_max());
  SpherepackIterator iter_high_res(sk_high_res.l_max(), sk_high_res.m_max());
  const auto& init_coefs = sk.coefficients();
  const auto& final_coefs = sk_high_res.coefficients();

  for (size_t l = 0; l <= sk.ylm_spherepack().l_max(); ++l) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      CHECK(init_coefs[iter.set(l, m)()] ==
            approx(final_coefs[iter_high_res.set(l, m)()]));
    }
  }

  for (size_t l = sk.ylm_spherepack().l_max() + 1;
       l <= sk_high_res.ylm_spherepack().l_max(); ++l) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      CHECK(final_coefs[iter_high_res.set(l, m)()] == approx(0.0));
    }
  }
}

void test_phys_spec_constructor_consistency() {
  const size_t l_max = 12;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const size_t physical_size = ylm::Spherepack::physical_size(l_max, l_max);

  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  MAKE_GENERATOR(generator);

  // create Strahlkorper using radius at collocation points
  const auto radius = make_with_random_values<DataVector>(
      make_not_null(&generator), distribution,
      DataVector(physical_size, std::numeric_limits<double>::signaling_NaN()));
  const Strahlkorper<Frame::Inertial> strahlkorper_physical(l_max, l_max,
                                                            radius, center);

  // create Strahlkorper using spectral coefficients
  const size_t spectral_size = ylm::Spherepack::spectral_size(l_max, l_max);
  ModalVector spectral_coefficients(
      spectral_size, std::numeric_limits<double>::signaling_NaN());
  std::copy(strahlkorper_physical.coefficients().begin(),
            strahlkorper_physical.coefficients().end(),
            spectral_coefficients.begin());
  const Strahlkorper<Frame::Inertial> strahlkorper_spectral(
      l_max, l_max, spectral_coefficients, center);

  CHECK(strahlkorper_physical == strahlkorper_spectral);
}

void test_average_radius() {
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const double r = 3.0;
  Strahlkorper<Frame::Inertial> s(4, 4, r, center);
  CHECK(s.average_radius() == approx(r));
}

void test_copy_and_move() {
  Strahlkorper<Frame::Inertial> s(4, 4, 3.0, {{0.1, 0.2, 0.3}});

  test_copy_semantics(s);
  auto s_copy = s;
  test_move_semantics(std::move(s), s_copy);
}

void test_physical_center() {
  const std::array<double, 3> physical_center = {{1.5, 0.5, 1.0}};
  const std::array<double, 3> expansion_center = {{0.0, 0.0, 0.0}};
  const double radius = 5.0;
  const int l_max = 9;

  Strahlkorper<Frame::Inertial> sk(l_max, l_max, radius, expansion_center);
  DataVector r(sk.ylm_spherepack().physical_size(), 0.);

  for (size_t s = 0; s < r.size(); ++s) {
    const double theta = sk.ylm_spherepack().theta_phi_points()[0][s];
    const double phi = sk.ylm_spherepack().theta_phi_points()[1][s];
    // Compute the distance (radius as a function of theta,phi) from
    // the expansion_center to a spherical surface of radius `radius`
    // centered at physical_center.
    const double a = 1.0;
    const double b = -2.0 * cos(phi) * sin(theta) * physical_center[0] -
                     2.0 * sin(phi) * sin(theta) * physical_center[1] -
                     2.0 * cos(theta) * physical_center[2];
    const double c = square(physical_center[0]) + square(physical_center[1]) +
                     square(physical_center[2]) - square(radius);
    auto roots = *real_roots(a, b, c);
    r[s] = std::max(roots[0], roots[1]);
  }
  // Construct a new Strahlkorper sk_test with the radius computed
  // above, centered at expansion_center, so that
  // sk_test.physical_center() should recover the physical center of
  // this surface.
  Strahlkorper<Frame::Inertial> sk_test(l_max, l_max, r, expansion_center);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(approx(gsl::at(physical_center, i)) ==
          gsl::at(sk_test.physical_center(), i));
  }
}

void test_point_is_contained() {
  // Construct a spherical Strahlkorper
  const double radius = 2.;
  const std::array<double, 3> center = {{-1.2, 3., 4.}};
  const Strahlkorper<Frame::Inertial> sphere(3, 2, radius, center);

  // Check whether two known points are contained.
  const std::array<double, 3> point_inside = {{-1.2, 1.01, 4.}};
  const std::array<double, 3> point_outside = {{-1.2, 3., 6.01}};
  CHECK(sphere.point_is_contained(point_inside));
  CHECK_FALSE(sphere.point_is_contained(point_outside));
}

template <typename Func>
void test_constructor_with_different_coefs(Func function) {
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const double r = 3.0;
  const double add_to_r = 1.34;
  Strahlkorper<Frame::Inertial> strahlkorper(4, 4, r, center);
  const Strahlkorper<Frame::Inertial> strahlkorper_test1(4, 4, r + add_to_r,
                                                         center);

  // Modify the 0,0 coefficient to add a constant to the radius.
  const auto strahlkorper_test2 = function(strahlkorper, add_to_r);

  CHECK_ITERABLE_APPROX(strahlkorper_test1.coefficients(),
                        strahlkorper_test2.coefficients());
}

// Helper function to create a random Strahlkorper and write it to an H5 file
Strahlkorper<Frame::Inertial> create_and_write_test_strahlkorper(
    const std::string& filename, const std::string& subfile_name,
    const size_t l_max, const std::array<double, 3>& center,
    const double time) {
  MAKE_GENERATOR(gen);
  const std::uniform_real_distribution<> dist(0.5, 2.0);
  const auto radius = make_with_random_values<DataVector>(
      make_not_null(&gen), dist,
      DataVector(ylm::Spherepack::physical_size(l_max, l_max),
                 std::numeric_limits<double>::signaling_NaN()));
  Strahlkorper<Frame::Inertial> strahlkorper(l_max, l_max, radius, center);

  h5::H5File<h5::AccessType::ReadWrite> h5_file(filename, true);
  std::vector<std::string> legend{};
  std::vector<double> data{};
  ylm::fill_ylm_legend_and_data(make_not_null(&legend), make_not_null(&data),
                                strahlkorper, time, l_max);
  auto& dat_file = h5_file.insert<h5::Dat>("/" + subfile_name, legend);
  dat_file.append(std::vector<std::vector<double>>{data});
  h5_file.close_current_object();

  return strahlkorper;
}

// Helper function to parse Strahlkorper options from a string
Strahlkorper<Frame::Inertial> parse_strahlkorper_from_options(
    const std::string& options_string) {
  Options::Parser<tmpl::list<OptionTags::Strahlkorper<Frame::Inertial>>> opts(
      "");
  opts.parse(options_string);
  return opts.get<OptionTags::Strahlkorper<Frame::Inertial>,
                  TestCreationMetavariables>();
}

void test_construct_from_options() {
  // Test construction from Radius and Center
  {
    CHECK(parse_strahlkorper_from_options("Strahlkorper:\n"
                                          " InitialL : 6\n"
                                          " InitialShape:\n"
                                          "   Sphere:\n"
                                          "     Center: [1.,2.,3.]\n"
                                          "     Radius: 4.5\n") ==
          Strahlkorper<Frame::Inertial>(6, 6, 4.5, {{1., 2., 3.}}));
  }

  // Test construction from file
  {
    const std::string test_filename = "TestStrahlkorperOptions.h5";
    const std::string subfile_name = "TestSurface_Ylm";
    const std::array<double, 3> expansion_center{{1.5, -0.5, 2.0}};
    const size_t l_max_original = 4;
    const double time = 1.23;

    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }

    const auto original_strahlkorper = create_and_write_test_strahlkorper(
        test_filename, subfile_name, l_max_original, expansion_center, time);

    // Test reading from file with the same l_max (no prolong/restrict)
    {
      const auto read_strahlkorper = parse_strahlkorper_from_options(
          "Strahlkorper:\n"
          " InitialL : 4\n"
          " InitialShape:\n"
          "   FromFile:\n"
          "     H5Filename: TestStrahlkorperOptions.h5\n"
          "     SubfileName: TestSurface_Ylm\n"
          "     Time: 1.23\n"
          "     TimeEpsilon: 1.0e-10\n"
          "     CheckFrame: true\n");

      CHECK(read_strahlkorper.l_max() == l_max_original);
      CHECK(read_strahlkorper.m_max() == l_max_original);
      CHECK(read_strahlkorper.expansion_center() == expansion_center);
      CHECK_ITERABLE_APPROX(read_strahlkorper.coefficients(),
                            original_strahlkorper.coefficients());
    }

    // Test reading from file with prolong to higher l_max
    {
      const size_t l_max_requested = 6;
      const auto read_strahlkorper = parse_strahlkorper_from_options(
          "Strahlkorper:\n"
          " InitialL : 6\n"
          " InitialShape:\n"
          "   FromFile:\n"
          "     H5Filename: TestStrahlkorperOptions.h5\n"
          "     SubfileName: TestSurface_Ylm\n"
          "     Time: 1.23\n"
          "     TimeEpsilon: 1.0e-10\n"
          "     CheckFrame: true\n");

      CHECK(read_strahlkorper.l_max() == l_max_requested);
      CHECK(read_strahlkorper.m_max() == l_max_requested);
      CHECK(read_strahlkorper.expansion_center() == expansion_center);

      const Strahlkorper<Frame::Inertial> expected_prolonged(
          l_max_requested, l_max_requested, original_strahlkorper);
      CHECK_ITERABLE_APPROX(read_strahlkorper.coefficients(),
                            expected_prolonged.coefficients());
    }

    // Test reading from file with restrict to lower l_max
    {
      const auto read_strahlkorper = parse_strahlkorper_from_options(
          "Strahlkorper:\n"
          " InitialL : 2\n"
          " InitialShape:\n"
          "   FromFile:\n"
          "     H5Filename: TestStrahlkorperOptions.h5\n"
          "     SubfileName: TestSurface_Ylm\n"
          "     Time: 1.23\n"
          "     TimeEpsilon: 1.0e-10\n"
          "     CheckFrame: true\n");

      CHECK(read_strahlkorper.l_max() == 2);
      CHECK(read_strahlkorper.m_max() == 2);
      CHECK(read_strahlkorper.expansion_center() == expansion_center);

      const Strahlkorper<Frame::Inertial> expected_restricted(
          2, 2, original_strahlkorper);
      CHECK_ITERABLE_APPROX(read_strahlkorper.coefficients(),
                            expected_restricted.coefficients());
    }

    file_system::rm(test_filename, true);
  }

  // Test failure case: missing H5 file
  {
    Options::Parser<tmpl::list<OptionTags::Strahlkorper<Frame::Inertial>>> opts(
        "");
    opts.parse(
        "Strahlkorper:\n"
        " InitialL : 4\n"
        " InitialShape:\n"
        "   FromFile:\n"
        "     H5Filename: NonexistentFile.h5\n"
        "     SubfileName: TestSurface_Ylm\n"
        "     Time: 1.23\n"
        "     TimeEpsilon: 1.0e-10\n"
        "     CheckFrame: true\n");
    CHECK_THROWS_WITH((opts.get<OptionTags::Strahlkorper<Frame::Inertial>,
                                TestCreationMetavariables>()),
                      Catch::Matchers::ContainsSubstring(
                          "Trying to open the file 'NonexistentFile.h5'") &&
                          Catch::Matchers::ContainsSubstring("does not exist"));
  }

  // Test failure case: error reading from file (invalid subfile)
  {
    const std::string test_filename = "TestStrahlkorperOptionsFailure.h5";

    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }

    create_and_write_test_strahlkorper(test_filename, "ValidSurface", 4,
                                       {{1.5, -0.5, 2.0}}, 1.23);

    Options::Parser<tmpl::list<OptionTags::Strahlkorper<Frame::Inertial>>> opts(
        "");
    opts.parse(
        "Strahlkorper:\n"
        " InitialL : 4\n"
        " InitialShape:\n"
        "   FromFile:\n"
        "     H5Filename: TestStrahlkorperOptionsFailure.h5\n"
        "     SubfileName: InvalidSurface\n"
        "     Time: 1.23\n"
        "     TimeEpsilon: 1.0e-10\n"
        "     CheckFrame: true\n");
    CHECK_THROWS_WITH(
        (opts.get<OptionTags::Strahlkorper<Frame::Inertial>,
                  TestCreationMetavariables>()),
        Catch::Matchers::ContainsSubstring("Cannot open the object"));

    file_system::rm(test_filename, true);
  }

  // Test failure case: time differs by more than epsilon.
  {
    const std::string test_filename = "TestStrahlkorperOptionsTimeEpsilon.h5";

    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }

    create_and_write_test_strahlkorper(test_filename, "TestSurface_Ylm", 4,
                                       {{1.5, -0.5, 2.0}}, 1.23);

    Options::Parser<tmpl::list<OptionTags::Strahlkorper<Frame::Inertial>>> opts(
        "");
    opts.parse(
        "Strahlkorper:\n"
        " InitialL : 4\n"
        " InitialShape:\n"
        "   FromFile:\n"
        "     H5Filename: TestStrahlkorperOptionsTimeEpsilon.h5\n"
        "     SubfileName: TestSurface_Ylm\n"
        "     Time: 1.2300000003\n"  // Differs by 3.0e-10 from actual time
        "     TimeEpsilon: 1.0e-10\n"
        "     CheckFrame: true\n");
    CHECK_THROWS_WITH(
        (opts.get<OptionTags::Strahlkorper<Frame::Inertial>,
                  TestCreationMetavariables>()),
        Catch::Matchers::ContainsSubstring("Could not find time"));

    file_system::rm(test_filename, true);
  }
}

void test_strahlkorper_from_other_strahlkorper() {
  const Strahlkorper<Frame::Inertial> inertial_strahlkorper{
      4_st, 1.2, std::array{1.0, 2.0, 3.0}};
  Strahlkorper<Frame::Grid> grid_strahlkorper{inertial_strahlkorper};

  const auto check_equal = [](const auto& s1, const auto& s2) {
    CHECK(s1.coefficients() == s2.coefficients());
    CHECK(s1.l_max() == s2.l_max());
    CHECK(s1.m_max() == s2.m_max());
    CHECK(s1.expansion_center() == s2.expansion_center());
  };

  check_equal(inertial_strahlkorper, grid_strahlkorper);

  grid_strahlkorper = Strahlkorper<Frame::Grid>{
      inertial_strahlkorper.coefficients(), inertial_strahlkorper};

  check_equal(inertial_strahlkorper, grid_strahlkorper);

  grid_strahlkorper =
      Strahlkorper<Frame::Grid>{8_st, 8_st, inertial_strahlkorper};

  check_equal(Strahlkorper<Frame::Grid>(8_st, 1.2, std::array{1.0, 2.0, 3.0}),
              grid_strahlkorper);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Strahlkorper.Strahlkorper",
                  "[NumericalAlgorithms][Unit]") {
  test_invert_spec_phys_transform();
  test_phys_spec_constructor_consistency();
  test_copy_and_move();
  test_average_radius();
  test_physical_center();
  test_point_is_contained();
  test_constructor_with_different_coefs(
      [](Strahlkorper<Frame::Inertial>& sk, double add_to_r) {
        auto coefs = sk.coefficients();  // make a copy
        coefs[0] += sqrt(8.0) * add_to_r;
        return Strahlkorper<Frame::Inertial>(coefs, sk);
      });
  test_constructor_with_different_coefs(
      [](const Strahlkorper<Frame::Inertial>& sk, double add_to_r) {
        auto coefs = sk.coefficients();  // make a copy
        coefs[0] += sqrt(8.0) * add_to_r;
        return Strahlkorper<Frame::Inertial>(coefs, sk);
      });
  test_constructor_with_different_coefs(
      [](Strahlkorper<Frame::Inertial>& sk, double add_to_r) {
        auto& coefs = sk.coefficients();  // no copy
        coefs[0] += sqrt(8.0) * add_to_r;
        return Strahlkorper<Frame::Inertial>(std::move(sk));
      });
  test_construct_from_options();
  test_strahlkorper_from_other_strahlkorper();
  {
    Strahlkorper<Frame::Inertial> s(4, 4, 2.0, {{1.0, 2.0, 3.0}});
    test_serialization(s);
  }
}
}  // namespace ylm
