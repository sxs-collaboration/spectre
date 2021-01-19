// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <size_t Dim>
struct ProductOfSinusoidsProxy : Poisson::Solutions::ProductOfSinusoids<Dim> {
  using Poisson::Solutions::ProductOfSinusoids<Dim>::ProductOfSinusoids;

  using field_tags = tmpl::list<
      Poisson::Tags::Field,
      ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>>;
  using source_tags = tmpl::list<Tags::FixedSource<Poisson::Tags::Field>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::ProductOfSinusoids<Dim>::variables(x,
                                                                  field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::ProductOfSinusoids<Dim>::variables(
        x, source_tags{});
  }
};

template <size_t Dim>
void test_solution(const std::array<double, Dim>& wave_numbers,
                   const std::string& options) {
  const ProductOfSinusoidsProxy<Dim> solution(wave_numbers);
  pypp::check_with_random_values<1>(
      &ProductOfSinusoidsProxy<Dim>::field_variables, solution,
      "ProductOfSinusoids", {"field", "field_gradient"}, {{{0., 2. * M_PI}}},
      std::make_tuple(wave_numbers), DataVector(5));
  pypp::check_with_random_values<1>(
      &ProductOfSinusoidsProxy<Dim>::source_variables, solution,
      "ProductOfSinusoids", {"source"}, {{{0., 2. * M_PI}}},
      std::make_tuple(wave_numbers), DataVector(5));

  const auto created_solution =
      TestHelpers::test_creation<Poisson::Solutions::ProductOfSinusoids<Dim>>(
          "WaveNumbers: " + options);
  CHECK(created_solution == solution);
  test_serialization(solution);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Poisson.ProductOfSinusoids",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};

  using AffineMap = domain::CoordinateMaps::Affine;
  {
    INFO("1D");
    test_solution<1>({{0.5}}, "[0.5]");

    using system =
        Poisson::FirstOrderSystem<1, Poisson::Geometry::FlatCartesian>;
    const Poisson::Solutions::ProductOfSinusoids<1> solution{{{0.5}}};
    const typename system::fluxes fluxes_computer{};
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap>
        coord_map{{-1., 1., 0., M_PI}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, fluxes_computer, coord_map, 1.e5, 3.,
        [](const auto&... /*unused*/) noexcept { return std::tuple<>{}; });
  }
  {
    INFO("2D");
    test_solution<2>({{0.5, 1.}}, "[0.5, 1.]");

    using system =
        Poisson::FirstOrderSystem<2, Poisson::Geometry::FlatCartesian>;
    const Poisson::Solutions::ProductOfSinusoids<2> solution{{{0.5, 0.5}}};
    const typename system::fluxes fluxes_computer{};
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap2D>
        coord_map{{{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, fluxes_computer, coord_map, 1.e5, 3.,
        [](const auto&... /*unused*/) noexcept { return std::tuple<>{}; });
  }
  {
    INFO("3D");
    test_solution<3>({{1., 0.5, 1.5}}, "[1., 0.5, 1.5]");

    using system =
        Poisson::FirstOrderSystem<3, Poisson::Geometry::FlatCartesian>;
    const Poisson::Solutions::ProductOfSinusoids<3> solution{{{0.5, 0.5, 0.5}}};
    const typename system::fluxes fluxes_computer{};
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap3D>
        coord_map{
            {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, fluxes_computer, coord_map, 1.e5, 3.,
        [](const auto&... /*unused*/) noexcept { return std::tuple<>{}; });
  }
}
