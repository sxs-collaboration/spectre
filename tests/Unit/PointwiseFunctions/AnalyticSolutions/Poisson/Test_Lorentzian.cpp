// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
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
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <size_t Dim>
struct LorentzianProxy : Poisson::Solutions::Lorentzian<Dim> {
  using Poisson::Solutions::Lorentzian<Dim>::Lorentzian;

  using field_tags = tmpl::list<
      Poisson::Tags::Field,
      ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>>;
  using source_tags = tmpl::list<Tags::FixedSource<Poisson::Tags::Field>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::Lorentzian<Dim>::variables(x, field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::Lorentzian<Dim>::variables(x, source_tags{});
  }
};

template <size_t Dim>
void test_solution() {
  const LorentzianProxy<Dim> solution{};
  pypp::check_with_random_values<1>(
      &LorentzianProxy<Dim>::field_variables, solution, "Lorentzian",
      {"field", "field_gradient", "field_flux"}, {{{-5., 5.}}},
      std::make_tuple(), DataVector(5));
  pypp::check_with_random_values<1>(
      &LorentzianProxy<Dim>::source_variables, solution, "Lorentzian",
      {"source"}, {{{-5., 5.}}}, std::make_tuple(), DataVector(5));

  const Poisson::Solutions::Lorentzian<Dim> check_solution{};
  const auto created_solution =
      TestHelpers::test_creation<Poisson::Solutions::Lorentzian<Dim>>("");
  CHECK(created_solution == check_solution);
  test_serialization(check_solution);
  test_copy_semantics(check_solution);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Poisson.Lorentzian",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};
  // 1D and 2D solutions are not implemented yet.
  test_solution<3>();

  {
    // Verify that the solution numerically solves the system and that the
    // discretization error decreases exponentially with polynomial order
    using system =
        Poisson::FirstOrderSystem<3, Poisson::Geometry::FlatCartesian>;
    const Poisson::Solutions::Lorentzian<3> solution{};
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                AffineMap3D>
        coord_map{
            {{-1., 1., -0.5, 0.5}, {-1., 1., -0.5, 0.5}, {-1., 1., -0.5, 0.5}}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, coord_map, 5.e1, 1.2,
        [](const auto&... /*unused*/) noexcept { return std::tuple<>{}; });
  }

  {
    // Verify that the solution also solves the non-euclidean system with a
    // Euclidean metric. This is more a test of the system than of the solution.
    using system = Poisson::FirstOrderSystem<3, Poisson::Geometry::Curved>;
    const Poisson::Solutions::Lorentzian<3> solution{};
    using AffineMap = domain::CoordinateMaps::Affine;
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                AffineMap3D>
        coord_map{
            {{-1., 1., -0.5, 0.5}, {-1., 1., -0.5, 0.5}, {-1., 1., -0.5, 0.5}}};
    Mesh<3> mesh{8, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
    const DataVector used_for_size{mesh.number_of_grid_points()};
    auto inv_spatial_metric =
        make_with_value<tnsr::II<DataVector, 3>>(used_for_size, 0.);
    get<0, 0>(inv_spatial_metric) = 1.;
    get<1, 1>(inv_spatial_metric) = 1.;
    get<2, 2>(inv_spatial_metric) = 1.;
    const auto spatial_christoffel_contracted =
        make_with_value<tnsr::i<DataVector, 3>>(used_for_size, 0.);
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
        solution, mesh, coord_map, 0.1, std::make_tuple(inv_spatial_metric),
        std::make_tuple(spatial_christoffel_contracted));
  }
}
