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
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <size_t Dim>
struct MoustacheProxy : Poisson::Solutions::Moustache<Dim> {
  using Poisson::Solutions::Moustache<Dim>::Moustache;

  using field_tags = tmpl::list<
      Poisson::Tags::Field,
      ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>>;
  using source_tags = tmpl::list<Tags::FixedSource<Poisson::Tags::Field>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::Moustache<Dim>::variables(x, field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const noexcept {
    return Poisson::Solutions::Moustache<Dim>::variables(x, source_tags{});
  }
};

template <size_t Dim>
void test_solution() {
  const MoustacheProxy<Dim> solution{};
  pypp::check_with_random_values<
      1, tmpl::list<Poisson::Tags::Field,
                    ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>,
                                  Frame::Inertial>>>(
      &MoustacheProxy<Dim>::field_variables, solution, "Moustache",
      {"field", "field_gradient"}, {{{0., 1.}}}, std::make_tuple(),
      DataVector(5));
  pypp::check_with_random_values<
      1, tmpl::list<Tags::FixedSource<Poisson::Tags::Field>>>(
      &MoustacheProxy<Dim>::source_variables, solution, "Moustache", {"source"},
      {{{0., 1.}}}, std::make_tuple(), DataVector(5));

  const auto created_solution =
      TestHelpers::test_creation<Poisson::Solutions::Moustache<Dim>>("");
  CHECK(created_solution == solution);
  test_serialization(solution);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Poisson.Moustache",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};

  using AffineMap = domain::CoordinateMaps::Affine;
  {
    INFO("1D");
    test_solution<1>();

    using system = Poisson::FirstOrderSystem<1, Poisson::Geometry::Euclidean>;
    const Poisson::Solutions::Moustache<1> solution{};
    const typename system::fluxes fluxes_computer{};
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap>
        coord_map{{-1., 1., 0., 1.}};
    FirstOrderEllipticSolutionsTestHelpers::
        verify_solution_with_power_law_convergence<system>(
            solution, fluxes_computer, coord_map, 3.e1, 2.5);
  }
  {
    INFO("2D");
    test_solution<2>();

    using system = Poisson::FirstOrderSystem<2, Poisson::Geometry::Euclidean>;
    const Poisson::Solutions::Moustache<2> solution{};
    const typename system::fluxes fluxes_computer{};
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap2D>
        coord_map{{{-1., 1., 0., 1.}, {-1., 1., 0., 1.}}};
    FirstOrderEllipticSolutionsTestHelpers::
        verify_solution_with_power_law_convergence<system>(
            solution, fluxes_computer, coord_map, 5., 2.);
  }
}
