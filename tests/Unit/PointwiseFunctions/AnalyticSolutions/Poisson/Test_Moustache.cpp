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
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
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
      ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>>;
  using source_tags = tmpl::list<Tags::FixedSource<Poisson::Tags::Field>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const {
    return Poisson::Solutions::Moustache<Dim>::variables(x, field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const {
    return Poisson::Solutions::Moustache<Dim>::variables(x, source_tags{});
  }
};

template <size_t Dim>
void test_solution() {
  const MoustacheProxy<Dim> solution{};
  pypp::check_with_random_values<1>(
      &MoustacheProxy<Dim>::field_variables, solution, "Moustache",
      {"field", "field_gradient", "field_flux"}, {{{0., 1.}}},
      std::make_tuple(), DataVector(5));
  pypp::check_with_random_values<1>(
      &MoustacheProxy<Dim>::source_variables, solution, "Moustache", {"source"},
      {{{0., 1.}}}, std::make_tuple(), DataVector(5));

  const auto created_solution =
      TestHelpers::test_creation<Poisson::Solutions::Moustache<Dim>>("");
  CHECK(created_solution == solution);
  test_serialization(solution);
  test_copy_semantics(solution);
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

    using system =
        Poisson::FirstOrderSystem<1, Poisson::Geometry::FlatCartesian>;
    const Poisson::Solutions::Moustache<1> solution{};
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                AffineMap>
        coord_map{{-1., 1., 0., 1.}};
    FirstOrderEllipticSolutionsTestHelpers::
        verify_solution_with_power_law_convergence<system>(solution, coord_map,
                                                           3.e1, 2.5);
  }
  {
    INFO("2D");
    test_solution<2>();

    using system =
        Poisson::FirstOrderSystem<2, Poisson::Geometry::FlatCartesian>;
    const Poisson::Solutions::Moustache<2> solution{};
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                AffineMap2D>
        coord_map{{{-1., 1., 0., 1.}, {-1., 1., 0., 1.}}};
    FirstOrderEllipticSolutionsTestHelpers::
        verify_solution_with_power_law_convergence<system>(solution, coord_map,
                                                           5., 2.);
  }
}
