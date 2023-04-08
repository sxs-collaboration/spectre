// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Poisson::Solutions {
namespace {

template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<::MathFunction<Dim, Frame::Inertial>,
                   MathFunctions::all_math_functions<Dim, Frame::Inertial>>>;
  };
  using component_list = tmpl::list<>;
};

template <size_t Dim>
struct MathFunctionProxy : MathFunction<Dim> {
  using MathFunction<Dim>::MathFunction;

  using field_tags = tmpl::list<
      Poisson::Tags::Field,
      ::Tags::deriv<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::FixedSource<Poisson::Tags::Field>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const {
    return MathFunction<Dim>::variables(x, field_tags{});
  }
};

template <size_t Dim>
void test_solution() {
  CAPTURE(Dim);
  const double amplitude = 0.5;
  const double width = 1.;
  const auto center = make_array<Dim>(0.);
  const MathFunctions::Gaussian<Dim, Frame::Inertial> gaussian{amplitude, width,
                                                               center};
  const MathFunction<Dim> solution{
      std::make_unique<MathFunctions::Gaussian<Dim, Frame::Inertial>>(
          gaussian)};
  register_factory_classes_with_charm<Metavariables<Dim>>();
  {
    INFO("Option-creation");
    const auto created =
        TestHelpers::test_creation<MathFunction<Dim>, Metavariables<Dim>>(
            "Function:\n"
            "  Gaussian:\n"
            "    Amplitude: 0.5\n"
            "    Width: 1.\n"
            "    Center: " +
                []() -> std::string {
              if constexpr (Dim == 1) {
                return "0.";
              } else if constexpr (Dim == 2) {
                return "[0., 0.]";
              } else {
                return "[0., 0., 0.]";
              }
            }());
    CHECK(solution == created);
  }
  {
    INFO("Semantics");
    test_serialization(solution);
    MathFunction<Dim> moved{
        std::make_unique<MathFunctions::Gaussian<Dim, Frame::Inertial>>(
            gaussian)};
    test_move_semantics(std::move(moved), solution);
  }
  {
    INFO("Properties");
    CHECK(solution.math_function() == gaussian);
  }
  {
    INFO("Random-value test");
    const MathFunctionProxy<Dim> proxy(
        std::make_unique<MathFunctions::Gaussian<Dim, Frame::Inertial>>(
            gaussian));
    pypp::check_with_random_values<1>(
        &MathFunctionProxy<Dim>::field_variables, proxy, "Gaussian",
        {"field", "field_gradient", "field_flux", "source"}, {{{0., 1.}}},
        std::make_tuple(amplitude, width, center), DataVector(5));
  }
  {
    INFO("Verify solution solves the Poisson system");
    using system =
        Poisson::FirstOrderSystem<Dim, Poisson::Geometry::FlatCartesian>;
    const auto coord_map = []() {
      using AffineMap = domain::CoordinateMaps::Affine;
      if constexpr (Dim == 1) {
        return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                     AffineMap>{{-1., 1., 0., 2.}};
      } else if constexpr (Dim == 2) {
        using AffineMap2D =
            domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
        return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                     AffineMap2D>{
            {{-1., 1., 0., 2.}, {-1., 1., 0., 2.}}};
      } else {
        using AffineMap3D =
            domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap,
                                                   AffineMap>;
        return domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                     AffineMap3D>{
            {{-1., 1., 0., 2.}, {-1., 1., 0., 2.}, {-1., 1., 0., 2.}}};
      }
    }();
    const Mesh<Dim> mesh{12, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
        solution, mesh, coord_map, 2.e-4);
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Poisson.MathFunction",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};
  test_solution<1>();
  test_solution<2>();
  test_solution<3>();
}

}  // namespace Poisson::Solutions
