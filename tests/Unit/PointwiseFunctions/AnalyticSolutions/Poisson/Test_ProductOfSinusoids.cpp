// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
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
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <size_t Dim, typename DataType>
struct ProductOfSinusoidsProxy
    : Poisson::Solutions::ProductOfSinusoids<Dim, DataType> {
  using Poisson::Solutions::ProductOfSinusoids<Dim,
                                               DataType>::ProductOfSinusoids;

  using field_tags =
      tmpl::list<Poisson::Tags::Field<DataType>,
                 ::Tags::deriv<Poisson::Tags::Field<DataType>,
                               tmpl::size_t<Dim>, Frame::Inertial>,
                 ::Tags::Flux<Poisson::Tags::Field<DataType>, tmpl::size_t<Dim>,
                              Frame::Inertial>>;
  using source_tags =
      tmpl::list<Tags::FixedSource<Poisson::Tags::Field<DataType>>>;

  tuples::tagged_tuple_from_typelist<field_tags> field_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const {
    return Poisson::Solutions::ProductOfSinusoids<Dim, DataType>::variables(
        x, field_tags{});
  }

  tuples::tagged_tuple_from_typelist<source_tags> source_variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x) const {
    return Poisson::Solutions::ProductOfSinusoids<Dim, DataType>::variables(
        x, source_tags{});
  }
};

template <size_t Dim, typename DataType>
void test_solution(const std::array<double, Dim>& wave_numbers,
                   const double complex_phase, const std::string& options) {
  const ProductOfSinusoidsProxy<Dim, DataType> solution(wave_numbers,
                                                        complex_phase);
  pypp::check_with_random_values<1>(
      &ProductOfSinusoidsProxy<Dim, DataType>::field_variables, solution,
      "ProductOfSinusoids", {"field", "field_gradient", "field_flux"},
      {{{0., 2. * M_PI}}}, std::make_tuple(wave_numbers, complex_phase),
      DataVector(5));
  pypp::check_with_random_values<1>(
      &ProductOfSinusoidsProxy<Dim, DataType>::source_variables, solution,
      "ProductOfSinusoids", {"source"}, {{{0., 2. * M_PI}}},
      std::make_tuple(wave_numbers, complex_phase), DataVector(5));

  const auto created_solution = TestHelpers::test_creation<
      Poisson::Solutions::ProductOfSinusoids<Dim, DataType>>(options);
  CHECK(created_solution == solution);
  test_serialization(solution);
  test_copy_semantics(solution);
}

template <typename DataType>
void test_product_of_sinusoids() {
  using AffineMap = domain::CoordinateMaps::Affine;
  double complex_phase = 0.0;
  std::string complex_options{};
  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    complex_phase = 0.7;
    complex_options = "\nComplexPhase: 0.7";
  }
  {
    INFO("1D");
    test_solution<1, DataType>({{0.5}}, complex_phase,
                               "WaveNumbers: [0.5]" + complex_options);

    using system =
        Poisson::FirstOrderSystem<1, Poisson::Geometry::FlatCartesian,
                                  DataType>;
    const Poisson::Solutions::ProductOfSinusoids<1, DataType> solution{
        {{0.5}}, complex_phase};
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                AffineMap>
        coord_map{{-1., 1., 0., M_PI}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, coord_map, 1.e5, 3.,
        [](const auto&... /*unused*/) { return std::tuple<>{}; });
  }
  {
    INFO("2D");
    test_solution<2, DataType>({{0.5, 1.}}, complex_phase,
                               "WaveNumbers: [0.5, 1.]" + complex_options);

    using system =
        Poisson::FirstOrderSystem<2, Poisson::Geometry::FlatCartesian,
                                  DataType>;
    const Poisson::Solutions::ProductOfSinusoids<2, DataType> solution{
        {{0.5, 0.5}}, complex_phase};
    using AffineMap2D =
        domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                AffineMap2D>
        coord_map{{{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, coord_map, 1.e5, 3.,
        [](const auto&... /*unused*/) { return std::tuple<>{}; });
  }
  {
    INFO("3D");
    test_solution<3, DataType>({{1., 0.5, 1.5}}, complex_phase,
                               "WaveNumbers: [1., 0.5, 1.5]" + complex_options);

    using system =
        Poisson::FirstOrderSystem<3, Poisson::Geometry::FlatCartesian,
                                  DataType>;
    const Poisson::Solutions::ProductOfSinusoids<3, DataType> solution{
        {{0.5, 0.5, 0.5}}, complex_phase};
    using AffineMap3D =
        domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                AffineMap3D>
        coord_map{
            {{-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}, {-1., 1., 0., M_PI}}};
    FirstOrderEllipticSolutionsTestHelpers::verify_smooth_solution<system>(
        solution, coord_map, 1.e5, 3.,
        [](const auto&... /*unused*/) { return std::tuple<>{}; });
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Poisson.ProductOfSinusoids",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Poisson"};
  test_product_of_sinusoids<DataVector>();
  test_product_of_sinusoids<ComplexDataVector>();
}
