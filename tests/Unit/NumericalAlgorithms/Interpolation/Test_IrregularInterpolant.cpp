// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"         // IWYU pragma: keep
#include "DataStructures/VariablesHelpers.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/PowX.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"
// IWYU pragma: no_forward_declare MathFunction
// IWYU pragma: no_forward_declare PowX
// IWYU pragma: no_forward_declare Tensor

namespace {

using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

const double inertial_coord_min = -0.3;
const double inertial_coord_max = 0.7;

template <size_t VolumeDim>
auto make_affine_map() noexcept;

template <>
auto make_affine_map<1>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max});
}

template <>
auto make_affine_map<2>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

template <>
auto make_affine_map<3>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

namespace TestTags {

template <size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
  static std::string name() noexcept { return "Vector"; }
  static auto fill_values(const MathFunctions::TensorProduct<Dim>& f,
                          const tnsr::I<DataVector, Dim>& x) noexcept {
    auto result = make_with_value<tnsr::I<DataVector, Dim>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      result.get(d) = (d + 0.5) * get(f_of_x);
    }
    return result;
  }
};

template <size_t Dim>
struct SymmetricTensor : db::SimpleTag {
  using type = tnsr::ii<DataVector, Dim>;
  static std::string name() noexcept { return "SymmetricTensor"; }
  static auto fill_values(const MathFunctions::TensorProduct<Dim>& f,
                          const tnsr::I<DataVector, Dim>& x) noexcept {
    auto result = make_with_value<tnsr::ii<DataVector, Dim>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {  // Symmetry
        result.get(i, j) = (i + j + 0.33) * get(f_of_x);
      }
    }
    return result;
  }
};

}  // namespace TestTags

template <size_t Dim>
void test_interpolate_to_points(const Mesh<Dim>& mesh) noexcept {
  // Fill target interpolation coordinates with random values
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(inertial_coord_min, inertial_coord_max);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const size_t number_of_points = 6;
  const auto target_x_inertial =
      make_with_random_values<tnsr::I<DataVector, Dim>>(
          nn_generator, nn_dist, DataVector(number_of_points));

  const auto coordinate_map = make_affine_map<Dim>();
  const auto target_x = [&target_x_inertial, &coordinate_map,
                         &number_of_points]() {
    tnsr::I<DataVector, Dim, Frame::Logical> result(number_of_points);
    for (size_t s = 0; s < number_of_points; ++s) {
      tnsr::I<double, Dim> x_inertial_local{};
      for (size_t d = 0; d < Dim; ++d) {
        x_inertial_local.get(d) = target_x_inertial.get(d)[s];
      }
      const auto x_local = coordinate_map.inverse(x_inertial_local).get();
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d)[s] = x_local.get(d);
      }
    }
    return result;
  }();

  // Set up interpolator. Need do this only once.
  const intrp::Irregular<Dim> irregular_interpolant(mesh, target_x);
  test_serialization(irregular_interpolant);

  // ... but we construct another interpolator to test operator!=
  {
    auto target_x_new = target_x;
    target_x_new.get(0)[0] *= 0.98;  // Change one point slightly.
    const intrp::Irregular<Dim> irregular_interpolant_new(mesh, target_x_new);
    CHECK(irregular_interpolant_new != irregular_interpolant);
  }

  // Coordinates on the grid
  const auto src_x = coordinate_map(logical_coordinates(mesh));

  // Set up variables
  using tags =
      tmpl::list<TestTags::Vector<Dim>, TestTags::SymmetricTensor<Dim>>;
  Variables<tags> src_vars(mesh.number_of_grid_points());
  Variables<tags> expected_dest_vars(number_of_points);

  // We will make polynomials of the form x^a y^b z^c ...
  // for all a,b,c, that result in exact interpolation.
  // IndexIterator loops over "a,b,c"
  for (IndexIterator<Dim> iter(mesh.extents()); iter; ++iter) {
    // Set up analytic solution.  We fill a Variables with this solution,
    // interpolate to arbitrary points, and then check that the
    // values at arbitrary points match this solution.
    // We choose polynomials so that interpolation is exact on an LGL grid.
    std::array<std::unique_ptr<MathFunction<1>>, Dim> functions;
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(functions, d) = std::make_unique<MathFunctions::PowX>(iter()[d]);
    }
    MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));

    // Fill source and expected destination Variables with analytic solution.
    tmpl::for_each<tags>([
      &f, &src_x, &target_x_inertial, &src_vars, &expected_dest_vars
    ](auto tag) noexcept {
      using Tag = tmpl::type_from<decltype(tag)>;
      get<Tag>(src_vars) = Tag::fill_values(f, src_x);
      get<Tag>(expected_dest_vars) = Tag::fill_values(f, target_x_inertial);
    });

    // Interpolate
    // (g++ 7.2.0 does not allow `const auto dest_vars` here)
    const Variables<tags> dest_vars =
        irregular_interpolant.interpolate(src_vars);

    tmpl::for_each<tags>([&dest_vars, &expected_dest_vars ](auto tag) noexcept {
      using Tag = tmpl::type_from<decltype(tag)>;
      CHECK_ITERABLE_APPROX(get<Tag>(dest_vars), get<Tag>(expected_dest_vars));
    });
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.IrregularInterpolant",
                  "[Unit][NumericalAlgorithms]") {
  const size_t start_points = 4;
  const size_t end_points = 6;
  for (size_t n0 = start_points; n0 < end_points; ++n0) {
    test_interpolate_to_points<1>(Mesh<1>{n0, Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto});
    for (size_t n1 = start_points; n1 < end_points; ++n1) {
      test_interpolate_to_points<2>(
          Mesh<2>{{{n0, n1}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
      for (size_t n2 = start_points; n2 < end_points; ++n2) {
        test_interpolate_to_points<3>(
            Mesh<3>{{{n0, n1, n2}},
                    Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto});
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.IrregularInterpolant.Meshes",
                  "[Unit][NumericalAlgorithms]") {
  const size_t start_points = 4;
  const size_t end_points = 6;
  for (size_t n0 = start_points; n0 < end_points; ++n0) {
    test_interpolate_to_points<1>(
        Mesh<1>{n0, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss});
    for (size_t n1 = start_points; n1 < end_points; ++n1) {
      test_interpolate_to_points<2>(Mesh<2>{
          {{n0, n1}}, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss});
      test_interpolate_to_points<2>(Mesh<2>{
          {{n0, n1}},
          {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
          {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}}});
      for (size_t n2 = start_points; n2 < end_points; ++n2) {
        test_interpolate_to_points<3>(Mesh<3>{{{n0, n1, n2}},
                                              Spectral::Basis::Legendre,
                                              Spectral::Quadrature::Gauss});
        test_interpolate_to_points<3>(Mesh<3>{
            {{n0, n1, n2}},
            {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
              Spectral::Basis::Legendre}},
            {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
              Spectral::Quadrature::GaussLobatto}}});
      }
    }
  }
}
