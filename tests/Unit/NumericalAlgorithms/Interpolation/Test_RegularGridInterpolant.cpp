// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
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
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/PowX.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare MathFunction
// IWYU pragma: no_forward_declare PowX
// IWYU pragma: no_forward_declare Tensor

namespace {

using Affine = CoordinateMaps::Affine;
using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

const double inertial_coord_min = -0.3;
const double inertial_coord_max = 0.7;

template <size_t VolumeDim>
auto make_affine_map() noexcept;

template <>
auto make_affine_map<1>() noexcept {
  return make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max});
}

template <>
auto make_affine_map<2>() noexcept {
  return make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

template <>
auto make_affine_map<3>() noexcept {
  return make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

namespace TestTags {

template <size_t Dim>
struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Scalar"; }
  static auto fill_values(const MathFunctions::TensorProduct<Dim>& f,
                          const tnsr::I<DataVector, Dim>& x) noexcept {
    return Scalar<DataVector>{{{get(f(x))}}};
  }
};

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

}  // namespace TestTags

// Test interpolation from source_mesh onto target_mesh.
template <size_t Dim>
void test_regular_interpolation(const Mesh<Dim>& source_mesh,
                                const Mesh<Dim>& target_mesh) noexcept {
  CAPTURE(source_mesh);
  CAPTURE(target_mesh);
  const auto map = make_affine_map<Dim>();
  const auto source_coords = map(logical_coordinates(source_mesh));
  const auto target_coords = map(logical_coordinates(target_mesh));

  // Set up variables
  using tags = tmpl::list<TestTags::ScalarTag<Dim>, TestTags::Vector<Dim>>;
  Variables<tags> source_vars(source_mesh.number_of_grid_points());
  Variables<tags> expected_result(target_mesh.number_of_grid_points());

  // Set up interpolator
  const intrp::RegularGrid<Dim> regular_grid_interpolant(source_mesh,
                                                         target_mesh);

  // We will make polynomials of the form x^a y^b z^c ...
  // for all a,b,c, that result in exact interpolation.
  // IndexIterator loops over "a,b,c"
  for (IndexIterator<Dim> iter(source_mesh.extents()); iter; ++iter) {
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
      &f, &source_coords, &target_coords, &source_vars, &expected_result
    ](auto tag) noexcept {
      using Tag = tmpl::type_from<decltype(tag)>;
      get<Tag>(source_vars) = Tag::fill_values(f, source_coords);
      get<Tag>(expected_result) = Tag::fill_values(f, target_coords);
    });

    // Interpolate
    // (g++ 7.2.0 does not allow `const auto result` here)
    const Variables<tags> result =
        regular_grid_interpolant.interpolate(source_vars);

    tmpl::for_each<tags>([&result, &expected_result ](auto tag) noexcept {
      using Tag = tmpl::type_from<decltype(tag)>;
      CHECK_ITERABLE_APPROX(get<Tag>(result), get<Tag>(expected_result));
    });
  }
}

// Test extrapolation from source_mesh to a non-overlapping grid of points. The
// target points share the Mesh source_mesh, but span [1,3] in each dimension.
template <size_t Dim>
void test_regular_extrapolation(const Mesh<Dim>& source_mesh) noexcept {
  CAPTURE(source_mesh);

  const auto map = make_affine_map<Dim>();
  const auto source_coords = map(logical_coordinates(source_mesh));
  const auto target_1d_logical_coords = [&source_mesh]() noexcept {
    auto result = std::array<DataVector, Dim>{{{}}};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(result, d) =
          get<0>(logical_coordinates(source_mesh.slice_through(d))) + 2.0;
    }
    return result;
  }
  ();
  const auto target_coords = [&source_mesh, &map ]() noexcept {
    auto result = logical_coordinates(source_mesh);
    for (size_t d = 0; d < Dim; ++d) {
      result.get(d) += 2.0;
    }
    return map(result);
  }
  ();

  // Set up variables
  using tags = tmpl::list<TestTags::ScalarTag<Dim>, TestTags::Vector<Dim>>;
  Variables<tags> source_vars(source_mesh.number_of_grid_points());
  Variables<tags> expected_result(source_mesh.number_of_grid_points());

  // Set up interpolator
  const intrp::RegularGrid<Dim> regular_grid_interpolant(
      source_mesh, target_1d_logical_coords);

  // Test only the highest-order polynomial x^a y^b z^c on the source mesh.
  std::array<std::unique_ptr<MathFunction<1>>, Dim> functions;
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(functions, d) =
        std::make_unique<MathFunctions::PowX>(source_mesh.extents(d) - 1);
  }
  MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));

  // Fill source and expected destination Variables with analytic solution.
  tmpl::for_each<tags>([
    &f, &source_coords, &target_coords, &source_vars, &expected_result
  ](auto tag) noexcept {
    using Tag = tmpl::type_from<decltype(tag)>;
    get<Tag>(source_vars) = Tag::fill_values(f, source_coords);
    get<Tag>(expected_result) = Tag::fill_values(f, target_coords);
  });

  // Interpolate
  // (g++ 7.2.0 does not allow `const auto result` here)
  const Variables<tags> result =
      regular_grid_interpolant.interpolate(source_vars);

  // For 3D extrapolation, errors get large-ish.
  Approx local_approx = Approx::custom().epsilon(1.e-9).scale(1.0);
  tmpl::for_each<tags>(
      [&result, &expected_result, &local_approx ](auto tag) noexcept {
        using Tag = tmpl::type_from<decltype(tag)>;
        CHECK_ITERABLE_CUSTOM_APPROX(get<Tag>(result),
                                     get<Tag>(expected_result), local_approx);
      });
}

void test_1d_regular_interpolation() {
  const size_t start_points = 4;
  const size_t end_points = 6;
  for (size_t n = start_points; n < end_points; ++n) {
    const auto mesh_gll = Mesh<1>{n, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto};
    const auto mesh_gl =
        Mesh<1>{n, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};
    const auto mesh_gll_high_res = Mesh<1>{n + 2, Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto};
    test_regular_interpolation(mesh_gll, mesh_gll_high_res);
    test_regular_interpolation(mesh_gll, mesh_gl);
    test_regular_interpolation(mesh_gl, mesh_gll);
    test_regular_extrapolation(mesh_gll);
  }
}

void test_2d_regular_interpolation() {
  const size_t start_points = 3;
  const size_t end_points = 5;
  for (size_t nx = start_points; nx < end_points; ++nx) {
    for (size_t ny = start_points; ny < end_points; ++ny) {
      const auto mesh_gll = Mesh<2>{{{nx, ny}},
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
      const auto mesh_gl = Mesh<2>{
          {{nx, ny}}, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};
      const auto mesh_gll_high_res =
          Mesh<2>{{{nx + 2, ny + 3}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto};
      test_regular_interpolation(mesh_gll, mesh_gll_high_res);
      test_regular_interpolation(mesh_gll, mesh_gl);
      test_regular_interpolation(mesh_gl, mesh_gll);
      test_regular_extrapolation(mesh_gll);
    }
  }
}

void test_3d_regular_interpolation() {
  const size_t start_points = 2;
  const size_t end_points = 4;
  for (size_t nx = start_points; nx < end_points; ++nx) {
    for (size_t ny = start_points; ny < end_points; ++ny) {
      for (size_t nz = start_points; nz < end_points; ++nz) {
        const auto mesh_gll = Mesh<3>{{{nx, ny, nz}},
                                      Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto};
        const auto mesh_gl = Mesh<3>{{{nx, ny, nz}},
                                     Spectral::Basis::Legendre,
                                     Spectral::Quadrature::Gauss};
        const auto mesh_gll_high_res =
            Mesh<3>{{{nx + 2, ny + 3, nz + 1}},
                    Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto};
        test_regular_interpolation(mesh_gll, mesh_gll_high_res);
        test_regular_interpolation(mesh_gll, mesh_gl);
        test_regular_interpolation(mesh_gl, mesh_gll);
        test_regular_extrapolation(mesh_gll);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.RegularGridInterpolant",
                  "[Unit][NumericalAlgorithms]") {
  test_1d_regular_interpolation();
  test_2d_regular_interpolation();
  test_3d_regular_interpolation();
}
