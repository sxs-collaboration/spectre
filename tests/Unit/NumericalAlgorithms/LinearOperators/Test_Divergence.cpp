// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare MathFunction
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Tags::div

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t VolumeDim>
auto make_affine_map();

template <>
auto make_affine_map<1>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine{-1.0, 1.0, -0.3, 0.7});
}

template <>
auto make_affine_map<2>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
}

template <>
auto make_affine_map<3>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}

template <size_t Dim, typename Frame>
struct Flux1 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame>;
  static auto flux(const MathFunctions::TensorProduct<Dim>& f,
                   const tnsr::I<DataVector, Dim, Frame>& x) {
    auto result = make_with_value<tnsr::I<DataVector, Dim, Frame>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      result.get(d) = (d + 0.5) * get(f_of_x);
    }
    return result;
  }
  static auto divergence_of_flux(const MathFunctions::TensorProduct<Dim>& f,
                                 const tnsr::I<DataVector, Dim, Frame>& x) {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.);
    const auto df = f.first_derivatives(x);
    for (size_t d = 0; d < Dim; ++d) {
      get(result) += (d + 0.5) * df.get(d);
    }
    return result;
  }
};

template <size_t Dim, typename Frame>
struct Flux2 : db::SimpleTag {
  using type = tnsr::Ij<DataVector, Dim, Frame>;
  static auto flux(const MathFunctions::TensorProduct<Dim>& f,
                   const tnsr::I<DataVector, Dim, Frame>& x) {
    auto result = make_with_value<tnsr::Ij<DataVector, Dim, Frame>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t j = 0; j < Dim; ++j) {
        result.get(d, j) = (d + 0.5) * (j + 0.25) * get(f_of_x);
      }
    }
    return result;
  }
  static auto divergence_of_flux(const MathFunctions::TensorProduct<Dim>& f,
                                 const tnsr::I<DataVector, Dim, Frame>& x) {
    auto result = make_with_value<tnsr::i<DataVector, Dim, Frame>>(x, 0.);
    const auto df = f.first_derivatives(x);
    for (size_t j = 0; j < Dim; ++j) {
      for (size_t d = 0; d < Dim; ++d) {
        result.get(j) += (d + 0.5) * (j + 0.25) * df.get(d);
      }
    }
    return result;
  }
};

template <size_t Dim, typename Frame>
using two_fluxes = tmpl::list<Flux1<Dim, Frame>, Flux2<Dim, Frame>>;

template <size_t Dim, typename Frame = Frame::Inertial>
void test_divergence_impl(
    const Mesh<Dim>& mesh,
    std::array<std::unique_ptr<MathFunction<1, Frame>>, Dim> functions) {
  const auto coordinate_map = make_affine_map<Dim>();
  const size_t num_grid_points = mesh.number_of_grid_points();
  const auto xi = logical_coordinates(mesh);
  const auto x = coordinate_map(xi);
  const auto inv_jacobian = coordinate_map.inv_jacobian(xi);
  MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));
  using flux_tags = two_fluxes<Dim, Frame>;
  Variables<flux_tags> fluxes(num_grid_points);
  Variables<db::wrap_tags_in<Tags::div, flux_tags>> expected_div_fluxes(
      num_grid_points);
  tmpl::for_each<flux_tags>([&x, &f, &fluxes, &expected_div_fluxes](auto tag) {
    using FluxTag = tmpl::type_from<decltype(tag)>;
    get<FluxTag>(fluxes) = FluxTag::flux(f, x);
    using DivFluxTag = Tags::div<FluxTag>;
    get<DivFluxTag>(expected_div_fluxes) = FluxTag::divergence_of_flux(f, x);
  });
  const auto div_fluxes = divergence<flux_tags>(fluxes, mesh, inv_jacobian);
  CHECK(div_fluxes.size() == expected_div_fluxes.size());
  CHECK(Dim * div_fluxes.size() == fluxes.size());
  for (size_t n = 0; n < div_fluxes.size(); ++n) {
    // clang-tidy: pointer arithmetic
    CHECK(div_fluxes.data()[n] ==                                  // NOLINT
          approx(expected_div_fluxes.data()[n]).epsilon(1.e-11));  // NOLINT
  }

  // Test divergence of a single tensor
  const auto div_vector =
      divergence(get<Flux1<Dim, Frame>>(fluxes), mesh, inv_jacobian);
  const auto& expected = get<Tags::div<Flux1<Dim, Frame>>>(div_fluxes);
  Approx local_approx = Approx::custom().epsilon(1.e-11).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(expected, div_vector, local_approx);
}

void test_divergence() {
  using TensorTag = Flux1<1, Frame::Inertial>;
  TestHelpers::db::test_prefix_tag<Tags::div<TensorTag>>("div(Flux1)");

  const size_t n0 = Spectral::maximum_number_of_points<
                        SpatialDiscretization::Basis::Legendre> /
                    2;
  const size_t n1 = Spectral::maximum_number_of_points<
                        SpatialDiscretization::Basis::Legendre> /
                        2 +
                    1;
  const size_t n2 = Spectral::maximum_number_of_points<
                        SpatialDiscretization::Basis::Legendre> /
                        2 -
                    1;
  const Mesh<1> mesh_1d{{{n0}},
                        SpatialDiscretization::Basis::Legendre,
                        SpatialDiscretization::Quadrature::GaussLobatto};
  const Mesh<2> mesh_2d{{{n0, n1}},
                        SpatialDiscretization::Basis::Legendre,
                        SpatialDiscretization::Quadrature::GaussLobatto};
  const Mesh<3> mesh_3d{{{n0, n1, n2}},
                        SpatialDiscretization::Basis::Legendre,
                        SpatialDiscretization::Quadrature::GaussLobatto};
  for (size_t a = 0; a < 5; ++a) {
    std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 1>
        functions_1d{
            {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a)}};
    test_divergence_impl(mesh_1d, std::move(functions_1d));
    for (size_t b = 0; b < 4; ++b) {
      std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 2>
          functions_2d{
              {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a),
               std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(b)}};
      test_divergence_impl(mesh_2d, std::move(functions_2d));
      for (size_t c = 0; c < 3; ++c) {
        std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 3>
            functions_3d{
                {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a),
                 std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(b),
                 std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(c)}};
        test_divergence_impl(mesh_3d, std::move(functions_3d));
      }
    }
  }
}

template <class MapType>
struct MapTag : db::SimpleTag {
  static constexpr size_t dim = MapType::dim;
  using target_frame = typename MapType::target_frame;
  using source_frame = typename MapType::source_frame;

  using type = MapType;
};

template <size_t Dim, typename Frame = Frame::Inertial>
void test_divergence_compute_item_impl(
    const Mesh<Dim>& mesh,
    std::array<std::unique_ptr<MathFunction<1, Frame>>, Dim> functions) {
  const auto coordinate_map = make_affine_map<Dim>();
  using map_tag = MapTag<std::decay_t<decltype(coordinate_map)>>;
  using mesh_tag = domain::Tags::Mesh<Dim>;
  using inv_jac_tag = domain::Tags::InverseJacobianCompute<
      map_tag, domain::Tags::LogicalCoordinates<Dim>>;
  using flux_tags = two_fluxes<Dim, Frame>;
  using flux_tag = Tags::Variables<flux_tags>;
  using div_tags = db::wrap_tags_in<Tags::div, flux_tags>;
  TestHelpers::db::test_compute_tag<
      Tags::DivVariablesCompute<flux_tag, mesh_tag, inv_jac_tag>>(
      "Variables(div(Flux1),div(Flux2))");
  TestHelpers::db::test_compute_tag<
      Tags::DivVectorCompute<Flux1<Dim, Frame>, mesh_tag, inv_jac_tag>>(
      "div(Flux1)");

  const size_t num_grid_points = mesh.number_of_grid_points();
  const auto xi = logical_coordinates(mesh);
  const auto x = coordinate_map(xi);
  const auto inv_jacobian = coordinate_map.inv_jacobian(xi);
  MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));
  Variables<flux_tags> fluxes(num_grid_points);
  Variables<div_tags> expected_div_fluxes(num_grid_points);

  tmpl::for_each<flux_tags>([&x, &f, &fluxes, &expected_div_fluxes](auto tag) {
    using FluxTag = tmpl::type_from<decltype(tag)>;
    get<FluxTag>(fluxes) = FluxTag::flux(f, x);
    using DivFluxTag = Tags::div<FluxTag>;
    get<DivFluxTag>(expected_div_fluxes) = FluxTag::divergence_of_flux(f, x);
  });

  auto box = db::create<
      db::AddSimpleTags<mesh_tag, flux_tag, map_tag>,
      db::AddComputeTags<
          domain::Tags::LogicalCoordinates<Dim>, inv_jac_tag,
          Tags::DivVariablesCompute<flux_tag, mesh_tag, inv_jac_tag>,
          Tags::DivVectorCompute<Flux1<Dim, Frame>, mesh_tag, inv_jac_tag>>>(
      mesh, fluxes, coordinate_map);

  const auto& div_fluxes = db::get<Tags::Variables<div_tags>>(box);

  CHECK(div_fluxes.size() == expected_div_fluxes.size());
  for (size_t n = 0; n < div_fluxes.size(); ++n) {
    // clang-tidy: pointer arithmetic
    CHECK(div_fluxes.data()[n] ==                                  // NOLINT
          approx(expected_div_fluxes.data()[n]).epsilon(1.e-11));  // NOLINT
  }

  const auto& div_flux1 =
      db::get<Tags::DivVectorCompute<Flux1<Dim, Frame>, mesh_tag, inv_jac_tag>>(
          box);
  const auto& expected = get<Tags::div<Flux1<Dim, Frame>>>(div_fluxes);
  Approx local_approx = Approx::custom().epsilon(1.e-11).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(expected, div_flux1, local_approx);
}

void test_divergence_compute() {
  const size_t n0 = Spectral::maximum_number_of_points<
                        SpatialDiscretization::Basis::Legendre> /
                    2;
  const size_t n1 = Spectral::maximum_number_of_points<
                        SpatialDiscretization::Basis::Legendre> /
                        2 +
                    1;
  const size_t n2 = Spectral::maximum_number_of_points<
                        SpatialDiscretization::Basis::Legendre> /
                        2 -
                    1;
  const Mesh<1> mesh_1d{{{n0}},
                        SpatialDiscretization::Basis::Legendre,
                        SpatialDiscretization::Quadrature::GaussLobatto};
  const Mesh<2> mesh_2d{{{n0, n1}},
                        SpatialDiscretization::Basis::Legendre,
                        SpatialDiscretization::Quadrature::GaussLobatto};
  const Mesh<3> mesh_3d{{{n0, n1, n2}},
                        SpatialDiscretization::Basis::Legendre,
                        SpatialDiscretization::Quadrature::GaussLobatto};
  for (size_t a = 0; a < 5; ++a) {
    std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 1>
        functions_1d{
            {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a)}};
    test_divergence_compute_item_impl(mesh_1d, std::move(functions_1d));
    for (size_t b = 0; b < 4; ++b) {
      std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 2>
          functions_2d{
              {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a),
               std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(b)}};
      test_divergence_compute_item_impl(mesh_2d, std::move(functions_2d));
      for (size_t c = 0; c < 3; ++c) {
        std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 3>
            functions_3d{
                {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a),
                 std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(b),
                 std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(c)}};
        test_divergence_compute_item_impl(mesh_3d, std::move(functions_3d));
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Divergence",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_divergence();
  test_divergence_compute();
}
