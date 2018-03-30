// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestingFramework.hpp"

template <size_t VolumeDim>
class MathFunction;
template <typename X, typename Symm, typename IndexList>
class Tensor;

namespace {
using Affine = CoordinateMaps::Affine;
using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D =
    CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t VolumeDim>
auto make_affine_map() noexcept;

template <>
auto make_affine_map<1>() noexcept {
  return make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine{-1.0, 1.0, -0.3, 0.7});
}

template <>
auto make_affine_map<2>() noexcept {
  return make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine2D{
      Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
}

template <>
auto make_affine_map<3>() noexcept {
  return make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
      Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
      Affine{-1.0, 1.0, 2.3, 2.8}});
}

template <size_t Dim, typename Frame>
struct Flux1 : db::DataBoxTag {
  using type = tnsr::I<DataVector, Dim, Frame>;
  static constexpr db::DataBoxString label = "Flux1";
  static auto flux(const MathFunctions::TensorProduct<Dim>& f,
                   const tnsr::I<DataVector, Dim, Frame>& x) noexcept {
    auto result = make_with_value<tnsr::I<DataVector, Dim, Frame>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      result.get(d) = (d + 0.5) * get(f_of_x);
    }
    return result;
  }
  static auto divergence_of_flux(
      const MathFunctions::TensorProduct<Dim>& f,
      const tnsr::I<DataVector, Dim, Frame>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.);
    const auto df = f.first_derivatives(x);
    for (size_t d = 0; d < Dim; ++d) {
      get(result) += (d + 0.5) * df.get(d);
    }
    return result;
  }
};

template <size_t Dim, typename Frame>
struct Flux2 : db::DataBoxTag {
  using type = tnsr::Ij<DataVector, Dim, Frame>;
  static constexpr db::DataBoxString label = "Flux2";
  static auto flux(const MathFunctions::TensorProduct<Dim>& f,
                   const tnsr::I<DataVector, Dim, Frame>& x) noexcept {
    auto result = make_with_value<tnsr::Ij<DataVector, Dim, Frame>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t j = 0; j < Dim; ++j) {
        result.get(d, j) = (d + 0.5) * (j + 0.25) * get(f_of_x);
      }
    }
    return result;
  }
  static auto divergence_of_flux(
      const MathFunctions::TensorProduct<Dim>& f,
      const tnsr::I<DataVector, Dim, Frame>& x) noexcept {
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
void test_divergence(
    const Index<Dim>& extents,
    std::array<std::unique_ptr<MathFunction<1>>, Dim> functions) noexcept {
  const auto coordinate_map = make_affine_map<Dim>();
  const size_t number_of_grid_points = extents.product();
  const auto xi = logical_coordinates(extents);
  const auto x = coordinate_map(xi);
  const auto inv_jacobian = coordinate_map.inv_jacobian(xi);
  MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));
  using flux_tags = two_fluxes<Dim, Frame>;
  Variables<flux_tags> fluxes(number_of_grid_points);
  Variables<db::wrap_tags_in<Tags::div, flux_tags, Frame>> expected_div_fluxes(
      number_of_grid_points);
  tmpl::for_each<flux_tags>([&x, &f, &fluxes,
                             &expected_div_fluxes ](auto tag) noexcept {
    using FluxTag = tmpl::type_from<decltype(tag)>;
    get<FluxTag>(fluxes) = FluxTag::flux(f, x);
    using DivFluxTag = Tags::div<FluxTag, Frame>;
    get<DivFluxTag>(expected_div_fluxes) = FluxTag::divergence_of_flux(f, x);
  });
  const auto div_fluxes = divergence<flux_tags>(fluxes, extents, inv_jacobian);
  CHECK(div_fluxes.size() == expected_div_fluxes.size());
  CHECK(Dim * div_fluxes.size() == fluxes.size());
  for (size_t n = 0; n < div_fluxes.size(); ++n) {
    // clang-tidy: pointer arithmetic
    CAPTURE_PRECISE(div_fluxes.data()[n] -                         // NOLINT
                    expected_div_fluxes.data()[n]);                // NOLINT
    CHECK(div_fluxes.data()[n] ==                                  // NOLINT
          approx(expected_div_fluxes.data()[n]).epsilon(1.e-11));  // NOLINT
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Divergence",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  const size_t n0 = Basis::lgl::maximum_number_of_pts / 2;
  const size_t n1 = Basis::lgl::maximum_number_of_pts / 2 + 1;
  const size_t n2 = Basis::lgl::maximum_number_of_pts / 2 - 1;
  const Index<1> extents_1d(n0);
  const Index<2> extents_2d(n0, n1);
  const Index<3> extents_3d(n0, n1, n2);
  for (size_t a = 0; a < 5; ++a) {
    std::array<std::unique_ptr<MathFunction<1>>, 1> functions_1d{
        {std::make_unique<MathFunctions::PowX>(a)}};
    test_divergence(extents_1d, std::move(functions_1d));
    for (size_t b = 0; b < 4; ++b) {
      std::array<std::unique_ptr<MathFunction<1>>, 2> functions_2d{
          {std::make_unique<MathFunctions::PowX>(a),
           std::make_unique<MathFunctions::PowX>(b)}};
      test_divergence(extents_2d, std::move(functions_2d));
      for (size_t c = 0; c < 3; ++c) {
        std::array<std::unique_ptr<MathFunction<1>>, 3> functions_3d{
            {std::make_unique<MathFunctions::PowX>(a),
             std::make_unique<MathFunctions::PowX>(b),
             std::make_unique<MathFunctions::PowX>(c)}};
        test_divergence(extents_3d, std::move(functions_3d));
      }
    }
  }
}

namespace {
template <class MapType>
struct MapTag : db::DataBoxTag {
  using type = MapType;
  static constexpr db::DataBoxString label = "MapTag";
};

template <size_t Dim, typename Frame = Frame::Inertial>
void test_divergence_compute_item(
    const Index<Dim>& extents,
    std::array<std::unique_ptr<MathFunction<1>>, Dim> functions) noexcept {
  const auto coordinate_map = make_affine_map<Dim>();
  using map_tag = MapTag<std::decay_t<decltype(coordinate_map)>>;
  using inv_jac_tag =
      Tags::InverseJacobian<map_tag, Tags::LogicalCoordinates<Dim>>;
  using flux_tags = two_fluxes<Dim, Frame>;
  using div_tag = Tags::div<flux_tags, inv_jac_tag>;

  const size_t number_of_grid_points = extents.product();
  const auto xi = logical_coordinates(extents);
  const auto x = coordinate_map(xi);
  const auto inv_jacobian = coordinate_map.inv_jacobian(xi);
  MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));
  Variables<flux_tags> fluxes(number_of_grid_points);
  Variables<db::wrap_tags_in<Tags::div, flux_tags, Frame>> expected_div_fluxes(
      number_of_grid_points);


  tmpl::for_each<flux_tags>([&x, &f, &fluxes,
                             &expected_div_fluxes ](auto tag) noexcept {
    using FluxTag = tmpl::type_from<decltype(tag)>;
    get<FluxTag>(fluxes) = FluxTag::flux(f, x);
    using DivFluxTag = Tags::div<FluxTag, Frame>;
    get<DivFluxTag>(expected_div_fluxes) = FluxTag::divergence_of_flux(f, x);
  });

  auto box = db::create<
      db::AddSimpleTags<Tags::Extents<Dim>, Tags::Variables<flux_tags>,
                        map_tag>,
      db::AddComputeTags<Tags::LogicalCoordinates<Dim>, inv_jac_tag, div_tag>>(
      extents, fluxes, coordinate_map);

  const auto& div_fluxes = db::get<div_tag>(box);

  CHECK(div_fluxes.size() == expected_div_fluxes.size());
  for (size_t n = 0; n < div_fluxes.size(); ++n) {
    // clang-tidy: pointer arithmetic
    CAPTURE_PRECISE(div_fluxes.data()[n] -                         // NOLINT
                    expected_div_fluxes.data()[n]);                // NOLINT
    CHECK(div_fluxes.data()[n] ==                                  // NOLINT
          approx(expected_div_fluxes.data()[n]).epsilon(1.e-11));  // NOLINT
  }
}
} // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Divergence.ComputeItem",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  const size_t n0 = Basis::lgl::maximum_number_of_pts / 2;
  const size_t n1 = Basis::lgl::maximum_number_of_pts / 2 + 1;
  const size_t n2 = Basis::lgl::maximum_number_of_pts / 2 - 1;
  const Index<1> extents_1d(n0);
  const Index<2> extents_2d(n0, n1);
  const Index<3> extents_3d(n0, n1, n2);
  for (size_t a = 0; a < 5; ++a) {
    std::array<std::unique_ptr<MathFunction<1>>, 1> functions_1d{
        {std::make_unique<MathFunctions::PowX>(a)}};
    test_divergence_compute_item(extents_1d, std::move(functions_1d));
    for (size_t b = 0; b < 4; ++b) {
      std::array<std::unique_ptr<MathFunction<1>>, 2> functions_2d{
          {std::make_unique<MathFunctions::PowX>(a),
           std::make_unique<MathFunctions::PowX>(b)}};
      test_divergence_compute_item(extents_2d, std::move(functions_2d));
      for (size_t c = 0; c < 3; ++c) {
        std::array<std::unique_ptr<MathFunction<1>>, 3> functions_3d{
            {std::make_unique<MathFunctions::PowX>(a),
             std::make_unique<MathFunctions::PowX>(b),
             std::make_unique<MathFunctions::PowX>(c)}};
        test_divergence_compute_item(extents_3d, std::move(functions_3d));
      }
    }
  }
}
