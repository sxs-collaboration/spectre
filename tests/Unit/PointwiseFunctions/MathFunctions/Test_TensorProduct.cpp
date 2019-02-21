// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

class DataVector;
template <size_t VolumeDim>
class MathFunction;

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t VolumeDim>
auto make_affine_map() noexcept;

template <>
auto make_affine_map<1>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine{-1.0, 1.0, -0.3, 0.7});
}

template <>
auto make_affine_map<2>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
}

template <>
auto make_affine_map<3>() noexcept {
  return domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}

template <typename T, size_t VolumeDim>
Scalar<T> expected_value(const tnsr::I<T, VolumeDim>& x,
                         const std::array<size_t, VolumeDim>& powers,
                         const double scale) noexcept {
  auto result = make_with_value<Scalar<T>>(x, scale);
  for (size_t d = 0; d < VolumeDim; ++d) {
    result.get() *= pow(x.get(d), gsl::at(powers, d));
  }
  return result;
}

template <typename T, size_t VolumeDim>
tnsr::i<T, VolumeDim> expected_first_derivs(
    const tnsr::I<T, VolumeDim>& x, const std::array<size_t, VolumeDim>& powers,
    const double scale) noexcept {
  auto result = make_with_value<tnsr::i<T, VolumeDim>>(x, scale);
  for (size_t d = 0; d < VolumeDim; ++d) {
    const size_t p = gsl::at(powers, d);
    for (size_t i = 0; i < VolumeDim; ++i) {
      if (d == i) {
        if (0 == p) {
          result.get(i) = 0.0;
        } else {
          result.get(i) *= p * pow(x.get(d), p - 1);
        }
      } else {
        result.get(i) *= pow(x.get(d), p);
      }
    }
  }
  return result;
}

template <typename T, size_t VolumeDim>
tnsr::ii<T, VolumeDim> expected_second_derivs(
    const tnsr::I<T, VolumeDim>& x, const std::array<size_t, VolumeDim>& powers,
    const double scale) noexcept {
  auto result = make_with_value<tnsr::ii<T, VolumeDim>>(x, scale);
  for (size_t d = 0; d < VolumeDim; ++d) {
    const size_t p = gsl::at(powers, d);
    for (size_t i = 0; i < VolumeDim; ++i) {
      if (d == i) {
        if (2 > p) {
          result.get(i, i) = 0.0;
        } else {
          result.get(i, i) *= p * (p - 1) * pow(x.get(d), p - 2);
        }
      } else {
        result.get(i, i) *= pow(x.get(d), p);
      }
      for (size_t j = i + 1; j < VolumeDim; ++j) {
        if (d == j or d == i) {
          if (0 == p) {
            result.get(i, j) = 0.0;
          } else {
            result.get(i, j) *= p * pow(x.get(d), p - 1);
          }
        } else {
          result.get(i, j) *= pow(x.get(d), p);
        }
      }
    }
  }
  return result;
}

template <size_t VolumeDim>
void test_tensor_product(
    const Mesh<VolumeDim>& mesh, const double scale,
    std::array<std::unique_ptr<MathFunction<1>>, VolumeDim>&& functions,
    const std::array<size_t, VolumeDim>& powers) noexcept {
  const auto coordinate_map = make_affine_map<VolumeDim>();
  const auto x = coordinate_map(logical_coordinates(mesh));
  MathFunctions::TensorProduct<VolumeDim> f(scale, std::move(functions));
  CHECK_ITERABLE_APPROX(f(x), expected_value(x, powers, scale));
  CHECK_ITERABLE_APPROX(f.first_derivatives(x),
                        expected_first_derivs(x, powers, scale));
  CHECK_ITERABLE_APPROX(f.second_derivatives(x),
                        expected_second_derivs(x, powers, scale));
  tnsr::I<double, VolumeDim> point{2.2};
  for (size_t d = 0; d < VolumeDim; ++d) {
    point.get(d) += d * 1.7;
  }
  CHECK_ITERABLE_APPROX(f(point), expected_value(point, powers, scale));
  CHECK_ITERABLE_APPROX(f.first_derivatives(point),
                        expected_first_derivs(point, powers, scale));
  CHECK_ITERABLE_APPROX(f.second_derivatives(point),
                        expected_second_derivs(point, powers, scale));
}

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Var1"; }
};

template <size_t VolumeDim>
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, VolumeDim, Frame::Inertial>;
  static std::string name() noexcept { return "Var2"; }
};

template <size_t VolumeDim>
using TwoVars = tmpl::list<Var1, Var2<VolumeDim>>;

template <size_t VolumeDim>
void test_with_numerical_derivatives(
    const Mesh<VolumeDim>& mesh, const double scale,
    std::array<std::unique_ptr<MathFunction<1>>, VolumeDim>&&
        functions) noexcept {
  const auto coordinate_map = make_affine_map<VolumeDim>();
  Variables<TwoVars<VolumeDim>> u(mesh.number_of_grid_points());
  const auto xi = logical_coordinates(mesh);
  const auto x = coordinate_map(xi);
  const auto inv_jacobian = coordinate_map.inv_jacobian(xi);
  MathFunctions::TensorProduct<VolumeDim> f(scale, std::move(functions));
  get<Var1>(u) = f(x);
  get<Var2<VolumeDim>>(u) = f.first_derivatives(x);
  const auto du =
      partial_derivatives<TwoVars<VolumeDim>>(u, mesh, inv_jacobian);
  const auto& dVar1 =
      get<Tags::deriv<Var1, tmpl::size_t<VolumeDim>, Frame::Inertial>>(du);
  Approx custom_approx = Approx::custom().epsilon(1.e-6);
  CHECK_ITERABLE_CUSTOM_APPROX(f.first_derivatives(x), dVar1, custom_approx);
  const auto& dVar2 = get<
      Tags::deriv<Var2<VolumeDim>, tmpl::size_t<VolumeDim>, Frame::Inertial>>(
      du);
  const auto d2f = f.second_derivatives(x);
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = 0; j < VolumeDim; ++j) {
      const auto& d2f_ij = d2f.get(i, j);
      const auto& dVar2_ij = dVar2.get(i, j);
      CHECK_ITERABLE_CUSTOM_APPROX(d2f_ij, dVar2_ij, custom_approx);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.MathFunctions.TensorProduct",
                  "[PointwiseFunctions][Unit]") {
  for (size_t a = 0; a < 5; ++a) {
    std::array<std::unique_ptr<MathFunction<1>>, 1> functions{
        {std::make_unique<MathFunctions::PowX>(a)}};
    test_tensor_product(Mesh<1>{4, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto},
                        1.5, std::move(functions), {{a}});
    for (size_t b = 0; b < 4; ++b) {
      std::array<std::unique_ptr<MathFunction<1>>, 2> functions_2d{
          {std::make_unique<MathFunctions::PowX>(a),
           std::make_unique<MathFunctions::PowX>(b)}};
      test_tensor_product(Mesh<2>{{{4, 3}},
                                  Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto},
                          2.5, std::move(functions_2d), {{a, b}});
      for (size_t c = 0; c < 3; ++c) {
        std::array<std::unique_ptr<MathFunction<1>>, 3> functions_3d{
            {std::make_unique<MathFunctions::PowX>(a),
             std::make_unique<MathFunctions::PowX>(b),
             std::make_unique<MathFunctions::PowX>(c)}};
        test_tensor_product(Mesh<3>{{{4, 3, 5}},
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto},
                            3.5, std::move(functions_3d), {{a, b, c}});
      }
    }
  }

  std::array<std::unique_ptr<MathFunction<1>>, 1> sinusoid{
      {std::make_unique<MathFunctions::Sinusoid>(1.0, 1.0, -1.0)}};

  test_with_numerical_derivatives(
      Mesh<1>{8, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
      1.5, std::move(sinusoid));

  std::array<std::unique_ptr<MathFunction<1>>, 3> generic_3d{
      {std::make_unique<MathFunctions::Sinusoid>(1.0, 1.0, -1.0),
       std::make_unique<MathFunctions::Gaussian>(1.0, 1.0, 0.4),
       std::make_unique<MathFunctions::PowX>(2)}};

  test_with_numerical_derivatives(
      Mesh<3>{8, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
      1.5, std::move(generic_3d));
}
