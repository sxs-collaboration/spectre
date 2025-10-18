// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Utilities/Autodiff/Autodiff.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
using Dual = autodiff::dual;
using Var = autodiff::var;

template <typename DataType>
DataType f1(const DataType& x, const double param) {
  return square(x) * param;
}

template <typename DataType>
std::array<DataType, 2> f2(const std::array<DataType, 2>& x) {
  return {{square(x[0]) * x[1], cube(x[1])}};
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Autodiff", "[Unit][DataStructures]") {
  {
    INFO("Single number forward mode");
    Dual x = 2.0;
    const double param = 3.0;
    const auto df_dx = autodiff::derivative(&f1<Dual>, wrt(x), at(x, param));
    CHECK(df_dx == approx(12.0));
  }
  {
    INFO("Single number reverse mode");
    const Var x = 2.0;
    const double param = 3.0;
    const Var u = f1<Var>(x, param);
    const auto [df_dx] =
        autodiff::derivatives(u, wrt(x));
    CHECK(df_dx == approx(12.0));
  }
  {
    INFO("SIMD number forward mode");
    using BatchType = simd::batch<double>;
    using BatchDual = autodiff::HigherOrderDual<2, BatchType>;
    BatchDual x = BatchType(2.0);
    const double param = 3.0;
    autodiff::seed<1>(x, 1.0);
    autodiff::seed<2>(x, 1.0);
    const auto fx = f1(x, param);
    const auto df_dx = autodiff::derivative<1>(fx);
    std::array<double, BatchType::size> lanes{};
    df_dx.store_unaligned(lanes.data());
    for (const double lane : lanes) {
      CHECK(lane == approx(12.0));
    }
    const auto df_dxx = autodiff::derivative<2>(fx);
    df_dxx.store_unaligned(lanes.data());
    for (const double lane : lanes) {
      CHECK(lane == approx(6.0));
    }
  }
  {
    INFO("SIMD number reverse mode");
    using BatchType = simd::batch<double>;
    using BatchVar = autodiff::Variable<BatchType>;
    BatchVar x = BatchType(2.0);
    const double param = 3.0;
    const auto u = f1(x, param);
    const auto [u_x] = derivativesx(u, wrt(x));
    const auto [u_xx] = derivativesx(u_x, wrt(x));
    std::array<double, BatchType::size> lanes{};
    autodiff::val(u_x).store_unaligned(lanes.data());
    for (const double lane : lanes) {
      CHECK(lane == approx(12.0));
    }
    autodiff::val(u_xx).store_unaligned(lanes.data());
    for (const double lane : lanes) {
      CHECK(lane == approx(6.0));
    }
  }
  {
    INFO("Jacobian forward mode");
    std::array<Dual, 2> x = {2.0, 3.0};
    std::array<std::array<double, 2>, 2> expected_df_dx{
        {{12.0, 4.0}, {0.0, 27.0}}};

    for (size_t j = 0; j < 2; ++j) {
      for (size_t i = 0; i < 2; ++i) {
        autodiff::seed<1>(x.at(i), 1.0);
        autodiff::seed<1>(x.at((i + 1) % 2), 0.0);
        const std::array<Dual, 2> fx = f2(x);
        const auto dfj_dxi = autodiff::derivative(gsl::at(fx, j));
        CHECK(dfj_dxi == approx(expected_df_dx.at(j).at(i)));
      }
    }
  }
  {
    INFO("Jacobian reverse mode");
    std::array<Var, 2> x = {2.0, 3.0};
    std::array<std::array<double, 2>, 2> expected_df_dx{
        {{12.0, 4.0}, {0.0, 27.0}}};
    std::array<Var, 2> fx = f2(x);

    for (size_t j = 0; j < 2; ++j) {
      const auto dfj_dxi = autodiff::derivatives(
          gsl::at(fx, j), wrt(x[0], x[1]));
      for (size_t i = 0; i < 2; ++i) {
        CHECK(dfj_dxi.at(i) == approx(expected_df_dx.at(j).at(i)));
      }
    }
  }
}
