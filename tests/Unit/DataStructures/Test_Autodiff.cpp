// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/utils/gradient.hpp>

#include "DataStructures/DynamicVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {

template <typename DataType>
DataType f1(const DataType& x, const double param) {
  return square(x) * param;
}

template <typename DataType>
DataType f2(const std::array<DataType, 2>& x) {
  return x[0] * square(x[1]);
}
template <typename DataType>
std::array<DataType, 2> f3(const std::array<DataType, 2>& x) {
  return {{square(x[0]) * x[1], cube(x[1])}};
}

template <typename DataType>
DataType f4(const tnsr::I<DataType, 2>& x, const Scalar<DataType>& y) {
  const auto f = tenex::evaluate<ti::I>(x(ti::I) * y());
  return get<0>(f);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Autodiff", "[Unit][DataStructures]") {
  // Test that autodiff works with the autodiff::real type in forward mode.
  // The autodiff::dual type supports higher cross-derivatives in forward mode.
  // Reverse mode with the autodiff::var type needs some adaptors to work with
  // Blaze vectors and tensors, similar to how it is implemented for Eigen in
  // the autodiff library (it works fine with single numbers though, if needed).
  {
    INFO("Single numbers");
    autodiff::real x = 2.0;
    const auto df_dx = autodiff::derivative(
        &f1<autodiff::real>, autodiff::wrt(x), autodiff::at(x, 3.0));
    CHECK(df_dx == approx(12.0));
  }
  {
    INFO("Vectorization");
    blaze::DynamicVector<autodiff::real> x{2.0, 3.0, 4.0};
    const auto df_dx =
        autodiff::derivative(&f1<blaze::DynamicVector<autodiff::real>>,
                             autodiff::wrt(x), autodiff::at(x, 3.0));
    const blaze::DynamicVector<double> df_dx_expected{12.0, 18.0, 24.0};
    CHECK_ITERABLE_APPROX(df_dx, df_dx_expected);
  }
  {
    INFO("Gradient");
    std::array<autodiff::real, 2> x = {2.0, 3.0};
    const auto df_dx = autodiff::derivative(
        &f2<autodiff::real>, autodiff::wrt(x[0]), autodiff::at(x));
    const auto df_dy = autodiff::derivative(
        &f2<autodiff::real>, autodiff::wrt(x[1]), autodiff::at(x));
    CHECK(df_dx == approx(9.0));
    CHECK(df_dy == approx(12.0));
    // Same as above, but using the gradient convenience function
    autodiff::real F{};
    std::vector<double> grad{};
    autodiff::gradient(&f2<autodiff::real>, autodiff::wrt(x[0], x[1]),
                       autodiff::at(x), F, grad);
    CHECK(grad[0] == approx(9.0));
    CHECK(grad[1] == approx(12.0));
  }
  {
    INFO("Jacobian");
    std::array<autodiff::real, 2> x = {2.0, 3.0};
    std::array<autodiff::real, 2> F{};
    blaze::DynamicMatrix<double> J{};
    autodiff::jacobian(&f3<autodiff::real>, autodiff::wrt(x[0], x[1]),
                       autodiff::at(x), F, J);
    const blaze::DynamicMatrix<double> J_expected{{12.0, 4.0}, {0.0, 27.0}};
    CHECK_ITERABLE_APPROX(J, J_expected);
  }
  {
    INFO("Tensors");
    tnsr::I<blaze::DynamicVector<autodiff::real>, 2> x{};
    get<0>(x) = blaze::DynamicVector<autodiff::real>{2.0, 3.0};
    get<1>(x) = blaze::DynamicVector<autodiff::real>{4.0, 5.0};
    Scalar<blaze::DynamicVector<autodiff::real>> y{};
    get(y) = blaze::DynamicVector<autodiff::real>{6.0, 7.0};
    const auto df_dx1 =
        autodiff::derivative(&f4<blaze::DynamicVector<autodiff::real>>,
                             autodiff::wrt(get<0>(x)), autodiff::at(x, y));
    const blaze::DynamicVector<double> df_dx1_expected{6.0, 7.0};
    CHECK_ITERABLE_APPROX(df_dx1, df_dx1_expected);
  }
}
