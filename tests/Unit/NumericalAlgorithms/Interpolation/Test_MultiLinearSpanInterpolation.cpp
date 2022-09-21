// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/MultiLinearSpanInterpolation.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim, size_t NumVar>
void test() {
  CAPTURE(Dim);
  CAPTURE(NumVar);

  MAKE_GENERATOR(gen);

  std::uniform_real_distribution<> dist_func(-1., 1.);
  std::uniform_int_distribution<size_t> dist_int(3, 5);

  // Main idea: The table performs a multi-linear interpolation
  // Each element in the table will be multi-linear function,
  // so we can use this directly as a test.
  // For simplicity, we will use the table to globally reproduce a
  // single multi-linear function.

  std::array<std::array<double, Dim>, NumVar> matrix;
  std::array<double, NumVar> vector_b;

  for (size_t i = 0; i < NumVar; ++i) {
    vector_b[i] = dist_func(gen);
    for (size_t j = 0; j < Dim; ++j) {
      matrix[i][j] = dist_func(gen);
    }
  }

  auto f = [&matrix, &vector_b](const std::array<double, Dim>& x) {
    std::array<double, NumVar> res;

    for (size_t i = 0; i < NumVar; ++i) {
      res[i] = vector_b[i];
      for (size_t j = 0; j < Dim; ++j) {
        res[i] += matrix[i][j] * x[j];
      }
    }
    return res;
  };

  // Create data data structures
  std::array<std::unique_ptr<double[]>, Dim> X_data;
  Index<Dim> num_x_points;

  // Initialize discrete grid for the tabulated interpolation routine

  size_t total_num_points = 1;
  for (size_t n = 0; n < Dim; ++n) {
    num_x_points[n] = dist_int(gen);
    total_num_points *= num_x_points[n];

    X_data[n] = std::move(std::make_unique<double[]>(num_x_points[n]));

    for (size_t m = 0; m < num_x_points[n]; m++) {
      X_data[n][m] = -1. + m * 2. / static_cast<double>(num_x_points[n] - 1);
    }
  }

  // Correct for NumVars
  total_num_points *= NumVar;

  // Allocate storage for main table
  auto dependent_variables = std::make_unique<double[]>(total_num_points);

  std::array<gsl::span<const double>, Dim> independent_data_view;

  // Fill dependent_variables

  for (size_t ijk = 0; ijk < total_num_points / NumVar; ++ijk) {
    Index<Dim> index;
    auto tmp = std::array<double, Dim>{};

    // Uncompress index
    size_t myind= ijk;
    for (size_t nn = 0; nn < Dim - 1; ++nn) {
      index[nn] = myind % num_x_points[nn];
      myind = (myind - index[nn])/num_x_points[nn];
      tmp[nn] = X_data[nn][index[nn]];
    }
    index[Dim - 1] = myind;
    tmp[Dim - 1] = X_data[Dim - 1][index[Dim - 1]];

    for (size_t nv = 0; nv < NumVar; ++nv) {
      std::array<double, NumVar> Fx = f(tmp);
      dependent_variables[nv + NumVar * ijk] = Fx[nv];
    }
  }

  std::array<double, Dim> x;
  std::array<size_t, NumVar> which_var;

  for (size_t which_Dim = 0; which_Dim < Dim; ++which_Dim) {
    independent_data_view[which_Dim] = gsl::span<const double>{
        X_data[which_Dim].get(), num_x_points[which_Dim]};
    x[which_Dim] = dist_func(gen);
  }

  for (size_t nv = 0; nv < NumVar; ++nv) {
    which_var[nv] = nv;
  }

  intrp::UniformMultiLinearSpanInterpolation<Dim, NumVar> uniform_intp(
      independent_data_view, {dependent_variables.get(), total_num_points},
      num_x_points);
  intrp::GeneralMultiLinearSpanInterpolation<Dim, NumVar> general_intp(
      independent_data_view, {dependent_variables.get(), total_num_points},
      num_x_points);

  for (size_t d = 0; d < Dim; ++d) {
    uniform_intp.extrapolate_above_data(d, true);
    general_intp.extrapolate_above_data(d, true);

    uniform_intp.extrapolate_below_data(d, true);
    general_intp.extrapolate_below_data(d, true);
  }

  auto y_expected = f(x);


  auto y_interpolated = uniform_intp.interpolate(which_var, x);
  auto y_interpolated_gen = general_intp.interpolate(which_var, x);

  for (size_t nv = 0; nv < NumVar; ++nv) {
    CHECK(std::abs(y_expected[nv] - y_interpolated[nv]) <
          1.e-12 * std::abs(y_expected[nv]));
    CHECK(std::abs(y_expected[nv] - y_interpolated_gen[nv]) <
          1.e-12 * std::abs(y_expected[nv]));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.MultiLinearSpanInterpolation",
                  "[Unit][NumericalAlgorithms]") {
    test<1,1>();
    test<2,3>();
    test<3,5>();
}
