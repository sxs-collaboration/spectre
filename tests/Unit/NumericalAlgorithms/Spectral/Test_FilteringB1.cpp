// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/ExponentialFilter.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/FilteringB1.hpp"
#include "NumericalAlgorithms/Spectral/FilteringB1.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using TestTags = tmpl::list<Tags::TempScalar<0>, Tags::TempI<1, 3>>;

// Compute the expected filtered Variables by iterating each tensor component
// individually. For each component, count the number of x-direction indices to
// determine parity, then apply_matrices with the corresponding filter directly
// on that component's DataVector. This is independent of the buffer-copying
// logic in the implementation.
template <typename VariableTags>
Variables<VariableTags> compute_expected(const Variables<VariableTags>& vars,
                                         const Mesh<3>& mesh,
                                         const double alpha,
                                         const unsigned half_power) {
  const Matrix empty{};
  std::array<Matrix, 3> filter_even = make_array<3>(empty);
  std::array<Matrix, 3> filter_odd = make_array<3>(empty);

  for (size_t d = 0; d < 3; d++) {
    gsl::at(filter_even, d) = Spectral::filtering::exponential_filter(
        mesh.slice_through(d), alpha, half_power, Spectral::Parity::Even);
    gsl::at(filter_odd, d) = Spectral::filtering::exponential_filter(
        mesh.slice_through(d), alpha, half_power, Spectral::Parity::Odd);
  }

  Variables<VariableTags> expected{mesh.number_of_grid_points()};

  tmpl::for_each<VariableTags>(
      [&]<typename Tag>(const tmpl::type_<Tag> /*meta*/) {
        using TensorType = typename Tag::type;
        constexpr auto index_types = TensorType::index_types();
        const auto& in_tensor = get<Tag>(vars);
        auto& out_tensor = get<Tag>(expected);

        for (size_t comp = 0; comp < TensorType::size(); ++comp) {
          const auto tensor_index = TensorType::get_tensor_index(comp);
          size_t x_count = 0;
          for (size_t i = 0; i < index_types.size(); ++i) {
            const bool is_x_spatial =
                gsl::at(index_types, i) == IndexType::Spatial and
                gsl::at(tensor_index, i) == 0;
            const bool is_x_spacetime =
                gsl::at(index_types, i) == IndexType::Spacetime and
                gsl::at(tensor_index, i) == 1;
            if (is_x_spatial or is_x_spacetime) {
              ++x_count;
            }
          }
          const auto& filter = (x_count % 2 == 0) ? filter_even : filter_odd;
          out_tensor[comp] =
              apply_matrices(filter, in_tensor[comp], mesh.extents());
        }
      });

  return expected;
}

template <typename VariableTags>
void test_b1_filter(const Mesh<3>& mesh, const double alpha,
                    const unsigned half_power) {
  CAPTURE(mesh);
  CAPTURE(alpha);
  CAPTURE(half_power);

  const size_t num_pts = mesh.number_of_grid_points();

  Variables<VariableTags> vars{num_pts};
  double val = 1.0;
  for (size_t i = 0; i < vars.size(); ++i) {
    vars.data()[i] = val;
    val += 1.3;
  }

  const Variables<VariableTags> expected =
      compute_expected(vars, mesh, alpha, half_power);

  Spectral::filtering::zernike_b1_exponential_filter(make_not_null(&vars), mesh,
                                                     alpha, half_power);

  CHECK_VARIABLES_APPROX(vars, expected);
}

// For a fixed alpha, the filter must damp the highest-order modal coefficient
// by exactly exp(-alpha). We construct nodal data equal to the highest-order
// ZernikeB1 basis function for each parity, apply the filter, and check the
// result is exp(-alpha) times the input. This verifies the magnitude of
// damping, not just self-consistency between the implementation and
// compute_expected.
void test_highest_mode_damping(const size_t n_r, const double alpha,
                               const unsigned half_power) {
  CAPTURE(n_r);
  CAPTURE(alpha);
  CAPTURE(half_power);

  // Spherical mesh: only radial dimension is non-trivial
  const Mesh<3> mesh{{{n_r, 1, 1}},
                     {{Spectral::Basis::ZernikeB1, Spectral::Basis::Cartoon,
                       Spectral::Basis::Cartoon}},
                     {{Spectral::Quadrature::GaussRadauUpper,
                       Spectral::Quadrature::SphericalSymmetry,
                       Spectral::Quadrature::SphericalSymmetry}}};
  const size_t num_pts = mesh.number_of_grid_points();
  const size_t N = 2 * n_r - 2;
  const DataVector& pts =
      Spectral::collocation_points<Spectral::Basis::ZernikeB1,
                                   Spectral::Quadrature::GaussRadauUpper>(n_r);
  const Approx local_approx = Approx::custom().epsilon(1e-12).scale(1.0);

  // Even parity (m=0): highest mode index is k = N
  {
    using ScalarTags = tmpl::list<Tags::TempScalar<0>>;
    Variables<ScalarTags> vars{num_pts, 0.0};
    get(get<Tags::TempScalar<0>>(vars)) =
        Spectral::compute_basis_function_value<Spectral::Basis::ZernikeB1>(N, 0,
                                                                           pts);
    const Variables<ScalarTags> vars_copy = vars;

    Spectral::filtering::zernike_b1_exponential_filter(make_not_null(&vars),
                                                       mesh, alpha, half_power);

    const double expected_factor = std::exp(-alpha);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get(get<Tags::TempScalar<0>>(vars)),
        expected_factor * get(get<Tags::TempScalar<0>>(vars_copy)),
        local_approx);
  }

  // Odd parity (m=1): highest mode index is k = N - 1
  if (n_r > 1) {
    using VectorTags = tmpl::list<Tags::TempI<0, 3>>;
    Variables<VectorTags> vars{num_pts, 0.0};
    get<0>(get<Tags::TempI<0, 3>>(vars)) =
        Spectral::compute_basis_function_value<Spectral::Basis::ZernikeB1>(
            N - 1, 1, pts);
    const Variables<VectorTags> vars_copy = vars;

    Spectral::filtering::zernike_b1_exponential_filter(make_not_null(&vars),
                                                       mesh, alpha, half_power);

    const double expected_factor = std::exp(-alpha);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<0>(get<Tags::TempI<0, 3>>(vars)),
        expected_factor * get<0>(get<Tags::TempI<0, 3>>(vars_copy)),
        local_approx);
  }
}

void test_spherical_mesh() {
  for (const size_t n_r : {2_st, 3_st, 5_st, 7_st, 8_st, 10_st}) {
    CAPTURE(n_r);
    const Mesh<3> mesh{{{n_r, 1, 1}},
                       {{Spectral::Basis::ZernikeB1, Spectral::Basis::Cartoon,
                         Spectral::Basis::Cartoon}},
                       {{Spectral::Quadrature::GaussRadauUpper,
                         Spectral::Quadrature::SphericalSymmetry,
                         Spectral::Quadrature::SphericalSymmetry}}};

    test_b1_filter<TestTags>(mesh, 10.0, 2u);
    test_b1_filter<TestTags>(mesh, 10.0, 4u);
    test_b1_filter<TestTags>(mesh, 36.0, 2u);
    test_b1_filter<TestTags>(mesh, 36.0, 4u);
  }
}

void test_axial_mesh() {
  for (const size_t n_r : {2_st, 3_st, 5_st, 6_st}) {
    for (const size_t n_y : {3_st, 4_st}) {
      CAPTURE(n_r);
      CAPTURE(n_y);
      const Mesh<3> mesh{
          {{n_r, n_y, 1}},
          {{Spectral::Basis::ZernikeB1, Spectral::Basis::Legendre,
            Spectral::Basis::Cartoon}},
          {{Spectral::Quadrature::GaussRadauUpper,
            Spectral::Quadrature::GaussLobatto,
            Spectral::Quadrature::AxialSymmetry}}};

      test_b1_filter<TestTags>(mesh, 10.0, 2u);
      test_b1_filter<TestTags>(mesh, 10.0, 4u);
      test_b1_filter<TestTags>(mesh, 36.0, 2u);
      test_b1_filter<TestTags>(mesh, 36.0, 4u);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.B1Filter",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test_spherical_mesh();
  test_axial_mesh();
  for (const size_t n_r : {3_st, 5_st, 8_st}) {
    test_highest_mode_damping(n_r, 36.0, 2u);
    test_highest_mode_damping(n_r, 10.0, 4u);
  }
}
}  // namespace
