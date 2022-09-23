// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <random>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// \brief Test evaluation of tensors where concrete time indices are used for
// RHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename Generator, typename DataType>
void test_rhs(const gsl::not_null<Generator*> generator,
              const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // \f$L_{a} = R_{at}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>
      La_from_R_at = tenex::evaluate<ti::a>(R(ti::a, ti::t));

  for (size_t a = 0; a < dim + 1; a++) {
    CHECK(La_from_R_at.get(a) == R.get(a, 0));
  }
}

// \brief Test evaluation of tensors where concrete time indices are used for
// LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename Generator, typename DataType>
void test_lhs(const gsl::not_null<Generator*> generator,
              const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  // \f$L_{at} = R_{a}\f$
  //
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  //
  // Assign a placeholder to the LHS tensor's components before it is computed
  // so that when test expressions below only compute time components, we can
  // check that LHS spatial components haven't changed
  auto Lat_from_R_a = make_with_value<
      Tensor<DataType, Symmetry<2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Inertial>>>>(
      used_for_size,
      TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::a, ti::t>(make_not_null(&Lat_from_R_a), R(ti::a));

  for (size_t a = 0; a < dim + 1; a++) {
    CHECK(Lat_from_R_a.get(a, 0) == R.get(a));
    for (size_t i = 0; i < dim; i++) {
      CHECK(Lat_from_R_a.get(a, i + 1) ==
            TestHelpers::tenex::component_placeholder_value<DataType>::value);
    }
  }
}

// \brief Test evaluation of tensors where concrete time indices are used for
// RHS and LHS spacetime indices
//
// \tparam DataType the type of data being stored in the expression operands
template <typename Generator, typename DataType>
void test_rhs_and_lhs(const gsl::not_null<Generator*> generator,
                      const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  constexpr size_t dim = 3;

  const auto R = make_with_random_values<
      Tensor<DataType, Symmetry<2, 2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>>>>(
      generator, distribution, used_for_size);

  // \f$L_{a}{}^{t}{}_{tb} = R_{tba}\f$
  //
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  //
  // Assign a placeholder to the LHS tensor's components before it is computed
  // so that when test expressions below only compute time components, we can
  // check that LHS spatial components haven't changed
  auto LaTtb_from_R_tba = make_with_value<
      Tensor<DataType, Symmetry<2, 3, 2, 1>,
             index_list<SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Up, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<dim, UpLo::Lo, Frame::Grid>>>>(
      used_for_size,
      TestHelpers::tenex::component_placeholder_value<DataType>::value);
  tenex::evaluate<ti::a, ti::T, ti::t, ti::b>(make_not_null(&LaTtb_from_R_tba),
                                              R(ti::t, ti::b, ti::a));

  for (size_t a = 0; a < dim + 1; a++) {
    for (size_t b = 0; b < dim + 1; b++) {
      CHECK(LaTtb_from_R_tba.get(a, 0, 0, b) == R.get(0, b, a));

      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          CHECK(
              LaTtb_from_R_tba.get(a, i + 1, j + 1, b) ==
              TestHelpers::tenex::component_placeholder_value<DataType>::value);
        }
      }
    }
  }
}

template <typename DataType>
void test_evaluate_spatial_spacetime_index(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);

  test_rhs(make_not_null(&generator), used_for_size);
  test_lhs(make_not_null(&generator), used_for_size);
  test_rhs_and_lhs(make_not_null(&generator), used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.EvaluateTimeIndex",
                  "[DataStructures][Unit]") {
  test_evaluate_spatial_spacetime_index(
      std::numeric_limits<double>::signaling_NaN());
  test_evaluate_spatial_spacetime_index(
      std::complex<double>(std::numeric_limits<double>::signaling_NaN(),
                           std::numeric_limits<double>::signaling_NaN()));
  test_evaluate_spatial_spacetime_index(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  test_evaluate_spatial_spacetime_index(
      ComplexDataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
