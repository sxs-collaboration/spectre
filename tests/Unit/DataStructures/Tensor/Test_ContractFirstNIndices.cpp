// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/ContractFirstNIndices.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

template <typename Generator, typename DataType>
void test(const gsl::not_null<Generator*> generator,
          const DataType& used_for_size) {
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  const auto U = make_with_random_values<tnsr::i<DataType, 3, Frame::Grid>>(
      generator, distribution, used_for_size);
  const auto V = make_with_random_values<tnsr::ii<DataType, 3, Frame::Grid>>(
      generator, distribution, used_for_size);

  using UV_type = tnsr::ijj<DataType, 3, Frame::Grid>;
  // contract 0 indices
  const UV_type UV_returned = contract_first_n_indices<0>(U, V);
  UV_type UV_filled{};
  contract_first_n_indices<0>(make_not_null(&UV_filled), U, V);

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = j; k < 3; k++) {
        const auto expected_result = U.get(i) * V.get(j, k);

        CHECK_ITERABLE_APPROX(UV_returned.get(i, j, k), expected_result);
        CHECK_ITERABLE_APPROX(UV_filled.get(i, j, k), expected_result);
      }
    }
  }

  const auto R =
      make_with_random_values<tnsr::abc<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  const auto S =
      make_with_random_values<tnsr::ABc<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);

  // tnsr::abCd
  using RS1_type =
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>;
  // contract first index (spacetime)
  const RS1_type RS1_returned = contract_first_n_indices<1>(R, S);
  RS1_type RS1_filled{};
  contract_first_n_indices<1>(make_not_null(&RS1_filled), R, S);

  for (size_t b = 0; b < 4; b++) {
    for (size_t c = 0; c < 4; c++) {
      for (size_t d = 0; d < 4; d++) {
        for (size_t e = 0; e < 4; e++) {
          DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
          for (size_t a = 0; a < 4; a++) {
            expected_sum += R.get(a, b, c) * S.get(a, d, e);
          }

          CHECK_ITERABLE_APPROX(RS1_returned.get(b, c, d, e), expected_sum);
          CHECK_ITERABLE_APPROX(RS1_filled.get(b, c, d, e), expected_sum);
        }
      }
    }
  }

  using RS2_type = tnsr::ab<DataType, 3, Frame::Inertial>;
  // contract first two indices (both spacetime)
  const RS2_type RS2_returned = contract_first_n_indices<2>(R, S);
  RS2_type RS2_filled{};
  contract_first_n_indices<2>(make_not_null(&RS2_filled), R, S);

  for (size_t c = 0; c < 4; c++) {
    for (size_t d = 0; d < 4; d++) {
      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t a = 0; a < 4; a++) {
        for (size_t b = 0; b < 4; b++) {
          expected_sum += R.get(a, b, c) * S.get(a, b, d);
        }
      }

      CHECK_ITERABLE_APPROX(RS2_returned.get(c, d), expected_sum);
      CHECK_ITERABLE_APPROX(RS2_filled.get(c, d), expected_sum);
    }
  }

  const auto G =
      make_with_random_values<tnsr::Ijaa<DataType, 3, Frame::Inertial>>(
          generator, distribution, used_for_size);
  // tnsr::iJA
  const auto H = make_with_random_values<
      Tensor<DataType, Symmetry<3, 2, 1>,
             index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>>(
      generator, distribution, used_for_size);

  using GH_type = tnsr::aaB<DataType, 3, Frame::Inertial>;
  // contract first two indices (both spatial) of two tensors of different rank
  const GH_type GH_returned = contract_first_n_indices<2>(G, H);
  GH_type GH_filled{};
  contract_first_n_indices<2>(make_not_null(&GH_filled), G, H);

  using HG_type = tnsr::Abb<DataType, 3, Frame::Inertial>;
  // for checking that opposite order of operands gives us the "same" result
  // mathematically (though the LHS index order will be different)
  const HG_type HG_returned = contract_first_n_indices<2>(H, G);
  HG_type HG_filled{};
  contract_first_n_indices<2>(make_not_null(&HG_filled), H, G);

  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      for (size_t c = 0; c < 4; c++) {
        DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
        for (size_t i = 0; i < 3; i++) {
          for (size_t j = 0; j < 3; j++) {
            expected_sum += G.get(i, j, a, b) * H.get(i, j, c);
          }
        }

        CHECK_ITERABLE_APPROX(GH_returned.get(a, b, c), expected_sum);
        CHECK_ITERABLE_APPROX(GH_filled.get(a, b, c), expected_sum);
        CHECK_ITERABLE_APPROX(HG_returned.get(c, a, b), expected_sum);
        CHECK_ITERABLE_APPROX(HG_filled.get(c, a, b), expected_sum);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.ContractFirstNIndices",
                  "[DataStructures][Unit]") {
  MAKE_GENERATOR(generator);

  test(make_not_null(&generator), std::numeric_limits<double>::signaling_NaN());
  test(make_not_null(&generator),
       DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
