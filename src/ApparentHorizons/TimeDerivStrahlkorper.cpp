// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/TimeDerivStrahlkorper.hpp"

#include <deque>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/FiniteDifference/NonUniform1D.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace ah {
namespace {
template <size_t NumTimes>
static DataVector compute_coefs(
    const std::deque<double>& times,
    const std::deque<const DataVector*>& coefficients) {
  const auto weights = fd::non_uniform_1d_weights<NumTimes>(times);

  DataVector new_coefficients{coefficients.front()->size(), 0.0};

  for (size_t i = 0; i < NumTimes; i++) {
    new_coefficients += *coefficients[i] * gsl::at(gsl::at(weights, 1), i);
  }

  return new_coefficients;
}
}  // namespace

template <typename Frame>
void time_deriv_of_strahlkorper(
    gsl::not_null<Strahlkorper<Frame>*> time_deriv,
    const std::deque<std::pair<double, ::Strahlkorper<Frame>>>&
        previous_strahlkorpers) {
  std::deque<double> times{};
  std::deque<const DataVector*> coefficients{};

  for (const auto& [time, strahlkorper] : previous_strahlkorpers) {
    // This only happens toward the beginning because the first time is NaN and
    // if that happens we can't actually take a derivative
    if (UNLIKELY(std::isnan(time))) {
      time_deriv->coefficients() =
          DataVector{strahlkorper.coefficients().size(), 0.0};
      return;
    }

    times.emplace_back(time);
    coefficients.emplace_back(&strahlkorper.coefficients());
  }

  DataVector new_coefficients{};

  // Switch needed to convert times size into template parameter
  switch (previous_strahlkorpers.size()) {
    case 2:
      new_coefficients = compute_coefs<2>(times, coefficients);
      break;
    case 3:
      new_coefficients = compute_coefs<3>(times, coefficients);
      break;
    case 4:
      new_coefficients = compute_coefs<4>(times, coefficients);
      break;
    default:
      ERROR("Unsupported size for number of previous Strahlkorpers "
            << previous_strahlkorpers.size());
  }

  time_deriv->coefficients() = std::move(new_coefficients);
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                           \
  template void time_deriv_of_strahlkorper(            \
      const gsl::not_null<Strahlkorper<FRAME(data)>*>, \
      const std::deque<std::pair<double, ::Strahlkorper<FRAME(data)>>>&);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::Frame::Grid, ::Frame::Distorted, ::Frame::Inertial))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah
