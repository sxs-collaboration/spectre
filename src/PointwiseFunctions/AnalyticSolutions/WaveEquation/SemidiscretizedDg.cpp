// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SemidiscretizedDg.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <pup.h>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace ScalarWave::Solutions {
SemidiscretizedDg::SemidiscretizedDg(
    const int harmonic, const std::array<double, 4>& amplitudes) noexcept
    : harmonic_(harmonic), amplitudes_(amplitudes) {}

namespace {
struct Mode {
  std::complex<double> frequency;
  ComplexDataVector pi_coefficients{2};
  ComplexDataVector phi_coefficients{2};
};

std::array<Mode, 4> get_modes(const tnsr::I<DataVector, 1>& x,
                              const int harmonic) noexcept {
  using namespace std::complex_literals;

  if (get<0>(x).size() != 2) {
    ERROR("SemidiscretizedDg solution only supports linear elements.");
  }

  const double element_length = get<0>(x)[1] - get<0>(x)[0];
  // Phase change across an element
  const double element_phase = harmonic * element_length;

  std::array<Mode, 4> result;
  {
    std::complex<double> one_plus_i_omega =
        sqrt(2.0 * exp(1.0i * element_phase) - 1.0);
    {
      Mode& mode = result[0];
      // Same as
      //   mode.frequency = 1.0i * (1.0 - one_plus_i_omega) / element_length
      // but better behaved as one_plus_i_omega -> 1
      mode.frequency = 4.0 * sin(element_phase) /
                       ((1.0 + exp(-1.0i * element_phase)) *
                        (1.0 + one_plus_i_omega) * element_length);
      mode.pi_coefficients[0] = sqrt(1.0 / one_plus_i_omega);
      mode.pi_coefficients[1] = 1.0 / mode.pi_coefficients[0];
      mode.phi_coefficients = -mode.pi_coefficients;
    }
    {
      Mode& mode = result[1];
      mode.frequency = 1.0i * (1.0 + one_plus_i_omega) / element_length;
      mode.pi_coefficients[0] = sqrt(1.0 / one_plus_i_omega);
      mode.pi_coefficients[1] = -1.0 / mode.pi_coefficients[0];
      mode.phi_coefficients = -mode.pi_coefficients;
    }
  }
  {
    std::complex<double> one_plus_i_omega =
        sqrt(2.0 * exp(-1.0i * element_phase) - 1.0);
    {
      Mode& mode = result[2];
      // Same as
      //   mode.frequency = 1.0i * (1.0 - one_plus_i_omega) / element_length
      // but better behaved as one_plus_i_omega -> 1
      mode.frequency = -4.0 * sin(element_phase) /
                       ((1.0 + exp(1.0i * element_phase)) *
                        (1.0 + one_plus_i_omega) * element_length);
      mode.pi_coefficients[0] = sqrt(one_plus_i_omega);
      mode.pi_coefficients[1] = 1.0 / mode.pi_coefficients[0];
      mode.phi_coefficients = mode.pi_coefficients;
    }
    {
      Mode& mode = result[3];
      mode.frequency = 1.0i * (1.0 + one_plus_i_omega) / element_length;
      mode.pi_coefficients[0] = sqrt(one_plus_i_omega);
      mode.pi_coefficients[1] = -1.0 / mode.pi_coefficients[0];
      mode.phi_coefficients = mode.pi_coefficients;
    }
  }
  return result;
}
}  // namespace

tuples::TaggedTuple<ScalarWave::Pi> SemidiscretizedDg::variables(
    const tnsr::I<DataVector, 1>& x, double t,
    tmpl::list<ScalarWave::Pi> /*meta*/) const noexcept {
  using namespace std::complex_literals;

  const auto modes = get_modes(x, harmonic_);
  const double spatial_phase = harmonic_ * get<0>(x)[0];
  DataVector pi{0.0, 0.0};
  for (size_t i = 0; i < 4; ++i) {
    const auto& mode = gsl::at(modes, i);
    pi += gsl::at(amplitudes_, i) *
          real(exp(1.0i * (spatial_phase + mode.frequency * t)) *
               mode.pi_coefficients);
  }
  return {Scalar<DataVector>(std::move(pi))};
}

tuples::TaggedTuple<ScalarWave::Phi<1>> SemidiscretizedDg::variables(
    const tnsr::I<DataVector, 1>& x, double t,
    tmpl::list<ScalarWave::Phi<1>> /*meta*/) const noexcept {
  using namespace std::complex_literals;

  const auto modes = get_modes(x, harmonic_);
  const double spatial_phase = harmonic_ * get<0>(x)[0];
  DataVector phi{0.0, 0.0};
  for (size_t i = 0; i < 4; ++i) {
    const auto& mode = gsl::at(modes, i);
    phi += gsl::at(amplitudes_, i) *
           real(exp(1.0i * (spatial_phase + mode.frequency * t)) *
                mode.phi_coefficients);
  }
  return {tnsr::i<DataVector, 1>{{{std::move(phi)}}}};
}

tuples::TaggedTuple<ScalarWave::Psi> SemidiscretizedDg::variables(
    const tnsr::I<DataVector, 1>& x, double t,
    tmpl::list<ScalarWave::Psi> /*meta*/) const noexcept {
  using namespace std::complex_literals;

  // There are two more modes that are just constant offsets of Psi,
  // but they are boring so we ignore them.
  const auto modes = get_modes(x, harmonic_);
  const double spatial_phase = harmonic_ * get<0>(x)[0];
  DataVector psi{0.0, 0.0};
  for (size_t i = 0; i < 4; ++i) {
    const auto& mode = gsl::at(modes, i);
    if (mode.frequency == 0.0) {
      psi -= gsl::at(amplitudes_, i) *
             real(t * exp(1.0i * spatial_phase) * mode.pi_coefficients);
    } else {
      psi -= gsl::at(amplitudes_, i) *
             real(exp(1.0i * (spatial_phase + mode.frequency * t)) /
                  (1.0i * mode.frequency) * mode.pi_coefficients);
    }
  }
  return {Scalar<DataVector>(std::move(psi))};
}

void SemidiscretizedDg::pup(PUP::er& p) noexcept {
  p | harmonic_;
  p | amplitudes_;
}
}  // namespace ScalarWave::Solutions
