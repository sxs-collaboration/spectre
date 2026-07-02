// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/Python/SwshDerivatives.hpp"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/Python/InterleavedHelpers.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"

namespace py = pybind11;

namespace py_bindings {
namespace {
// Apply a single spin-weighted angular derivative to nodal values supplied as a
// real, interleaved [re, im, re, im, ...] array (matching the goldberg_to_nodal
// and nodal_to_goldberg bindings), returning the result in the same layout.
template <typename DerivativeKind, int Spin>
DataVector apply_swsh_derivative(const DataVector& nodal_values_interleaved,
                                 const size_t l_max,
                                 const size_t number_of_radial_points) {
  const size_t expected_size =
      2 * Spectral::Swsh::number_of_swsh_collocation_points(l_max) *
      number_of_radial_points;
  if (nodal_values_interleaved.size() != expected_size) {
    throw std::invalid_argument(
        "The interleaved nodal values have " +
        std::to_string(nodal_values_interleaved.size()) +
        " entries, but l_max = " + std::to_string(l_max) + " with " +
        std::to_string(number_of_radial_points) +
        " radial point(s) requires 2 * (l_max + 1) * (2 * l_max + 1) * "
        "number_of_radial_points = " +
        std::to_string(expected_size) + " entries.");
  }
  const SpinWeighted<ComplexDataVector, Spin> input{
      detail::interleaved_to_complex<ComplexDataVector>(
          nodal_values_interleaved)};
  const auto output = Spectral::Swsh::angular_derivative<
      DerivativeKind, Spectral::Swsh::ComplexRepresentation::Interleaved, Spin>(
      l_max, number_of_radial_points, input);
  return detail::complex_to_interleaved(output.data());
}

// Bind one derivative operator over the spin weights for which it is
// instantiated in the core library (see SwshDerivatives.cpp).
template <typename DerivativeKind, int... Spins>
void bind_derivative_kind(py::module& m, const std::string& name) {
  m.def(
      name.c_str(),
      [](const DataVector& nodal_values, const size_t l_max,
         const size_t number_of_radial_points, const int spin) {
        DataVector result{};
        bool spin_handled = false;
        (
            [&] {
              if (spin == Spins) {
                result = apply_swsh_derivative<DerivativeKind, Spins>(
                    nodal_values, l_max, number_of_radial_points);
                spin_handled = true;
              }
            }(),
            ...);
        if (not spin_handled) {
          throw std::invalid_argument(
              "The requested spin weight " + std::to_string(spin) +
              " is not supported for this derivative operator.");
        }
        return result;
      },
      py::arg("nodal_values"), py::arg("l_max"),
      py::arg("number_of_radial_points"), py::arg("spin"));
}
}  // namespace

void bind_swsh_derivatives(py::module& m) {  // NOLINT
  bind_derivative_kind<Spectral::Swsh::Tags::Eth, -2, -1, 0, 1, 2>(m, "eth");
  bind_derivative_kind<Spectral::Swsh::Tags::Ethbar, -1, 0, 1, 2>(m, "ethbar");
  bind_derivative_kind<Spectral::Swsh::Tags::EthEth, -2, -1, 0>(m, "eth_eth");
  bind_derivative_kind<Spectral::Swsh::Tags::EthbarEthbar, 0, 1, 2>(
      m, "ethbar_ethbar");
  bind_derivative_kind<Spectral::Swsh::Tags::EthEthbar, -2, -1, 0, 1, 2>(
      m, "eth_ethbar");
  bind_derivative_kind<Spectral::Swsh::Tags::EthbarEth, -2, -1, 0, 1, 2>(
      m, "ethbar_eth");
}
}  // namespace py_bindings
