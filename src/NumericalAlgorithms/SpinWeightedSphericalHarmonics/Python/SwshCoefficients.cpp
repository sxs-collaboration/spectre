// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/Python/SwshCoefficients.hpp"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/ComplexDataView.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/Python/InterleavedHelpers.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace py = pybind11;

namespace py_bindings {
namespace {
// goldberg modes -> libsharp modes -> nodal collocation values. Operates on a
// single angular slice (number_of_radial_points == 1).
template <int Spin>
DataVector goldberg_to_nodal_for_spin(const DataVector& goldberg_interleaved,
                                      const size_t l_max) {
  const size_t expected_size = 2 * square(l_max + 1);
  if (goldberg_interleaved.size() != expected_size) {
    throw std::invalid_argument(
        "The interleaved goldberg modes have " +
        std::to_string(goldberg_interleaved.size()) +
        " entries, but l_max = " + std::to_string(l_max) +
        " requires 2 * (l_max + 1)^2 = " + std::to_string(expected_size) +
        " entries.");
  }
  const SpinWeighted<ComplexModalVector, Spin> goldberg_modes{
      detail::interleaved_to_complex<ComplexModalVector>(goldberg_interleaved)};
  const auto libsharp_modes =
      Spectral::Swsh::goldberg_to_libsharp_modes(goldberg_modes, l_max);
  const auto nodal_values =
      Spectral::Swsh::inverse_swsh_transform(l_max, 1, libsharp_modes);
  return detail::complex_to_interleaved(nodal_values.data());
}

// nodal collocation values -> libsharp modes -> goldberg modes.
template <int Spin>
DataVector nodal_to_goldberg_for_spin(const DataVector& nodal_interleaved,
                                      const size_t l_max) {
  const size_t expected_size =
      2 * Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  if (nodal_interleaved.size() != expected_size) {
    throw std::invalid_argument(
        "The interleaved nodal values have " +
        std::to_string(nodal_interleaved.size()) +
        " entries, but l_max = " + std::to_string(l_max) +
        " requires 2 * (l_max + 1) * (2 * l_max + 1) = " +
        std::to_string(expected_size) + " entries.");
  }
  const SpinWeighted<ComplexDataVector, Spin> nodal_values{
      detail::interleaved_to_complex<ComplexDataVector>(nodal_interleaved)};
  const auto libsharp_modes =
      Spectral::Swsh::swsh_transform(l_max, 1, nodal_values);
  const auto goldberg_modes =
      Spectral::Swsh::libsharp_to_goldberg_modes(libsharp_modes, l_max);
  return detail::complex_to_interleaved(goldberg_modes.data());
}
}  // namespace

void bind_goldberg_to_nodal(py::module& m) {  // NOLINT
  m.def(
      "goldberg_to_nodal",
      [](const DataVector& goldberg_modes_interleaved, const size_t l_max,
         const int spin) {
        DataVector result{};
        bool spin_handled = false;
        const auto try_spin = [&]<int Spin>() {
          if (spin == Spin) {
            result = goldberg_to_nodal_for_spin<Spin>(
                goldberg_modes_interleaved, l_max);
            spin_handled = true;
          }
        };
        (try_spin.template operator()<-2>(), try_spin.template operator()<-1>(),
         try_spin.template operator()<0>(), try_spin.template operator()<1>(),
         try_spin.template operator()<2>());
        if (not spin_handled) {
          throw std::invalid_argument(
              "Spin weight " + std::to_string(spin) +
              " is outside the supported range [-2, 2].");
        }
        return result;
      },
      py::arg("goldberg_modes_interleaved"), py::arg("l_max"), py::arg("spin"));
}

void bind_nodal_to_goldberg(py::module& m) {  // NOLINT
  m.def(
      "nodal_to_goldberg",
      [](const DataVector& nodal_values_interleaved, const size_t l_max,
         const int spin) {
        DataVector result{};
        bool spin_handled = false;
        const auto try_spin = [&]<int Spin>() {
          if (spin == Spin) {
            result = nodal_to_goldberg_for_spin<Spin>(nodal_values_interleaved,
                                                      l_max);
            spin_handled = true;
          }
        };
        (try_spin.template operator()<-2>(), try_spin.template operator()<-1>(),
         try_spin.template operator()<0>(), try_spin.template operator()<1>(),
         try_spin.template operator()<2>());
        if (not spin_handled) {
          throw std::invalid_argument(
              "Spin weight " + std::to_string(spin) +
              " is outside the supported range [-2, 2].");
        }
        return result;
      },
      py::arg("nodal_values_interleaved"), py::arg("l_max"), py::arg("spin"));
}
}  // namespace py_bindings
