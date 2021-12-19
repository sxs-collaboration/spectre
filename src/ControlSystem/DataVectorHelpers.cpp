// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/DataVectorHelpers.hpp"

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

template <size_t N>
DataVector array_to_datavector(const std::array<double, N>& arr) {
  DataVector result{arr.size(), 0.0};
  for (size_t i = 0; i < N; i++) {
    result[i] = gsl::at(arr, i);
  }
  return result;
}

template DataVector array_to_datavector(const std::array<double, 3>&);
