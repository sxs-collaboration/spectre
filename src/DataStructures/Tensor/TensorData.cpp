// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/TensorData.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

TensorComponent::TensorComponent(std::string n, DataVector d) noexcept
    : name(std::move(n)), data(std::move(d)) {}

void TensorComponent::pup(PUP::er& p) noexcept {
  p | name;
  p | data;
}

std::ostream& operator<<(std::ostream& os, const TensorComponent& t) noexcept {
  return os << "(" << t.name << ", " << t.data << ")";
}

bool operator==(const TensorComponent& lhs,
                const TensorComponent& rhs) noexcept {
  return lhs.name == rhs.name and lhs.data == rhs.data;
}

bool operator!=(const TensorComponent& lhs,
                const TensorComponent& rhs) noexcept {
  return not(lhs == rhs);
}

ExtentsAndTensorVolumeData::ExtentsAndTensorVolumeData(
    std::vector<size_t> extents_in,
    std::vector<TensorComponent> components) noexcept
    : extents(std::move(extents_in)),
      tensor_components(std::move(components)) {}

void ExtentsAndTensorVolumeData::pup(PUP::er& p) noexcept {
  p | extents;
  p | tensor_components;
}

ElementVolumeData::ElementVolumeData(
    std::vector<size_t> extents_in, std::vector<TensorComponent> components,
    std::vector<Spectral::Basis> basis_in,
    std::vector<Spectral::Quadrature> quadrature_in) noexcept
    : ExtentsAndTensorVolumeData(std::move(extents_in), std::move(components)),
      basis(std::move(basis_in)),
      quadrature(std::move(quadrature_in)) {}

void ElementVolumeData::pup(PUP::er& p) noexcept {
  ExtentsAndTensorVolumeData::pup(p);
  p | quadrature;
  p | basis;
}
