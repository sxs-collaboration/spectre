// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GetOutput.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
template <size_t Dim>
struct ElementId;
template <size_t Dim>
struct Mesh;
/// \endcond

/*!
 * \ingroup DataStructuresGroup
 * \brief An untyped tensor component with a name for observation.
 *
 * The name should be just the name of the tensor component, such as 'Psi_xx'.
 * It must not include any slashes ('/').
 */
struct TensorComponent {
  TensorComponent() = default;
  TensorComponent(std::string in_name, DataVector in_data);
  TensorComponent(std::string in_name, std::vector<float> in_data);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
  std::string name{};
  std::variant<DataVector, std::vector<float>> data{};
};

std::ostream& operator<<(std::ostream& os, const TensorComponent& t);

bool operator==(const TensorComponent& lhs, const TensorComponent& rhs);

bool operator!=(const TensorComponent& lhs, const TensorComponent& rhs);

/*!
 * \ingroup DataStructuresGroup
 * \brief Holds tensor components on a grid, to be written into an H5 file
 */
struct ElementVolumeData {
  ElementVolumeData() = default;
  ElementVolumeData(std::string element_name_in,
                    std::vector<TensorComponent> components,
                    std::vector<size_t> extents_in,
                    std::vector<Spectral::Basis> basis_in,
                    std::vector<Spectral::Quadrature> quadrature_in);
  template <size_t Dim>
  ElementVolumeData(const ElementId<Dim>& element_id,
                    std::vector<TensorComponent> components,
                    const Mesh<Dim>& mesh);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
  /// Name of the grid (should be human-readable). For standard volume data this
  /// name must be the string representation of an ElementId, such as
  /// [B0,(L0I1,L0I0,L3I2)]. Code that reads the volume data may rely on this
  /// pattern to reconstruct the ElementId.
  std::string element_name{};
  /// All tensor components on the grid
  std::vector<TensorComponent> tensor_components{};
  /// Number of grid points in every dimension of the grid
  std::vector<size_t> extents{};
  /// Spectral::Basis in every dimension of the grid
  std::vector<Spectral::Basis> basis{};
  /// Spectral::Quadrature in every dimension of the grid
  std::vector<Spectral::Quadrature> quadrature{};
};

bool operator==(const ElementVolumeData& lhs, const ElementVolumeData& rhs);
bool operator!=(const ElementVolumeData& lhs, const ElementVolumeData& rhs);
