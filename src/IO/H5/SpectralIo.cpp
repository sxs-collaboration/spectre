// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "SpectralIo.hpp"

#include <array>
#include <string>
#include <vector>

#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Basis.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Quadrature.hpp"

namespace h5_detail {

std::array<SpatialDiscretization::Basis, 4> allowed_bases() {
  return {SpatialDiscretization::Basis::Chebyshev,
          SpatialDiscretization::Basis::Legendre,
          SpatialDiscretization::Basis::FiniteDifference,
          SpatialDiscretization::Basis::SphericalHarmonic};
}

std::array<SpatialDiscretization::Quadrature, 5> allowed_quadratures() {
  return {SpatialDiscretization::Quadrature::Gauss,
          SpatialDiscretization::Quadrature::GaussLobatto,
          SpatialDiscretization::Quadrature::CellCentered,
          SpatialDiscretization::Quadrature::FaceCentered,
          SpatialDiscretization::Quadrature::Equiangular};
}

void write_dictionary(const std::string& dict_name,
                      const std::vector<std::string>& values,
                      const h5::detail::OpenGroup& observation_group) {
  h5::write_to_attribute<std::string>(observation_group.id(), dict_name,
                                      values);
}

std::vector<std::string> decode_with_dictionary_name(
    const std::string& dict_name, const std::vector<int>& decodable,
    const h5::detail::OpenGroup& observation_group) {
  const auto dict =
      h5::read_rank1_attribute<std::string>(observation_group.id(), dict_name);

  std::vector<std::string> decoded(decodable.size());
  for (size_t i = 0; i < decodable.size(); i++) {
    decoded[i] = dict[static_cast<size_t>(decodable[i])];
  }
  return decoded;
}

}  // namespace h5_detail
