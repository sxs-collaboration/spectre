// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <string>
#include <vector>

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace h5::detail {
class OpenGroup;
}  // namespace h5::detail

namespace h5_detail {
/// We maintain a list of bases and quadratures which are compatible
/// with IO, this allows for efficient storage of Spectral::Basis in
/// volume data files (currently there are 3 bases)
std::array<Spectral::Basis, 3> allowed_bases();

/// We maintain a list of quadratures which are compatible
/// with IO, this allows for efficient storage of Spectral::Quadrature in
/// volume data files (currently there are 4 quadratures)
std::array<Spectral::Quadrature, 4> allowed_quadratures();

/// Write a dictionary as an attribute to the volume file, can be used
/// to decode integer sequence as values[i] represents the string
/// value encoded with integer i in the h5 file
void write_dictionary(const std::string& dict_name,
                      const std::vector<std::string>& values,
                      const h5::detail::OpenGroup& observation_group);

/// A dictionary `dict_name` is used to decode the integer vector `decodable`
/// into an vector of strings.  The `dict_name` should correspond to
/// a dictionary written with `h5_detail::write_dictionary`
std::vector<std::string> decode_with_dictionary_name(
    const std::string& dict_name, const std::vector<int>& decodable,
    const h5::detail::OpenGroup& observation_group);

}  // namespace h5_detail
