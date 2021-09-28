// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/SwshTags.hpp"

namespace Spectral {
namespace Swsh {
namespace Tags {
namespace detail {
template <>
std::string compose_spin_weighted_derivative_name<Eth>(
    const std::string& suffix) {
  return "Eth(" + suffix + ")";
}
template <>
std::string compose_spin_weighted_derivative_name<EthEth>(
    const std::string& suffix) {
  return "EthEth(" + suffix + ")";
}
template <>
std::string compose_spin_weighted_derivative_name<EthEthbar>(
    const std::string& suffix) {
  return "EthEthbar(" + suffix + ")";
}
template <>
std::string compose_spin_weighted_derivative_name<Ethbar>(
    const std::string& suffix) {
  return "Ethbar(" + suffix + ")";
}
template <>
std::string compose_spin_weighted_derivative_name<EthbarEth>(
    const std::string& suffix) {
  return "EthbarEth(" + suffix + ")";
}
template <>
std::string compose_spin_weighted_derivative_name<EthbarEthbar>(
    const std::string& suffix) {
  return "EthbarEthbar(" + suffix + ")";
}
template <>
std::string compose_spin_weighted_derivative_name<InverseEth>(
    const std::string& suffix) {
  return "InverseEth(" + suffix + ")";
}
template <>
std::string compose_spin_weighted_derivative_name<InverseEthbar>(
    const std::string& suffix) {
  return "InverseEthbar(" + suffix + ")";
}
template <>
std::string compose_spin_weighted_derivative_name<NoDerivative>(
    const std::string& suffix) {
  return "NoDerivative(" + suffix + ")";
}
}  // namespace detail
}  // namespace Tags
}  // namespace Swsh
}  // namespace Spectral
