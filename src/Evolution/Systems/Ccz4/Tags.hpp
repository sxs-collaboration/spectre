// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

namespace Ccz4 {
namespace Tags {
/*!
 * \brief The conformal factor that rescales the spatial metric
 *
 * \details If \f$\gamma_{ij}\f$ is the spatial metric, then we define
 * \f$\phi = (det(\gamma_{ij}))^{-1/6}\f$.
 */
template <typename DataType>
struct ConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The conformally scaled spatial metric
 *
 * \details If \f$\phi\f$ is the conformal factor and \f$\gamma_{ij}\f$ is the
 * spatial metric, then we define
 * \f$\bar{\gamma}_{ij} = \phi^2 \gamma_{ij}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
using ConformalMetric =
    gr::Tags::Conformal<gr::Tags::SpatialMetric<Dim, Frame, DataType>, 2>;

/*!
 * \brief The conformally scaled inverse spatial metric
 *
 * \details If \f$\phi\f$ is the conformal factor and \f$\gamma^{ij}\f$ is the
 * inverse spatial metric, then we define
 * \f$\bar{\gamma}^{ij} = \phi^{-2} \gamma^{ij}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
using InverseConformalMetric =
    gr::Tags::Conformal<gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>,
                        -2>;
}  // namespace Tags

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * Groups option tags related to the Ccz4 evolution system.
 */
struct Group {
  static std::string name() noexcept { return "Ccz4"; }
  static constexpr Options::String help{
      "Options for the CCZ4 evolution system"};
  using group = evolution::OptionTags::SystemGroup;
};
}  // namespace OptionTags
}  // namespace Ccz4
