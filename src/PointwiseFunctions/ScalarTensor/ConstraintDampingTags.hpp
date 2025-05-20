// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"

/// \cond
namespace ScalarTensor::OptionTags {
struct Group;
}  // namespace ScalarTensor::OptionTags
/// \endcond

namespace ScalarTensor {
namespace OptionTags {
/*!
 * \brief A DampingFunction to compute the constraint damping parameter
 * \f$\gamma_1\f$.
 */
template <size_t VolumeDim, typename Fr>
struct DampingFunctionGamma1 {
  using type =
      std::unique_ptr<::ConstraintDamping::DampingFunction<VolumeDim, Fr>>;
  static constexpr Options::String help{
      "DampingFunction for damping parameter gamma1"};
  using group = ::ScalarTensor::OptionTags::Group;
};

/*!
 * \brief A DampingFunction to compute the constraint damping parameter
 * \f$\gamma_2\f$.
 */
template <size_t VolumeDim, typename Fr>
struct DampingFunctionGamma2 {
  using type =
      std::unique_ptr<::ConstraintDamping::DampingFunction<VolumeDim, Fr>>;
  static constexpr Options::String help{
      "DampingFunction for damping parameter gamma2"};
  using group = ::ScalarTensor::OptionTags::Group;
};
}  // namespace OptionTags

namespace Tags {
/// \copydoc ScalarTensor::OptionTags::DampingFunctionGamma1
template <size_t VolumeDim, typename Fr>
struct DampingFunctionGamma1 : db::SimpleTag {
  using DampingFunctionType =
      ::ConstraintDamping::DampingFunction<VolumeDim, Fr>;
  using type = std::unique_ptr<DampingFunctionType>;
  using option_tags = tmpl::list<
      ::ScalarTensor::OptionTags::DampingFunctionGamma1<VolumeDim, Fr>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& damping_function) {
    return damping_function->get_clone();
  }
};

/// \copydoc ScalarTensor::OptionTags::DampingFunctionGamma2
template <size_t VolumeDim, typename Fr>
struct DampingFunctionGamma2 : db::SimpleTag {
  using DampingFunctionType =
      ::ConstraintDamping::DampingFunction<VolumeDim, Fr>;
  using type = std::unique_ptr<DampingFunctionType>;
  using option_tags = tmpl::list<
      ::ScalarTensor::OptionTags::DampingFunctionGamma2<VolumeDim, Fr>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& damping_function) {
    return damping_function->get_clone();
  }
};
}  // namespace Tags
}  // namespace ScalarTensor
