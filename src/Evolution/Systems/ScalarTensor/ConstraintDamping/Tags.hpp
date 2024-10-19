// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Systems/ScalarTensor/ConstraintDamping/DampingFunction.hpp"
#include "Options/String.hpp"

/// \cond
namespace ScalarTensor::OptionTags {
struct Group;
}  // namespace ScalarTensor::OptionTags
/// \endcond

namespace ScalarTensor::ConstraintDamping {
namespace OptionTags {

template <size_t VolumeDim, typename Fr>
struct DampingFunctionGamma1 {
  using type = std::unique_ptr<
      ::ScalarTensor::ConstraintDamping::DampingFunction<VolumeDim, Fr>>;
  static constexpr Options::String help{
      "DampingFunction for damping parameter gamma1"};
  using group = ScalarTensor::OptionTags::Group;
};

template <size_t VolumeDim, typename Fr>
struct DampingFunctionGamma2 {
  using type = std::unique_ptr<
      ::ScalarTensor::ConstraintDamping::DampingFunction<VolumeDim, Fr>>;
  static constexpr Options::String help{
      "DampingFunction for damping parameter gamma2"};
  using group = ScalarTensor::OptionTags::Group;
};
}  // namespace OptionTags

namespace Tags {

/*!
 * \brief A DampingFunction to compute the constraint damping parameter
 * \f$\gamma_0\f$.
 */
template <size_t VolumeDim, typename Fr>
struct DampingFunctionGamma1 : db::SimpleTag {
  using DampingFunctionType =
      ::ScalarTensor::ConstraintDamping::DampingFunction<VolumeDim, Fr>;
  using type = std::unique_ptr<DampingFunctionType>;
  using option_tags = tmpl::list<::ScalarTensor::ConstraintDamping::OptionTags::
                                     DampingFunctionGamma1<VolumeDim, Fr>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& damping_function) {
    return damping_function->get_clone();
  }
};

/*!
 * \brief A DampingFunction to compute the constraint damping parameter
 * \f$\gamma_0\f$.
 */
template <size_t VolumeDim, typename Fr>
struct DampingFunctionGamma2 : db::SimpleTag {
  using DampingFunctionType =
      ::ScalarTensor::ConstraintDamping::DampingFunction<VolumeDim, Fr>;
  using type = std::unique_ptr<DampingFunctionType>;
  using option_tags = tmpl::list<::ScalarTensor::ConstraintDamping::OptionTags::
                                     DampingFunctionGamma2<VolumeDim, Fr>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& damping_function) {
    return damping_function->get_clone();
  }
};
}  // namespace Tags
}  // namespace ScalarTensor::ConstraintDamping
