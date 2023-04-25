// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Tags/SystemTags.hpp"

#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace control_system::Tags::detail {
template <size_t Dim>
void initialize_tuner(
    const gsl::not_null<::TimescaleTuner*> tuner,
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
    const double initial_time, const std::string& name) {
  // We get the functions of time in order to get the number of components
  // so we can resize the number of timescales. Since we are only concerned
  // with the number of components, we don't care about the initial
  // expiration times. Rotation is special because the number of components in
  // the function of time is 4 (quaternion) but the number of components
  // controlled is 3 (omega), so we hardcode this value.
  const auto functions_of_time = domain_creator->functions_of_time();

  // The only reason the functions of time wouldn't have this control system is
  // if the control system is inactive. Once we remove the ability to read in
  // SpEC control systems, this can be handled outside of this function
  if (functions_of_time.count(name) == 1) {
    if (not tuner->timescales_have_been_set()) {
      const auto& function_of_time = functions_of_time.at(name);

      const auto* casted_quat_fot_2 =
          dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<2>*>(
              function_of_time.get());
      const auto* casted_quat_fot_3 =
          dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
              function_of_time.get());

      size_t num_components = 0;
      if (casted_quat_fot_2 != nullptr or casted_quat_fot_3 != nullptr) {
        num_components = 3;
      } else {
        num_components = function_of_time->func(initial_time)[0].size();
      }

      tuner->resize_timescales(num_components);
    }
  } else {
    if (not tuner->timescales_have_been_set()) {
      // The control system isn't active so just set it to one component
      tuner->resize_timescales(1);
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template void initialize_tuner(                                        \
      const gsl::not_null<::TimescaleTuner*> tuner,                      \
      const std::unique_ptr<::DomainCreator<DIM(data)>>& domain_creator, \
      const double initial_time, const std::string& name);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace control_system::Tags::detail
