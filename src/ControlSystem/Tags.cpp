// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Tags.hpp"

#include "Utilities/GenerateInstantiations.hpp"

namespace control_system::Tags::detail {
template <size_t Dim>
void initialize_tuner(
    const gsl::not_null<::TimescaleTuner*> tuner,
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
    const double initial_time, const std::string& name) {
  if (not tuner->timescales_have_been_set()) {
    // We get the functions of time in order to get the number of components
    // so we can resize the number of timescales. Since we are only concerned
    // with the number of components, we don't care about the initial
    // expiration times.
    const auto functions_of_time = domain_creator->functions_of_time();
    const size_t num_components =
        functions_of_time.at(name)->func(initial_time)[0].size();
    tuner->resize_timescales(num_components);
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
