// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SecondOrderWrapper.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace SecondOrderScalarWave::Solutions {
template <typename SolutionType>
SecondOrderWrapper<SolutionType>::SecondOrderWrapper(CkMigrateMessage* msg)
    : evolution::initial_data::InitialData(msg) {}

template <typename SolutionType>
std::unique_ptr<evolution::initial_data::InitialData>
SecondOrderWrapper<SolutionType>::get_clone() const {
  return std::make_unique<SecondOrderWrapper<SolutionType>>(*this);
}

template <typename SolutionType>
tuples::TaggedTuple<SecondOrderScalarWave::Tags::Psi,
                    SecondOrderScalarWave::Tags::Pi,
                    SecondOrderScalarWave::Tags::Phi<
                        SecondOrderWrapper<SolutionType>::volume_dim>>
SecondOrderWrapper<SolutionType>::variables(
    const tnsr::I<DataVector, SecondOrderWrapper<SolutionType>::volume_dim>& x,
    const double t,
    const tmpl::list<SecondOrderScalarWave::Tags::Psi,
                     SecondOrderScalarWave::Tags::Pi,
                     SecondOrderScalarWave::Tags::Phi<
                         SecondOrderWrapper<SolutionType>::volume_dim>>
    /*meta*/) const {
  auto scalar_wave_vars = wrapped_solution_.variables(
      x, t,
      tmpl::list<ScalarWave::Tags::Psi, ScalarWave::Tags::Pi,
                 ScalarWave::Tags::Phi<volume_dim>>{});
  return {std::move(get<ScalarWave::Tags::Psi>(scalar_wave_vars)),
          std::move(get<ScalarWave::Tags::Pi>(scalar_wave_vars)),
          std::move(get<ScalarWave::Tags::Phi<volume_dim>>(scalar_wave_vars))};
}

template <typename SolutionType>
tuples::TaggedTuple<SecondOrderScalarWave::Tags::Psi,
                    SecondOrderScalarWave::Tags::Pi>
SecondOrderWrapper<SolutionType>::variables(
    const tnsr::I<DataVector, SecondOrderWrapper<SolutionType>::volume_dim>& x,
    const double t,
    const tmpl::list<SecondOrderScalarWave::Tags::Psi,
                     SecondOrderScalarWave::Tags::Pi> /*meta*/) const {
  auto scalar_wave_vars = wrapped_solution_.variables(
      x, t,
      tmpl::list<ScalarWave::Tags::Psi, ScalarWave::Tags::Pi,
                 ScalarWave::Tags::Phi<volume_dim>>{});
  return {std::move(get<ScalarWave::Tags::Psi>(scalar_wave_vars)),
          std::move(get<ScalarWave::Tags::Pi>(scalar_wave_vars))};
}

template <typename SolutionType>
tuples::TaggedTuple<::Tags::dt<SecondOrderScalarWave::Tags::Psi>,
                    ::Tags::dt<SecondOrderScalarWave::Tags::Pi>>
SecondOrderWrapper<SolutionType>::variables(
    const tnsr::I<DataVector, SecondOrderWrapper<SolutionType>::volume_dim>& x,
    const double t,
    const tmpl::list<::Tags::dt<SecondOrderScalarWave::Tags::Psi>,
                     ::Tags::dt<SecondOrderScalarWave::Tags::Pi>> /*meta*/)
    const {
  auto scalar_wave_dt_vars = wrapped_solution_.variables(
      x, t,
      tmpl::list<::Tags::dt<ScalarWave::Tags::Psi>,
                 ::Tags::dt<ScalarWave::Tags::Pi>,
                 ::Tags::dt<ScalarWave::Tags::Phi<volume_dim>>>{});
  // The second-order-in-space system does not evolve Phi, so its time
  // derivative from the wrapped solution is discarded.
  return {
      std::move(get<::Tags::dt<ScalarWave::Tags::Psi>>(scalar_wave_dt_vars)),
      std::move(get<::Tags::dt<ScalarWave::Tags::Pi>>(scalar_wave_dt_vars))};
}

template <typename SolutionType>
void SecondOrderWrapper<SolutionType>::pup(PUP::er& p) {
  evolution::initial_data::InitialData::pup(p);
  p | wrapped_solution_;
}

template <typename SolutionType>
PUP::able::PUP_ID SecondOrderWrapper<SolutionType>::my_PUP_ID = 0;

#define SECOND_ORDER_WRAPPER_SOLUTION_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define SECOND_ORDER_WRAPPER_INSTANTIATE(_, data)                      \
  template class SecondOrderScalarWave::Solutions::SecondOrderWrapper< \
      SECOND_ORDER_WRAPPER_SOLUTION_TYPE(data)>;
}  // namespace SecondOrderScalarWave::Solutions
