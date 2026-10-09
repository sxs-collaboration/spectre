// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/InitialDataUtilities/WithNoise.hpp"

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace evolution::initial_data {

template <typename TensorType>
void add_noise_to_tensor(const gsl::not_null<TensorType*> tensor,
                         const double amplitude,
                         const NoiseAmplitudeType amplitude_type,
                         const size_t element_seed,
                         const size_t component_offset) {
  double effective_amplitude = amplitude;
  if (amplitude_type == NoiseAmplitudeType::Relative) {
    double max_abs = 0.0;
    for (const auto& component : *tensor) {
      max_abs = std::max(max_abs, max(abs(component)));
    }
    effective_amplitude *= max_abs;
  }
  if (effective_amplitude == 0.0) {
    return;
  }
  std::uniform_real_distribution<double> dist{-effective_amplitude,
                                              effective_amplitude};
  for (size_t i = 0; i < TensorType::size(); ++i) {
    size_t comp_seed = element_seed;
    boost::hash_combine(comp_seed, component_offset + i);
    std::mt19937_64 gen{comp_seed};
    for (double& val : (*tensor)[i]) {
      val += dist(gen);
    }
  }
}

template <size_t Dim>
size_t make_element_seed(
    const size_t base_seed,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  size_t element_seed = base_seed;
  for (size_t d = 0; d < Dim; ++d) {
    boost::hash_combine(element_seed, inertial_coords.get(d)[0]);
  }
  return element_seed;
}

std::ostream& operator<<(std::ostream& os, const NoiseAmplitudeType value) {
  switch (value) {
    case NoiseAmplitudeType::Absolute:
      return os << "Absolute";
    case NoiseAmplitudeType::Relative:
      return os << "Relative";
    default:
      ERROR("Unknown NoiseAmplitudeType: " << static_cast<int>(value));
  }
}

WithNoise::WithNoise(const WithNoise& rhs)
    : InitialData(rhs),
      solution_(rhs.solution_ != nullptr ? rhs.solution_->get_clone()
                                         : nullptr),
      amplitude_(rhs.amplitude_),
      amplitude_type_(rhs.amplitude_type_),
      seed_(rhs.seed_),
      variables_(rhs.variables_) {}

WithNoise& WithNoise::operator=(const WithNoise& rhs) {
  if (this == &rhs) {
    return *this;
  }
  InitialData::operator=(rhs);
  solution_ = rhs.solution_ != nullptr ? rhs.solution_->get_clone() : nullptr;
  amplitude_ = rhs.amplitude_;
  amplitude_type_ = rhs.amplitude_type_;
  seed_ = rhs.seed_;
  variables_ = rhs.variables_;
  return *this;
}

WithNoise::WithNoise(
    std::unique_ptr<evolution::initial_data::InitialData> solution,
    const double amplitude, const NoiseAmplitudeType amplitude_type,
    std::optional<size_t> seed, std::vector<std::string> variables)
    : solution_(std::move(solution)),
      amplitude_(amplitude),
      amplitude_type_(amplitude_type),
      seed_(seed.value_or(std::random_device{}())),
      variables_(std::move(variables)) {
  if (dynamic_cast<const WithNoise*>(solution_.get()) != nullptr) {
    ERROR("WithNoise cannot wrap another WithNoise. Nesting is not supported.");
  }
}

WithNoise::WithNoise(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> WithNoise::get_clone()
    const {
  return std::make_unique<WithNoise>(*this);
}

void WithNoise::pup(PUP::er& p) {
  evolution::initial_data::InitialData::pup(p);
  p | solution_;
  p | amplitude_;
  p | amplitude_type_;
  p | seed_;
  p | variables_;
}

bool operator==(const WithNoise& lhs, const WithNoise& rhs) {
  if (lhs.amplitude_ != rhs.amplitude_ or
      lhs.amplitude_type_ != rhs.amplitude_type_ or lhs.seed_ != rhs.seed_ or
      lhs.variables_ != rhs.variables_) {
    return false;
  }
  if ((lhs.solution_ == nullptr) != (rhs.solution_ == nullptr)) {
    return false;
  }
  if (lhs.solution_ == nullptr) {
    return true;
  }
  // Compare inner solutions via PUP serialization. Requires Charm++ factory
  // classes to be registered (register_factory_classes_with_charm) before use.
  return serialize(lhs.solution_) == serialize(rhs.solution_);
}

bool operator!=(const WithNoise& lhs, const WithNoise& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID WithNoise::my_PUP_ID = 0;  // NOLINT

}  // namespace evolution::initial_data

template <>
evolution::initial_data::NoiseAmplitudeType
Options::create_from_yaml<evolution::initial_data::NoiseAmplitudeType>::create<
    void>(const Options::Option& options) {
  const auto value = options.parse_as<std::string>();
  if (value == "Absolute") {
    return evolution::initial_data::NoiseAmplitudeType::Absolute;
  } else if (value == "Relative") {
    return evolution::initial_data::NoiseAmplitudeType::Relative;
  }
  PARSE_ERROR(options.context(),
              "Invalid NoiseAmplitudeType '"
                  << value
                  << "'. Valid choices are 'Absolute' and 'Relative'.");
}

namespace evolution::initial_data {
template void add_noise_to_tensor(gsl::not_null<Scalar<DataVector>*> tensor,
                                  double amplitude,
                                  NoiseAmplitudeType amplitude_type,
                                  size_t element_seed, size_t component_offset);

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TENSOR(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_TENSOR(_, data)                             \
  template void add_noise_to_tensor(                            \
      gsl::not_null<tnsr::TENSOR(data) < DataVector, DIM(data), \
                    Frame::Inertial>*> tensor,                  \
      double amplitude, NoiseAmplitudeType amplitude_type,      \
      size_t element_seed, size_t component_offset);

#define INSTANTIATE_DIM(_, data)     \
  template size_t make_element_seed( \
      size_t base_seed,              \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& inertial_coords);

GENERATE_INSTANTIATIONS(INSTANTIATE_TENSOR, (1, 2, 3), (i, I, ii, aa, iaa))
GENERATE_INSTANTIATIONS(INSTANTIATE_DIM, (1, 2, 3))

#undef INSTANTIATE_DIM
#undef INSTANTIATE_TENSOR
#undef TENSOR
#undef DIM
}  // namespace evolution::initial_data
