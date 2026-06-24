// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <string>
#include <unordered_map>

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegionGenerator.hpp"
#include "Utilities/TMPL.hpp"

template <size_t VolumeDim>
class ElementId;
namespace PUP {
class er;
}  // namespace PUP

namespace {
struct OptInt {
  using type = int;
};

struct OptString {
  using type = std::string;
};

class ValidGenerator {
 public:
  explicit ValidGenerator();

  using creation_tags = tmpl::list<OptInt, OptString>;

  explicit ValidGenerator(int int_opt, const std::string& string_opt);

  std::unordered_map<std::string, size_t> regions() const;

  template <size_t Dim>
  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p);
};
static_assert(evolution::dg::equal_rate_region_generator<ValidGenerator, 1>);
static_assert(evolution::dg::equal_rate_region_generator<ValidGenerator, 2>);
static_assert(evolution::dg::equal_rate_region_generator<ValidGenerator, 3>);

class ValidNoOptGenerator {
 public:
  explicit ValidNoOptGenerator();

  using creation_tags = tmpl::list<>;

  std::unordered_map<std::string, size_t> regions() const;

  template <size_t Dim>
  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p);
};
static_assert(
    evolution::dg::equal_rate_region_generator<ValidNoOptGenerator, 1>);
static_assert(
    evolution::dg::equal_rate_region_generator<ValidNoOptGenerator, 2>);
static_assert(
    evolution::dg::equal_rate_region_generator<ValidNoOptGenerator, 3>);

// Won't test every single clause since it's rather verbose, but check
// a few ways that generators could be invalid.

class BadGeneratorPup {
 public:
  explicit BadGeneratorPup();

  using creation_tags = tmpl::list<OptInt, OptString>;

  explicit BadGeneratorPup(int int_opt, const std::string& string_opt);

  std::unordered_map<std::string, size_t> regions() const;

  template <size_t Dim>
  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;
};
static_assert(
    not evolution::dg::equal_rate_region_generator<BadGeneratorPup, 3>);

class BadGeneratorArguments {
 public:
  explicit BadGeneratorArguments();

  using creation_tags = tmpl::list<OptInt, OptString>;

  explicit BadGeneratorArguments(int int_opt);

  std::unordered_map<std::string, size_t> regions() const;

  template <size_t Dim>
  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p);
};
static_assert(
    not evolution::dg::equal_rate_region_generator<BadGeneratorArguments, 3>);

class BadGeneratorNoConst {
 public:
  explicit BadGeneratorNoConst();

  using creation_tags = tmpl::list<OptInt, OptString>;

  explicit BadGeneratorNoConst(int int_opt, const std::string& string_opt);

  std::unordered_map<std::string, size_t> regions();

  template <size_t Dim>
  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p);
};
static_assert(
    not evolution::dg::equal_rate_region_generator<BadGeneratorNoConst, 3>);

class BadGeneratorMissingMethod {
 public:
  explicit BadGeneratorMissingMethod();

  using creation_tags = tmpl::list<OptInt, OptString>;

  explicit BadGeneratorMissingMethod(int int_opt,
                                     const std::string& string_opt);

  template <size_t Dim>
  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p);
};
static_assert(not evolution::dg::equal_rate_region_generator<
              BadGeneratorMissingMethod, 3>);
}  // namespace
