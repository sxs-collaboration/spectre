// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
namespace Tags {
struct MyDouble : db::SimpleTag {
  using type = double;
};
struct MyScalar : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct MyDouble2 : db::SimpleTag {
  using type = double;
};
}  // namespace Tags

template <size_t NumberOfArgs, size_t NumberOfReturnTags>
struct Function {
  using return_tags = tmpl::conditional_t<
      NumberOfReturnTags == 0, tmpl::list<>,
      tmpl::conditional_t<NumberOfReturnTags == 1, tmpl::list<Tags::MyDouble2>,
                          tmpl::list<Tags::MyDouble2, Tags::MyDouble>>>;
  using argument_tags = tmpl::conditional_t<
      NumberOfArgs == 0, tmpl::list<>,
      tmpl::conditional_t<NumberOfArgs == 1, tmpl::list<Tags::MyDouble>,
                          tmpl::list<Tags::MyDouble, Tags::MyScalar>>>;
  static_assert(NumberOfArgs < 3);
  static_assert(NumberOfReturnTags < 3);

  static double apply() { return 3.1; }

  double operator()() { return 3.3; }

  static double apply(const gsl::not_null<double*> result) {
    *result = 4.1;
    return 5.1;
  }

  double operator()(const gsl::not_null<double*> result) {
    *result = 4.3;
    return 5.3;
  }

  static double apply(const gsl::not_null<double*> result,
                      const gsl::not_null<double*> result2) {
    *result = 4.2;
    *result2 = 2 * *result;
    return 5.2;
  }

  double operator()(const gsl::not_null<double*> result,
                    const gsl::not_null<double*> result2) {
    *result = 4.4;
    *result2 = 2 * *result;
    return 5.4;
  }

  static double apply(const gsl::not_null<double*> result, const double input) {
    *result = input;
    return 2.0 * input;
  }

  double operator()(const gsl::not_null<double*> result, const double input) {
    *result = 3.0 * input;
    return input;
  }

  static double apply(const double input) { return 2.0 * input; }

  double operator()(const double input) { return input; }

  static double apply(const gsl::not_null<double*> result,
                      const gsl::not_null<double*> result2,
                      const double input) {
    *result = input;
    *result2 = 2 * *result;
    return 2.0 * input;
  }

  double operator()(const gsl::not_null<double*> result,
                    const gsl::not_null<double*> result2, const double input) {
    *result = 3.0 * input;
    *result2 = 2 * *result;
    return input;
  }

  static double apply(const double input, const Scalar<DataVector>& my_scalar) {
    CHECK(get(my_scalar) == 3.2);
    return 2.0 * input;
  }

  double operator()(const double input, const Scalar<DataVector>& my_scalar) {
    CHECK(get(my_scalar) == 3.2);
    return input;
  }

  static double apply(const gsl::not_null<double*> result, const double input,
                      const Scalar<DataVector>& my_scalar) {
    CHECK(get(my_scalar) == 3.2);
    *result = input;
    return 2.0 * input;
  }

  double operator()(const gsl::not_null<double*> result, const double input,
                    const Scalar<DataVector>& my_scalar) {
    CHECK(get(my_scalar) == 3.2);
    *result = 3.0 * input;
    return input;
  }

  static double apply(const gsl::not_null<double*> result,
                      const gsl::not_null<double*> result2, const double input,
                      const Scalar<DataVector>& my_scalar) {
    CHECK(get(my_scalar) == 3.2);
    *result = input;
    *result2 = 2 * *result;
    return 2.0 * input;
  }

  double operator()(const gsl::not_null<double*> result,
                    const gsl::not_null<double*> result2, const double input,
                    const Scalar<DataVector>& my_scalar) {
    CHECK(get(my_scalar) == 3.2);
    *result = 3.0 * input;
    *result2 = 2 * *result;
    return input;
  }
};

// Test that desired tags are retrieved.
void test_tagged_containers() {
  const auto box = db::create<db::AddSimpleTags<Tags::MyDouble>>(1.2);
  tuples::TaggedTuple<Tags::MyDouble2> tuple(4.4);
  Variables<tmpl::list<Tags::MyScalar>> vars(10ul, 3.2);

  CHECK(get<Tags::MyDouble>(box) == 1.2);
  CHECK(get<Tags::MyDouble2>(tuple) == 4.4);
  CHECK(get<Tags::MyScalar>(vars).get() == 3.2);

  // Note: tests do not pass if DataVector only has a single component!
  const auto box2 = db::create<db::AddSimpleTags<Tags::MyScalar>>(
      Scalar<DataVector>{10ul, 1.2});

  CHECK(get<Tags::MyScalar>(vars, box2).get() == 3.2);
  CHECK(get<Tags::MyScalar>(box2, vars).get() == 1.2);

  CHECK(get<Tags::MyScalar>(tuple, box2, vars).get() == 1.2);
  CHECK(get<Tags::MyScalar>(box2, tuple, vars).get() == 1.2);
  CHECK(get<Tags::MyScalar>(box2, vars, tuple).get() == 1.2);

  CHECK(get<Tags::MyDouble>(box, tuple) == 1.2);
  CHECK(get<Tags::MyDouble2>(tuple, vars) == 4.4);
  CHECK(get<Tags::MyScalar>(vars, box).get() == 3.2);

  CHECK(get<Tags::MyDouble>(tuple, box) == 1.2);
  CHECK(get<Tags::MyDouble2>(vars, tuple) == 4.4);
  CHECK(get<Tags::MyScalar>(box, vars).get() == 3.2);

  CHECK(get<Tags::MyDouble>(box, tuple, vars) == 1.2);
  CHECK(get<Tags::MyDouble2>(tuple, vars, box) == 4.4);
  CHECK(get<Tags::MyScalar>(vars, box, tuple).get() == 3.2);

  CHECK(get<Tags::MyDouble>(box, vars, tuple) == 1.2);
  CHECK(get<Tags::MyDouble2>(tuple, box, vars) == 4.4);
  CHECK(get<Tags::MyScalar>(vars, tuple, box).get() == 3.2);

  INFO("Test invoke and apply");
  tuples::TaggedTuple<Tags::MyDouble, Tags::MyDouble2> tuple2(0.0, 0.0);

  get<Tags::MyDouble2>(tuple) = 1.0;
  CHECK(apply(make_not_null(&tuple), Function<0, 0>{}, box, vars) == 3.1);
  CHECK(get<Tags::MyDouble2>(tuple) == 1.0);
  get<Tags::MyDouble2>(tuple) = 0.0;
  CHECK(invoke(make_not_null(&tuple), Function<0, 0>{}, box, vars) == 3.3);
  CHECK(get<Tags::MyDouble2>(tuple) == 0.0);

  CHECK(apply(make_not_null(&tuple), Function<0, 1>{}, box, vars) == 5.1);
  CHECK(get<Tags::MyDouble2>(tuple) == 4.1);
  get<Tags::MyDouble2>(tuple) = 0.0;
  CHECK(invoke(make_not_null(&tuple), Function<0, 1>{}, box, vars) == 5.3);
  CHECK(get<Tags::MyDouble2>(tuple) == 4.3);
  get<Tags::MyDouble2>(tuple) = 1.0;

  CHECK(apply(make_not_null(&tuple2), Function<0, 2>{}, box, vars) == 5.2);
  CHECK(get<Tags::MyDouble2>(tuple2) == 4.2);
  CHECK(get<Tags::MyDouble>(tuple2) == 8.4);
  get<Tags::MyDouble>(tuple2) = get<Tags::MyDouble2>(tuple2) = 0.0;
  CHECK(invoke(make_not_null(&tuple2), Function<0, 2>{}, box, vars) == 5.4);
  CHECK(get<Tags::MyDouble2>(tuple2) == 4.4);
  CHECK(get<Tags::MyDouble>(tuple2) == 8.8);
  get<Tags::MyDouble>(tuple2) = get<Tags::MyDouble2>(tuple2) = 0.0;

  get<Tags::MyDouble2>(tuple) = 1.0;
  CHECK(apply(make_not_null(&tuple), Function<1, 0>{}, box, vars) ==
        2.0 * get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple) == 1.0);
  get<Tags::MyDouble2>(tuple) = 2.0;
  CHECK(invoke(make_not_null(&tuple), Function<1, 0>{}, box, vars) ==
        get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple) == 2.0);
  get<Tags::MyDouble2>(tuple) = 0.0;

  CHECK(apply(make_not_null(&tuple), Function<1, 1>{}, box, vars) ==
        2.0 * get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple) == get<Tags::MyDouble>(box));
  get<Tags::MyDouble2>(tuple) = 0.0;
  CHECK(invoke(make_not_null(&tuple), Function<1, 1>{}, box, vars) ==
        get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple) == 3.0 * get<Tags::MyDouble>(box));
  get<Tags::MyDouble2>(tuple) = 0.0;

  CHECK(apply(make_not_null(&tuple2), Function<1, 2>{}, box, vars) ==
        2.0 * get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple2) == get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble>(tuple2) == 2.0 * get<Tags::MyDouble2>(tuple2));
  get<Tags::MyDouble>(tuple2) = get<Tags::MyDouble2>(tuple2) = 0.0;
  CHECK(invoke(make_not_null(&tuple2), Function<1, 2>{}, box, vars) ==
        get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple2) == 3.0 * get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble>(tuple2) == 2.0 * get<Tags::MyDouble2>(tuple2));
  get<Tags::MyDouble>(tuple2) = get<Tags::MyDouble2>(tuple2) = 0.0;

  CHECK(apply(make_not_null(&tuple), Function<2, 0>{}, box, vars) ==
        2.0 * get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple) == 0.0);
  get<Tags::MyDouble2>(tuple) = 0.0;
  CHECK(invoke(make_not_null(&tuple), Function<2, 0>{}, box, vars) ==
        get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple) == 0.0);
  get<Tags::MyDouble2>(tuple) = 0.0;

  CHECK(apply(make_not_null(&tuple), Function<2, 1>{}, box, vars) ==
        2.0 * get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple) == get<Tags::MyDouble>(box));
  get<Tags::MyDouble2>(tuple) = 0.0;
  CHECK(invoke(make_not_null(&tuple), Function<2, 1>{}, box, vars) ==
        get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple) == 3.0 * get<Tags::MyDouble>(box));
  get<Tags::MyDouble2>(tuple) = 0.0;

  CHECK(apply(make_not_null(&tuple2), Function<2, 2>{}, box, vars) ==
        2.0 * get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple2) == get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble>(tuple2) == 2.0 * get<Tags::MyDouble2>(tuple2));
  get<Tags::MyDouble>(tuple2) = get<Tags::MyDouble2>(tuple2) = 0.0;
  CHECK(invoke(make_not_null(&tuple2), Function<2, 2>{}, box, vars) ==
        get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble2>(tuple2) == 3.0 * get<Tags::MyDouble>(box));
  CHECK(get<Tags::MyDouble>(tuple2) == 2.0 * get<Tags::MyDouble2>(tuple2));
  get<Tags::MyDouble>(tuple2) = get<Tags::MyDouble2>(tuple2) = 0.0;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.TaggedContainers",
                  "[Unit][DataStructures]") {
  test_tagged_containers();
}
