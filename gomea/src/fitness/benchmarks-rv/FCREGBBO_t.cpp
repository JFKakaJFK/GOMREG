#include "gomea/src/fitness/benchmarks-rv.hpp"
#include <cassert>
#include <cmath>

namespace gomea {
namespace fitness {

FCREGBBO_t::FCREGBBO_t(int expression_idx,
                       vec_t<double> ages,
                       vec_t<double> volumes,
                       double vtr) : BBOFitnessFunction_t(3, vtr)
{
  assert(0 <= expression_idx && expression_idx <= 2);
  assert(ages.size() == volumes.size());
  this->name = "FCREG (BBO)";
  this->expression_idx = expression_idx;
  this->ages = ages;
  this->volumes = volumes;
  this->initialize();
}

double
FCREGBBO_t::predict(vec_t<double>& x, double age)
{
  switch (expression_idx) {
    case 0:
      return x[2] / std::exp(x[1] * std::pow(age, x[0]));
    case 1:
      return (1.0 / x[2]) * std::exp(x[1] * std::exp(age / x[0]));
    case 2:
      return x[2] / std::exp(x[1] * std::exp(age * age / x[0]));
    default:
      std::exit(1);
  }
}

double
FCREGBBO_t::objectiveFunction(int objective_index, vec_t<double>& variables) 
{
  double mse = 0.0;
  for (int i = 0; i < ages.size(); i++) {
    mse += std::pow(volumes[i] - predict(variables, ages[i]), 2);
  }
  mse /= static_cast<double>(ages.size());
  return std::isfinite(mse) ? mse : 1e308;
}

double
FCREGBBO_t::constraintFunction(vec_t<double>& variables) 
{
  const int N = 1; // 10;
  double vs[N+1];
  double sum_of_decreases = 0.0;
  for (int i = 0; i <= N; i++) {
    vs[i] = predict(variables,
                    100.0 / static_cast<double>(N) * static_cast<double>(i));
    // if (i > 0) {
    //   sum_of_decreases += std::max(0.0, vs[i - 1] - vs[i]);
    // }
  }

  // double v0_deviation = std::max(0.0, std::abs(vs[0] - 0.005) - 0.005);
  double v0_deviation = std::max(0.0, vs[0] - 0.001);
  double v100_deviation = std::max(0.0, vs[N] - 1500.0);

  double cv = v0_deviation + v100_deviation + sum_of_decreases;
  return std::isfinite(cv) ? cv : 1e308;
}

}
}