#include "headers/functions.hpp"

#include <cmath>

Functions::ActivationFun Functions::sigmoid(float lambda)
{
    return [=] (float x) {return 1.f/(1.f + exp(-lambda*x));};
}

Functions::ActivationFun Functions::heavyside(float gapAbscissa)
{
    return [=] (float x) {return (x < gapAbscissa) ? 0 : 1;};
}

Functions::ActivationFun Functions::hyperTan()
{
    return [=] (float x) {return tanh(2*x/3);};
}

Functions::ErrorFun Functions::l2Norm()
{
    return [] (Eigen::VectorXf v1, Eigen::VectorXf v2) {return (v1-v2).squaredNorm();};
}
