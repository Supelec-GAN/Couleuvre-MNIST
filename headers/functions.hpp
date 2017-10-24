#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <functional>
#include <eigen3/Eigen/Dense>

struct Functions
{
        using ActivationFun = std::function<float(float)>;
        using ErrorFun      = std::function<float(Eigen::VectorXf, Eigen::VectorXf)>;

        static ActivationFun    sigmoid(float lambda);
        static ActivationFun    heavyside(float gapAbscissa);

        static ErrorFun         l2Norm();
};

#endif // FUNCTIONS_HPP
