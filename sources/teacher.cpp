#include <eigen3/Eigen/Dense>

#include "headers/teacher.hpp"


Teacher::Teacher(NeuralNetwork::Ptr network)
: mNetwork(std::move(network))
, mErrorFun(Functions::l2Norm())
{}

Teacher::Teacher(NeuralNetwork* network)
: mNetwork(network)
, mErrorFun(Functions::l2Norm())
{}

void Teacher::backProp(Eigen::MatrixXf input, Eigen::MatrixXf desiredOutput, float step, float dx)
{
    Eigen::MatrixXf xnPartialDerivative = errorVector(mNetwork->process(input), desiredOutput, dx);

    propError(xnPartialDerivative, step);
}

void Teacher::propError(Eigen::MatrixXf xnPartialDerivative, float step)
{
    for(auto itr = mNetwork->rbegin(); itr != mNetwork->rend(); ++itr)
    {
        xnPartialDerivative = itr->backProp(xnPartialDerivative, step);
    }
}

Eigen::MatrixXf Teacher::errorVector(Eigen::MatrixXf output, Eigen::MatrixXf desiredOutput, float dx)
{
    Eigen::MatrixXf errorVect = Eigen::MatrixXf::Zero(output.size(), 1);

    for(unsigned int i(0); i < output.size(); ++i)
    {
        Eigen::MatrixXf deltaX(Eigen::MatrixXf::Zero(output.size(), 1));
        deltaX(i) = dx;
        errorVect(i) = (mErrorFun(output + deltaX, desiredOutput) - mErrorFun(output, desiredOutput))/dx;
    }


    return errorVect;
}
