#include "headers/neuronlayer.hpp"

//*************CONSTRUCTEUR*************
//**************************************

NeuronLayer::NeuronLayer(unsigned int inputSize, unsigned int outputSize, std::function<float(float)> activationF)
: mPoids(Eigen::MatrixXf::Random(outputSize, inputSize))
, mBiais(Eigen::MatrixXf::Random(outputSize, 1))
, mActivationFun(activationF)
, mBufferActivationLevel(Eigen::MatrixXf::Zero(outputSize, 1))
, mBufferInput(Eigen::MatrixXf::Zero(inputSize, 1))
{}


//*************PROPAGATION**************
//**************************************

Eigen::VectorXf NeuronLayer::process(Eigen::VectorXf inputs)
{
    mBufferInput = inputs;
    mBufferActivationLevel = mPoids*inputs - mBiais;
    Eigen::VectorXf output = mBufferActivationLevel;

    for(unsigned int i(0); i < output.size(); i++)
        output[i] = mActivationFun(output[i]);

    return output;
}

//***********RETROPROPAGATION***********
//**************************************


Eigen::VectorXf NeuronLayer::backProp(Eigen::VectorXf xnPartialDerivative, float step)
{
    // Calcul de ynPartialDerivative
    Eigen::VectorXf ynPartialDerivative = fnDerivativeMatrix()*xnPartialDerivative;

    //Mise à jour des poids
    Eigen::MatrixXf wnPartialDerivative = ynPartialDerivative*(mBufferInput.transpose());
    mPoids -= step*wnPartialDerivative;

    // Mise à jour des biais
    mBiais += step*ynPartialDerivative;

    //Retour de x(n-1)PartialDerivative
    return mPoids.transpose()*ynPartialDerivative;
}

Eigen::MatrixXf NeuronLayer::fnDerivativeMatrix() const
{
    auto fnDerivated = [this] (float x, float dx)
                        {
                            return (mActivationFun(x+dx) - mActivationFun(x))/dx;
                        };

    Eigen::VectorXf fnDerivativeMat(mBufferActivationLevel.size());
    for(auto i(0); i < mBufferActivationLevel.size(); ++i)
        fnDerivativeMat(i) = fnDerivated(mBufferActivationLevel(i), 0.05);

    return Eigen::MatrixXf(fnDerivativeMat.asDiagonal());
}

void NeuronLayer::reset()
{
    mPoids = Eigen::MatrixXf::Random(mPoids.rows(), mPoids.cols());
    mBiais = Eigen::VectorXf::Random(mBiais.size());
}

//*************AUXILIAIRES**************
//**************************************

std::ostream& operator<<(std::ostream& flux, NeuronLayer n)
{
    flux << n.mPoids;
    return flux;
}
