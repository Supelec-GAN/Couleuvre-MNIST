#include "headers/neuralnetwork.hpp"

NeuralNetwork::NeuralNetwork(){}


NeuralNetwork::NeuralNetwork(unsigned int nbLayer, unsigned int nbInputs, std::vector<unsigned int> arrayNbNeuronsPerLayer, std::vector<std::function<float(float)>> arrayActivationFunctionPerLayer)
{
    push_back(NeuronLayer(  nbInputs,
                            arrayNbNeuronsPerLayer[0],
                            arrayActivationFunctionPerLayer[0]
                            )) ;
	
    for(unsigned int i=0; i < nbLayer - 1; ++i)
	{
        push_back(NeuronLayer(      arrayNbNeuronsPerLayer[i],
									arrayNbNeuronsPerLayer[i+1],
									arrayActivationFunctionPerLayer[i+1]
									));
	}
}

NeuralNetwork::NeuralNetwork(unsigned int nbLayer, unsigned int nbInputs, std::vector<unsigned int> arrayNbNeuronsPerLayer)
{
    push_back(NeuronLayer(  nbInputs,
                            arrayNbNeuronsPerLayer[0]
                            )) ;
	
	for(unsigned int i=0; i < nbLayer - 1; ++i)
	{
        push_back(NeuronLayer(  arrayNbNeuronsPerLayer[i],
                                arrayNbNeuronsPerLayer[i+1]
                                ));
	}
}


Eigen::VectorXf NeuralNetwork::process(Eigen::VectorXf input)
{
	for(auto itr = begin(); itr != end(); ++itr)
		input = (*itr).process(input);
	
	return input;
}


//*************AUXILIAIRES**************
//**************************************

std::ostream& operator<<(std::ostream& flux, NeuralNetwork network)
{
    NeuralNetwork::iterator it;
    for (it = network.begin(); it != network.end(); it++)
    {
        flux << *it << "\n" << std::endl;
    }
    return flux;
}

