#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>

#include "headers/application.hpp"
#include "headers/mnist_reader.h"

using namespace std;

int main()
{   
    srand(time(0));

    try
    {
        // Construction du réseau de neurones
        std::vector<unsigned int> sizes{ {2,2,2,1} };
        std::vector<Functions::ActivationFun> funs{ {Functions::sigmoid(3.f), Functions::sigmoid(3.f), Functions::sigmoid(3.f)} };
        std::shared_ptr<NeuralNetwork> network(new NeuralNetwork(sizes, funs));

        mnist_reader reader("MNIST/test-images-10k", "MNIST/test-labels-10k");

        std::vector<Eigen::VectorXf> v;
        Eigen::VectorXi i;
        reader.ReadMNIST(v, i);

        //Construction de l'application qui gère tout
        //Application appXOR(network, teachBatch, testBatch);

        //appXOR.runExperiments(300, 100, 1000);
    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
    return 0;
}
