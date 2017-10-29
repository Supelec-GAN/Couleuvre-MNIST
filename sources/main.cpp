#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>

#include "headers/neuronlayer.hpp"
#include "headers/neuralnetwork.hpp"
#include "headers/functions.hpp"
#include "headers/teacher.hpp"
#include "headers/CSVFile.h"
#include "headers/application.hpp"

using namespace std;

int main()
{   
    try
    {
        // Construction du générateurs d'inputs
        srand(time(0));
        std::default_random_engine generator((std::random_device())());
        std::uniform_real_distribution<float> distribution(-1.f,1.f);

        // Construction du réseau de neurones
        std::vector<unsigned int> sizes{ {2,2,2,1} };
        std::vector<Functions::ActivationFun> funs{ {Functions::sigmoid(3.f), Functions::sigmoid(3.f), Functions::sigmoid(3.f)} };
        std::shared_ptr<NeuralNetwork> network(new NeuralNetwork(sizes, funs));

        // Construction du batch d'exemples de test
        Application::Batch testBatch;
        for(size_t i{0}; i < 1000; i++)
        {
            Eigen::VectorXf input(2);
            Eigen::VectorXf output(1);
            input(0)  = distribution(generator);
            input(1)  = distribution(generator);
            output(0) = ((input(0) >= 0) ^ (input(1) >= 0)) ? 1 : 0;

            Application::Sample sample(input, output);
            testBatch.push_back(sample);
        }

        // Construction du batch d'exemples d'apprentissage
        Application::Batch teachBatch;
        for(size_t i{0}; i < 1000000; i++)
        {
            Eigen::VectorXf input(2);
            Eigen::VectorXf output(1);
            input(0)  = distribution(generator);
            input(1)  = distribution(generator);
            output(0) = ((input(0) >= 0) ^ (input(1) >= 0)) ? 1 : 0;


            Application::Sample sample(input, output);
            teachBatch.push_back(sample);
        }

        //Construction de l'application qui gère tout
        Application appXOR(network, teachBatch, testBatch);

        appXOR.runExperiments(300, 100, 1000);
    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
    return 0;
}
