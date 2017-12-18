#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <ctime>

#include "headers/application.hpp"
#include "headers/mnist_reader.h"

using namespace std;

int main()
{   
    srand(time(0));

    try
    {
        mnist_reader readerTrain("MNIST/train-images-60k", "MNIST/train-labels-60k");
        std::vector<Eigen::MatrixXf> imageTrain;
        Eigen::MatrixXi labelTrain;
        readerTrain.ReadMNIST(imageTrain, labelTrain);

        mnist_reader readerTest("MNIST/test-images-10k", "MNIST/test-labels-10k");
        std::vector<Eigen::MatrixXf> imageTest;
        Eigen::MatrixXi labelTest;
        readerTest.ReadMNIST(imageTest, labelTest);

        Application::Batch batchTrain;
        Application::Batch batchTest;

        for(auto i(0); i< labelTrain.size(); i++)
        {
            Eigen::MatrixXf outputTrain = Eigen::MatrixXf::Zero(10,1);
            outputTrain(labelTrain(i)) = 1;
            batchTrain.push_back(Application::Sample(imageTrain[i], outputTrain));
        }
        cout << "Chargement du Batch d'entrainement effectué !" << endl;
        for(auto i(0); i< labelTest.size(); i++)
        {
            Eigen::MatrixXf outputTest = Eigen::MatrixXf::Zero(10,1);
            outputTest(labelTest(i)) = 1;
            batchTest.push_back(Application::Sample(imageTest[i], outputTest));
        }
        cout << "Chargement du Batch de test effectué !" << endl;

        //Construction de l'application qui gère tout
        Application appMNIST(batchTrain, batchTest);
        appMNIST.runExperiments();
    }
    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
    return 0;
}
