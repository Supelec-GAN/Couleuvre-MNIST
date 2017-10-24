#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>

#include "headers/neuronlayer.hpp"
#include "headers/neuralnetwork.hpp"
#include "headers/functions.hpp"
#include "headers/teacher.hpp"
#include "headers/mnist_reader.h"
#include "CSVFile.h"


using namespace std;

int main()
{
   srand(192786327);

   mnist_reader mnist("/home/manu/Documents/MNIST/t10k-images.idx3-ubyte","/home/manu/Documents/MNIST/t10k-labels.idx1-ubyte");
   vector<Eigen::VectorXf> image;
   Eigen::VectorXi label;
   mnist.ReadMNIST(image, label);
   /*for(auto i(0); i< 728; i++)
   {
       if (i%28 == 0)
           cout << endl;
       if (image[2][i] > 80)
           cout << 1;
       else cout << 0;
   }*/



   try 
   {
   csvfile csv("resultat.csv"); 
    
   NeuronLayer inputLayer(784,3000, Functions::sigmoid(4.f));
   NeuronLayer innerLayer1(3000,1500, Functions::sigmoid(4.f));
   NeuronLayer innerLayer6(1500,1000, Functions::sigmoid(4.f));
   NeuronLayer innerLayer7(1000,500, Functions::sigmoid(4.f));
   NeuronLayer outputLayer(500,10, Functions::sigmoid(4.f));

   NeuralNetwork::Ptr network(new NeuralNetwork(std::vector<NeuronLayer>({{inputLayer, innerLayer1, innerLayer6, innerLayer7, outputLayer}})));

   Teacher teacher(network);

   //std::default_random_engine generator;
   //std::uniform_real_distribution<float> distribution(-1.f,1.f);

   //Eigen::VectorXf input(2);
   //Eigen::VectorXf desiredOutput(1);

   csv << "nbApprentissage" << "erreur" << "nbReussite" << endrow;
   for(size_t i(0); i < 10000; i++)
   {
       auto input = image[i];
       Eigen::VectorXf target= Eigen::MatrixXf::Zero(10, 1);
       target[label[i]] = 1;
       std::cout << "Input no : " << i << "\n";
       //std::cout << "Entrée : " << input.transpose() << "\n";
       std::cout << "Sortie attendue : " << label[i] << "\n";
       //std::cout << "Poids : \n" << *network << "\n";
       std::cout << "Sortie : " << endl << network->process(input) << std::endl;
       teacher.backProp(input, target, 0.1);
   
   }
   }
   catch (const std::exception& ex) 
   {
       std::cout << "Exception was thrown: " << ex.what() << std::endl;
   }
   return 0;
}
