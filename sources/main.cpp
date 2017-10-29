#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>

#include "headers/neuronlayer.hpp"
#include "headers/neuralnetwork.hpp"
#include "headers/functions.hpp"
#include "headers/teacher.hpp"
#include "headers/mnist_reader.h"
#include "headers/CSVFile.h"


using namespace std;

int main()
{
   srand(192786327);

   mnist_reader mnist("/home/manu/Documents/MNIST/train-images.idx3-ubyte","/home/manu/Documents/MNIST/train-labels.idx1-ubyte"); //Il faut mettre les chemins absolus pour ouvrir les fichiers
   vector<Eigen::VectorXf> image;
   Eigen::VectorXi label;
   mnist.ReadMNIST(image, label);
   /*for(auto i(0); i< 728; i++)
   {
       if (i%28 == 0)
           cout << endl;
       if (image[1][i] > 0.01)
           cout << 1;
       else cout << 0;
   }*/



   try 
   {
   csvfile csv("resultat.csv"); 
    
   NeuronLayer inputLayer(784,2000, Functions::sigmoid(1.f));
   NeuronLayer innerLayer1(2000,1000, Functions::sigmoid(1.0f));
   NeuronLayer innerLayer2(1000,500, Functions::sigmoid(1.0f));
   NeuronLayer outputLayer(500,10, Functions::sigmoid(0.1f));

   NeuralNetwork::Ptr network(new NeuralNetwork(std::vector<NeuronLayer>({{inputLayer, innerLayer1, innerLayer2, outputLayer}})));

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
       if (label[i]== 4){
       target[label[i]] = 1;

       if (i%2 == 0)
       {
       std::cout << "Input no : " << i << "\n";
       //std::cout << "Entrée : " << input.transpose() << "\n";
       std::cout << "Sortie attendue : " << label[i] << "\n";
       //std::cout << "Poids : \n" << *network << "\n";

       std::cout << "Sortie : " << endl << network->process(input) << std::endl;

       std::cout << "Erreur : " << endl << Functions::l2Norm()(network->process(input),target) << std::endl;

       } }

       teacher.backProp(input, target, 0.2);
   
   }
   }
   catch (const std::exception& ex) 
   {
       std::cout << "Exception was thrown: " << ex.what() << std::endl;
   }
   return 0;
}
