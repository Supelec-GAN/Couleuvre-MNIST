#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <list>
#include <vector>
#include "neuronlayer.hpp"

class NeuralNetwork : public std::list<NeuronLayer>
{
    public:
        using Ptr = std::shared_ptr<NeuralNetwork>;

    public:
	
	/// Constructeur permettant d'initialiser une réseau neuronal vide
        NeuralNetwork();
	
    /// Constructeur permettant d'initialiser un réseau neuronal complet
    /**
     * \param nbLayer le nombre de couches du réseau
	 * \param nbInputs ne nombre d'entrées du réseau
     * \param arrayNbNeuronsPerLayer la liste du nombre de neurone par couche (nbLayer éléments)
     * \param arrayActivationFunctionPerLayer la liste des fonctions d'activation de chaque couche (nbLayer éléments)
	 */
        NeuralNetwork(unsigned int nbLayer, unsigned int nbInputs, std::vector<unsigned int> arrayNbNeuronsPerLayer, std::vector<std::function<float(float)>> arrayActivationFunctionPerLayer);
	
	/// Constructeur permettant d'initialiser un réseau neuronal avec la fonction par défaut
	/**
	 * \param nbLayer le nombre de couches du réseau
	 * \param nbInputs ne nombre d'entrées du réseau
	 * \param arrayNbNeuronsPerLayer la liste du nombre de neurone par couche (nbLayer éléments)
	 */
        NeuralNetwork(unsigned int nbLayer, unsigned int nbInputs, std::vector<unsigned int> arrayNbNeuronsPerLayer);

    /// Constructeur permettant d'initialiser le réseau neuronal avec un conteneur de neuronLayer
    /**
     *  \param Container un conteneur (vector, list...) de NeuronLayer
     *  \param layerList la liste des couches de neurones
     */
        template <typename Container>
        NeuralNetwork(Container layerList);

        Eigen::VectorXf process(Eigen::VectorXf input);


    public:
    /// Fonction utilitaire permettant d'afficher le réseau de neurones
    /**
     * Cette fonction affiche les matrices de poids des différents layers du réseau
     */
    friend std::ostream& operator<<(std::ostream& flux, NeuralNetwork network);

};

#include "headers/neuralnetwork.inl"

#endif // NEURALNETWORK_HPP
