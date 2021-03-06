#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include <eigen3/Eigen/Dense>
#include <vector>
#include <random>

#include <headers/rapidjson/document.h>
#include "headers/neuralnetwork.hpp"
#include "headers/teacher.hpp"
#include "headers/statscollector.hpp"

///Classe destinée à gérer l'ensemble d'un projet
/**
 * La classe supervise l'apprentissage d'un réseau de neurones par rapport au batchs de données qu'on lui fournit
 * et sort les résultats dans un fichier csv
 */
class Application
{
    public:
        struct Config
        {
            float step;
            float dx;

            unsigned int nbExperiments;
            unsigned int nbLoopsPerExperiment;
            unsigned int nbTeachingsPerLoop;

            std::vector<unsigned int> neuronLayerSizes;
        };

    public:
        /// Un alias pour désigner un donnée (Entrée, Sortie)
        using Sample = std::pair<Eigen::MatrixXf, Eigen::MatrixXf>;
        /// Un alias pour désigner un batch de données (Entrée, Sortie)
        using Batch = std::vector<Sample>;

    public:
        /// Constructeur par batchs
        /**
         * Ce constructeur supervise le projet par rapport au réseau de neurones donné et aux batchs de tests et d'apprentissages donnés en paramètre
         * @param network le réseau avec lequel on travaille
         * @param teachingBatch le batch des données servant à l'apprentissage
         * @param testingBatch le batch des données de test
         */
        Application(Batch teachingBatch, Batch testingBatch, const std::string& configFileName = "config.json");


        void runExperiments();
        void runSingleExperiment();

        /// Effectue une run de tests
        /**
         * Effectue une run de test dont le nombre de tests est passé en paramètres
         * @param nbTests le nombre de tests à faire pendant la run
         */
        float runTest(bool returnErrorRate = true);

        /// Effectue une run d'apprentissage
        /**
         * Effectue une run d'apprentissage dont le nombre d'apprentissages est passé en paramètres
         * @param nbTeachings le nombre d'apprentissages à faire pendant la run
         */
        void runTeach();

        void resetExperiment();

    private:
        /// Fonction pour charger la configuration de l'application
        Config  loadConfig(const std::string& configFileName);
        Config  getConfig(rapidjson::Document& document);
        void    displayConfig();

    private:
        /// Configuration de l'application
        Config              mConfig;

        /// Le réseau avec lequel on travaille
        NeuralNetwork::Ptr  mNetwork;
        /// Le teacher qui permet de superviser l'apprentissage du réseau
        Teacher             mTeacher;

        /// Le batch contenant tous les samples d'apprentissage du projet
        Batch               mTeachingBatch;
        /// Le batch contenant tous les samples de test du projet
        Batch               mTestingBatch;

        Stats::StatsCollector mStatsCollector;
        /// Un compteur permettant d'indicer les données exportées
        unsigned int        mTestCounter;
};

#endif // APPLICATION_HPP
