#include "headers/application.hpp"
#include <math.h>

Application::Application(NeuralNetwork::Ptr network, Batch teachingBatch, Batch testingBatch)
: mNetwork(network)
, mTeacher(mNetwork)
, mTeachingBatch(teachingBatch)
, mTestingBatch(testingBatch)
, mStatsCollector()
, mTestCounter(0)
{}

Application::Application(   NeuralNetwork::Ptr network,
                            std::function<Eigen::VectorXf (Eigen::VectorXf)> modelFunction,
                            std::vector<Eigen::VectorXf> teachingInputs,
                            std::vector<Eigen::VectorXf> testingInputs)
: mNetwork(network)
, mTeacher(mNetwork)
, mStatsCollector()
, mTestCounter(0)
{
    // Génère le batch d'apprentissage à partir des entrées et de la fonction à modéliser
    for(size_t i{0}; i < teachingInputs.size(); ++i)
        mTeachingBatch.push_back(Sample(teachingInputs[i], modelFunction(teachingInputs[i])));
    // Génère le batch d'apprentissage à partir des entrées et de la fonction à modéliser
    for(size_t i{0}; i < testingInputs.size(); ++i)
        mTestingBatch.push_back(Sample(testingInputs[i], modelFunction(testingInputs[i])));
}

void Application::runExperiments(unsigned int nbExperiments, unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    for(unsigned int index{0}; index < nbExperiments; ++index)
    {
        runSingleExperiment(nbLoops, nbTeachingsPerLoop);
        std::cout << "Exp num. " << (index+1) << " finie !" << std::endl;
        resetExperiment();
    }

    mStatsCollector.exportData(true);
}

void Application::runSingleExperiment(unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    for(unsigned int loopIndex{0}; loopIndex < nbLoops; ++loopIndex)
    {
        int timeref = time(0);
        runTeach(nbTeachingsPerLoop);
        mStatsCollector[loopIndex].addResult(runTest());
        std::cout << "Apprentissage num. : " << (loopIndex+1)*nbTeachingsPerLoop << std::endl;
        std::cout << "Durée : " << time(0) - timeref << "s" << std::endl;
        if (loopIndex==0)
        {
            int temps_total = (time(0)-timeref)*nbLoops;
            std::cout << "Durée de l'Expérience : " << temps_total << "s ou " << temps_total/3600 << "h " << (temps_total%3600)/60 << "min " << temps_total%60 << "s" << std::endl;
        }
    }
}

void Application::resetExperiment()
{
    mNetwork->reset();
}

void Application::runTeach(unsigned int nbTeachings)
{
    std::uniform_int_distribution<> distribution(0, mTeachingBatch.size()-1);
    std::mt19937 randomEngine((std::random_device())());

    for(unsigned int index{0}; index < nbTeachings; index++)
    {
        auto sample{mTeachingBatch[distribution(randomEngine)]};
        mTeacher.backProp(sample.first, sample.second);
    }
}

float Application::runTest(int limit)
{
    float errorMean{0};

    for(auto itr = mTestingBatch.begin(); itr != mTestingBatch.end() && limit-- != 0; ++itr)
    {
        auto output{mNetwork->process(itr->first)};
        errorMean += sqrt((output - itr->second).squaredNorm());
    }

    return errorMean/static_cast<float>(mTestingBatch.size());
}
