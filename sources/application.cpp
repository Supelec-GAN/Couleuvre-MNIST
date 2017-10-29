#include "headers/application.hpp"
#include "headers/dataset.hpp"
#include <math.h>

Application::Application(NeuralNetwork::Ptr network, Batch teachingBatch, Batch testingBatch)
: mNetwork(network)
, mTeacher(mNetwork)
, mTeachingBatch(teachingBatch)
, mTestingBatch(testingBatch)
, mDataCollector()
, mStatsCollector()
, mTestCounter(0)
{}

Application::Application(   NeuralNetwork::Ptr network,
                            std::function<Eigen::VectorXf (Eigen::VectorXf)> modelFunction,
                            std::vector<Eigen::VectorXf> teachingInputs,
                            std::vector<Eigen::VectorXf> testingInputs)
: mNetwork(network)
, mTeacher(mNetwork)
, mDataCollector()
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
        resetExperiment();
    }

    mStatsCollector.exportData(true);
}

void Application::runSingleExperiment(unsigned int nbLoops, unsigned int nbTeachingsPerLoop)
{
    for(unsigned int loopIndex{0}; loopIndex < nbLoops; ++loopIndex)
    {
        runTeach(nbTeachingsPerLoop);
        mStatsCollector[loopIndex].addResult(runTest());
    }
}

void Application::resetExperiment()
{
    mNetwork->reset();
}

void Application::runTeach(unsigned int nbTeachings)
{
    auto samples(generateBatch(nbTeachings));

    for(auto itr = samples.begin(); itr != samples.end(); ++itr)
        mTeacher.backProp(itr->first, itr->second);
}

float Application::runTest()
{
    float errorMean{0};

    for(auto itr = mTestingBatch.begin(); itr != mTestingBatch.end(); ++itr)
    {
        auto output{mNetwork->process(itr->first)};
        errorMean += sqrt((output - itr->second).squaredNorm());
    }

    return errorMean/static_cast<float>(mTestingBatch.size());
}

Application::Batch Application::generateBatch(unsigned int batchSize) const
{
    std::uniform_int_distribution<> distribution(0, batchSize-1);
    std::mt19937 randomEngine((std::random_device())());

    Batch batch;

    for(unsigned int i{0}; i < batchSize; ++i)
        batch.push_back(mTeachingBatch[distribution(randomEngine)]);

    return batch;
}
