#ifndef SIGNALHANDLER_HPP
#define SIGNALHANDLER_HPP

#include "application.hpp"

#include <memory>
#include <iostream>
#include <csignal>

///La classe qui dit aux objets importants de lancer les procédures d'abort
struct SignalApplier
{
    ~SignalApplier()
    {
        std::cout << "SignalApplier deleted" << std::endl;
    }

    void handle(int signum)
    {
        std::cout << "Interruption Signal Caught : " << signum << "\nCurrently being handled..." << std::endl;
        ptr->handleInterrupt();
        std::cout << "Interruption signal handled" << std::endl;
    }

    std::shared_ptr<Application> ptr;
};

/// La classe qui catch l'interruption et la redirige sur SignalApplier
struct SignalHandler
{

    static void handle(int signum)
    {
        ptr->handle(signum);
        exit(signum);
    }

    static SignalApplier* ptr;  // Non c'est pas un pointer nu c'est pas vrai. Je nierai avoir eu connaissance de cette portion du code
};

SignalApplier* SignalHandler::ptr;


#endif // SIGNALHANDLER_HPP
