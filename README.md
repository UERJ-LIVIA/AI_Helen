# AI_Helen

###### About the project

This project is based on the AI-Feynman code.

AI-Feynman is a physics-inspired method for symbolic regression. This method corresponds to finding an equation for a set of unknown data.

The AI-Feynman code method uses brute force to search for equations. Our project implemented the use of genetic algorithms in the search. 
The use of genetic algorithms has made the process of finding the equation faster.

###### Implementation

The dataset used was 'example. 1' from Feynman's library.

The data have a discrete distribution and to verify symmetries we need a continuous dataset. Because of this, we create and train a neural network so that it returns a continuous function in order to verify existing symmetries.

Subsequently, we implement the genetic algorithm to optimize the search. See recommendations and how to use our code below.

###### Execution

Initially, check if there is a GPU available for code execution, if there is, select it. With the GPU, the execution will be more efficient.

Then, download 'example1.txt', this file is available on Silviu Marian Udrescu's GitHub.

## Attention 

To execute the def SR_GA function it is necessary to have JULIA installed on the computer, as we use the genetic programming Pysr package that has the JULIA interface.


###### Execution
@article{udrescu2020ai,
  title={AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity},
  author={Udrescu, Silviu-Marian and Tan, Andrew and Feng, Jiahai and Neto, Orisvaldo and Wu, Tailin and Tegmark, Max},
  journal={arXiv preprint arXiv:2006.10782},
  year={2020}
}
