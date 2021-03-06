# AI_Helen

###### About the project

AI-Feynman is a physics-inspired method for symbolic regression. This method corresponds to finding an equation for a set of unknown data.

The AI-Feynman code method uses brute force to look up equations. Our project implemented the use of genetic algorithms and variable reduction to find the equation.

These implementations made the process of finding the equation faster.

###### Implementation

The dataset used was *'example1.txt'* from Feynman's library.

The data have a discrete distribution and to verify symmetries we need a continuous dataset. Therefore, we create and train a neural network so that it returns a continuous function to check for existing symmetries.

The original dataset has 4 columns of independent variables. With the check_translational_symmetry_plus def it was verified that the columns (0,1) and (2,3) have symmetries. Then, we subtracted these symmetric columns with the assignment of a new variable.

Subsequently, we implement the genetic algorithm on our reduced dataset to optimize the search time. See recommendations and how to use our code below.

###### Execution

To use the pysr_utils, check first if there is a GPU available for code execution, if there is, select it. With the GPU, the execution will be more efficient.

Then, download *'example1.txt'*, this file is available on Silviu Marian Udrescu's GitHub. It's necessary that the data file be a numpy array or csv. But don't worry, has a function to verify this.

## Attention 

To execute the **def SR_GA** function it is necessary to have JULIA installed on the computer, as we use the genetic programming Pysr package that has the JULIA interface.

###### Results

In **def show_results** we go back to our original variable sets and the original size dataset showing the correct equation founded!

Using the genetic algorithm, the search time for the equation was 8 times faster. Use the code and check!

For more information, please contact us.

###### Citation
@article{udrescu2020ai,
  title={AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity},
  author={Udrescu, Silviu-Marian and Tan, Andrew and Feng, Jiahai and Neto, Orisvaldo and Wu, Tailin and Tegmark, Max},
  journal={arXiv preprint arXiv:2006.10782},
  year={2020}
}
