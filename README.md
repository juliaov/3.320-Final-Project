{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "5f60bbc9",
   "metadata": {},
   "source": [
    "# Accelerating metadynamics by adversarial attacks on biasing potential"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "952dfb70",
   "metadata": {},
   "source": [
    "### Code:\n",
    "CV_uncertainty.ipynb contains the code for enchanced metadynamics method applied to 1 dimensional potential well.\n",
    "\n",
    "2D.ipynb contains the code for enchanced metadynamics method applied to 2 dimensional potential well.\n",
    "\n",
    "alaninedypeptide.ipynb contains the code for enchanced metadynamics method applied to the real system containing alanine dypeptide."
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "57265da7",
   "metadata": {},
   "source": [
    "### Metadynamics"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3daa6c6e",
   "metadata": {},
   "source": [
    "Metadynamics is an enhanced sampling technic that allows us to estimate free energy of the system in the more efficient way than normal molecular dynamics (MD). How does it work? Let me explain this on the toy example of a particle in a potential well. If we initialize the particle at a random point of the well and run classical MD, the particle will go to the minimum and will be stacked there for quite a long time. There is a possibility for a particle to overcome the barrier, however, it is very low, therefore it will take us a long time to go to a different minimum and sample the whole collective variable space (Fig.1). Therefore, a more advanced technic, namely, metadynamics is needed."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b555052c",
   "metadata": {},
   "source": [
    "<img src=\"metad.gif\" alt=\"Figure 1\" width=\"400\"/>\n",
    "<div align=\"center\"> Figure 1."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5bb49d2d",
   "metadata": {},
   "source": [
    "Let’s take a look at metadynamics algorithm, which is also reffered to as a computational sand. Each time the particle visits a specific point in collective variable (CV) space, we will add a biasing potential to that point, which will add an additional force and make our particle go to another point in CV space. The bigger the time a particle will spend at one point the higher the barrier gets, which you can see from the formula below. After some time, the particle will explore the whole CV space and the sum of biasing potential together with the free energy will be constant, which allows us to estimate the free energy, using the following formula. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "198ed581",
   "metadata": {},
   "source": [
    "<img src=\"Picture3.png\" alt=\"\" width=\"500\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e8a97432",
   "metadata": {},
   "source": [
    "<img src=\"Picture2.png\" alt=\"\" width=\"400\"/>\n",
    "<div align=\"center\"> Metadynamics pseudocode."
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "070a54c1",
   "metadata": {},
   "source": [
    "### Why do we need to enhance metadynamics?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c75bbbab",
   "metadata": {},
   "source": [
    "Since the method is producing good results, the following question arises: why do we need to enhance metadynamics? First of all, even for relatively small systems, such alanine dipeptide, it will take the algorithm up to 100 ns to converge. For systems with a lot of atoms or systems like proteins it takes even longer to transverse the entire free energy landscape (𝜇𝑠). And last but not the least, in case of AIMD and NNFF it takes a lot of computational resources to run the trajectory even for few nanoseconds. Therefore, exploration of the configurational space using these computational methods is very expensive. To overcome these problems we suggest the use of adversarial attacks."
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "92f15379",
   "metadata": {},
   "source": [
    "### Adversarial attacks on neural networks"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "0dbe44e1",
   "metadata": {},
   "source": [
    "Adversarial attacks are meant to fool a neural network, be it classifier or regressor. The smallest addition of noise that will not be comprehensible in human eyes can make the neural network wrong. Adversarial attacks are mostly harmful threats to an existing ML model, but they can be used wisely as well. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c2fd8d5",
   "metadata": {},
   "source": [
    "<img src=\"Picture5.png\" alt=\"\" width=\"300\"/>\n",
    "<div align=\"center\"> Example of adversarial attack."
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "8b148977",
   "metadata": {},
   "source": [
    "### Adversarial attack to improve Neural Force fields"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "7909b0dc",
   "metadata": {},
   "source": [
    "Neural Network inter-atomic potentials are currently very reliable, being able to predict with DFT accuracy but also lot faster. However, they are highly restricted to training data region. Adversarial attack has been recently used to sample geometry that NFF is highly uncertain about. These frames may have been achieved alternatively by molecular dynamics but they take up a lot of time. These adversarial attacks can obtain highly uncertain frames quite fast and then they can be put back in training data for active learning of the ML model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "70a40921",
   "metadata": {},
   "source": [
    "<img src=\"Picture7.png\" alt=\"\" width=\"600\"/>"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "b39f663d",
   "metadata": {},
   "source": [
    "### Accelerating metadynamics"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e2ea86a",
   "metadata": {},
   "source": [
    "To combine metdynamics with adversarial attacks, first, we will initialize the three different trajectories within one ensemble, in our case NVE, and run these trajectories for some time. The resulting bias potential will be different for all three trajectories. When the bias potentials are less different from each other, it means that there is a higher possibility that they converged to the value of free energy and respective region of configurational space was explored enough. However, when they are very different from each other, which means there is high uncertainty, the opposite statement is true. Therefore our goal is to move to the place of the higher uncertainty."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b160baca",
   "metadata": {},
   "source": [
    "In order to implement that we suggest the following algorithm. Suppose, at the time t the CV of the first system has the value s1. First, we calculate the variance of bias potential for all points in configurational space. As was said earlier, our goal is to maximize the uncertainty. However, the following remark is very important. We want not only to maximize the uncertainty but we also want to prevent our system from going to the high-energy region. Therefore, we need to maximize the following function Q. After that we calculate the gradient of the Q with respect to s and move from point s1 to point s2, which will correspond in the following change of normal coordinates $\\Delta r$. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f2c8245",
   "metadata": {},
   "source": [
    "<img src=\"Picture8.png\" alt=\"\" width=\"800\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10ce8d08",
   "metadata": {},
   "source": [
    "To summarize the new approach, the algorithm for metadynamics with the use of adversarial attacks is presented below."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e3776525",
   "metadata": {},
   "source": [
    "<img src=\"Picture9.png\" alt=\"\" width=\"400\"/>"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "c123f2a6",
   "metadata": {},
   "source": [
    "### 1D potential well"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "db071636",
   "metadata": {},
   "source": [
    "For 1D double well, we start with three NVE trajectories with different initial position and velocities. Arbitrary units are chosen but to compare everything they are the same between metadynamics and adversarial attack. With adversarial attack, the bias potential became diffusive much earlier and thus we can sample faster and converge to free energy faster than actual metadynamics."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "352fb67c",
   "metadata": {},
   "source": [
    "<img src=\"Picture11.png\" alt=\"\" width=\"700\"/> "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3f916bd",
   "metadata": {},
   "source": [
    "### 2D potential well"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "46218caf",
   "metadata": {},
   "source": [
    "For any CV we give, adversarial attacks help us to achieve uniformity along the CV-direction much faster than metadynamics. \n",
    "Even for a ‘bad’ CV. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0c82fd1",
   "metadata": {},
   "source": [
    "<div align=\"center\"> Metadynamics\n",
    "<img src=\"md.gif\" alt=\"\" width=\"400\"/> "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15ccb8dd",
   "metadata": {},
   "source": [
    "<div align=\"center\"> Metadynamics + Adversarial attacks\n",
    "<img src=\"md adv.gif\" alt=\"\" width=\"400\"/> "
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "71c37b7d",
   "metadata": {},
   "source": [
    "As we mentioned before, the adversarial attacks ‘guide’ us in the CV-direction, while with metadynamics we have more disperse trajectory."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f15e60e6",
   "metadata": {},
   "source": [
    "<img src=\"Picture16.png\" alt=\"\" width=\"700\"/> "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b8e9052",
   "metadata": {},
   "source": [
    "Adversarial attacks, thus, help us to achieve the diffusive behavior in CV-direction much faster. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14bd6d4d",
   "metadata": {},
   "source": [
    "<img src=\"Picture17.png\" alt=\"\" width=\"700\"/> "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04b5a58c",
   "metadata": {},
   "source": [
    "### Alanine Dipeptide"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a6bcbe0",
   "metadata": {},
   "source": [
    "Alanine Dipeptide is a common case study for enhanced sampling. We took from paper w as 0.2kJ/mol and $\\sigma$ as $17^0$. We took $\\tau$ as 5fs with timestep of 0.5fs. The potential is a Schnet Neural force field trained on 160k dataset of varied range of torsion angles. A normal NVE run for 1ps was done to obtain atomic frames that can be compared with both methods. Three NVE ensembles were considered at different temp of 400 K, 500 K, 600 K.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "67342436",
   "metadata": {},
   "source": [
    "<img src=\"Picture18.png\" alt=\"\" width=\"700\"/> "
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "02f257dd",
   "metadata": {},
   "source": [
    "Each of the simulations were run for 50ps which is time consuming for NNFF. In one case, only metadynamics is used whereas in another adversarial attack is performed with a particular set of hyperparameters."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c55a9fd6",
   "metadata": {},
   "source": [
    "<img src=\"Picture19.png\" alt=\"\" width=\"700\"/> "
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "bff01d99",
   "metadata": {},
   "source": [
    "Preliminary jumps in adversarial attack help to get out of local regions and thus sample faster and more space in 50ps."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15995c65",
   "metadata": {},
   "source": [
    "<img src=\"Picture20.png\" alt=\"\" width=\"700\"/> "
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "3763d21c",
   "metadata": {},
   "source": [
    "### Summary\n",
    "1. Adversarial attacks along with metadynamics has an edge over normal metadynamics . \n",
    "2. It gets to diffusive region in collective variable much faster and thus can contribute to faster convergence of free energy surface.\n",
    "3. The hyper parameters perform intuitively and thus can be implemented by user’s discretion easily.\n",
    "4. Moving from toy models, the method even works for molecular system like alanine dipeptide. It traverses more collective variable space than in normal metadynamics.\n",
    "5. Since the adversarial attack serves as an extra push in the direction of change of collective variable, it achieves faster diffusion, which for any cv good or bad can lead to faster convergence. Therefore, we still need a good collective variable for sampling."
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "f3886e81",
   "metadata": {},
   "source": [
    "### References\n",
    "1. https://sites.google.com/site/giovannibussi/gallery#TOC-Metadynamics\n",
    "2. A. Laio and M. Parrinello, \"Escaping free-​energy minima.\"Proceedings of the National Academy of Sciences99.20 (2002): 12562-​12566.\n",
    "3. J. Phys. Chem. B 2018, 122, 21, 5508–5514.\n",
    "4. D. Schwalbe-Koda, A. R. Tan, and R. Gómez-Bombarelli, Differentiable Sampling of Molecular Geometries with Uncertainty-Based Adversarial Attacks, Nat. Commun. In press (2021).\n",
    "5. J. Phys. Chem. B 2010, 114, 16, 5632–5642"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
