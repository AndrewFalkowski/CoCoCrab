# CoCoCrab

Co-Optimization of Composition in CrabNet

## Premise

Traditional, data-driven materials discovery involves screening chemical systems with machine learning algorithms and selecting candidates that excel in a target property. The number of candidates to screen grows infinitely large as the number of included elements and the fractional resolution of compositions increases. The computational infeasibility and probability of overlooking a successful candidate grow likewise. Our approach shifts the optimization focus from model parameters to the fractions of each element in a composition. Using a pretrained network, CrabNet, and writing a custom loss function to govern a vector of element fractions, compositions can be optimized such that a predicted property is maximized or minimized withing a chemical system.

More information is available in the CoCoCrab manuscript here: https://doi.org/10.26434/chemrxiv.14680578.v1
