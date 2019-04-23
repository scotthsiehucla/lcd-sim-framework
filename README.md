# lcd-sim-framework
MATLAB simulation framework for estimating low contrast detectability (LCD) of many spheres simultaneously.

## ICD_detectability_rapid.m 

This file clears the workspace, initializes a phantom, reconstructs it multiple different ways, and analyzes the results with a model observer.

## ICD_simple.m 

This file performs a simple form of iterative reconstruction. It includes code for a basic forward projection, and uses built-in RADON and IRADON for an FBP reconstruction. The iterative reconstruction is done using iterative coordinate descent (ICD).
