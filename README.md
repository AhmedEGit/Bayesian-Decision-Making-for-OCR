# Bayesian-Decision-Making-for-OCR

# Project Overview

This project implements an Optical Character Recognition (OCR) system based on Bayesian Decision Making. The OCR system is designed to classify 10×10 pixel image chips containing two characters: "A" and "C" using a Bayesian decision framework. The system is trained and evaluated on two cases:

- Discrete Measurements: Features are discrete values, and a probabilistic decision rule is applied.
- Continuous Measurements: Features are modeled as continuous values following a normal distribution.

The objective is to classify image chips with minimal error using Bayesian inference and decision theory while evaluating the classification performance under different settings.

# Key Concepts

- Bayesian Decision Theory: A statistical framework for making optimal decisions under uncertainty by combining prior knowledge with observed data to minimize decision risk.
- OCR (Optical Character Recognition): A method for converting images of text into machine-readable characters.
- Bayesian Risk: A metric to quantify the cost of making incorrect decisions under uncertain information.
- Loss Function: A measure of the cost associated with classification errors. The project uses the "zero-one loss" for simplifying decision-making.

# Features

 Discrete Measurements Case:
    - Feature Extraction: Computes a discrete feature, x, based on the difference between pixel intensities on the left and right halves of the image, normalized to the range [−10,10].
    - Classification Strategy: Uses a 21-element vector to represent the optimal decision q(x), where 0 indicates "A" and 1 indicates "C."
    - Bayesian Risk: Calculates the Bayesian risk using the zero-one loss matrix and evaluates performance.

Continuous Measurements Case:
    - Feature Extraction: Computes a continuous feature, x, based on the unnormalized pixel intensity difference between the left and right halves of the image.
    - Normal Distribution Assumption: Assumes that features for classes "A" and "C" follow normal distributions with given means, variances, and prior probabilities.
    - Optimal Decision Rule: Derives decision thresholds by solving a quadratic inequality, resulting in up to two classification thresholds.
    - Bayesian Risk: Minimizes classification error based on continuous feature values.

Optimal Decision Strategy:
        Derives and applies the optimal Bayesian strategy to minimize misclassification risk.
        Experimentation with the loss function to evaluate its effect on the decision-making process.
