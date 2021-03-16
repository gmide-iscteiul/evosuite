package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.Randomness;

/**
 * GreyWolfOptimizer implementation
 *
 * @author
 */
public class GreyWolfOptimizer<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = -8811115659916973474L;
	// private static final long serialVersionUID = 5043503777821916152L;

	private final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(GreyWolfOptimizer.class);

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public GreyWolfOptimizer(ChromosomeFactory<T> factory) {
		super(factory);
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();

		double a = 2 * (1.0 - this.progress()); // [2,0] adapted to all stopping conditions

		T alpha = population.get(0).clone();
		T beta = population.get(1).clone();
		T delta = population.get(2).clone();

		newGeneration.add(alpha);
		newGeneration.add(beta);
		newGeneration.add(delta);

		for (int i = 3; i < population.size(); i++) {
			T wolf = population.get(i).clone();

			/*
			 * A- exploration -> mutation || exploitation -> crossover 
			 * C- emphasizes mutation
			 * 
			 */
			double A = 2 * a * Randomness.nextDouble() - a;// Equation (3.3)
			double C = 2 * Randomness.nextDouble(); // Equation (3.4)
			try {
				if (Math.abs(A) < 2 * Properties.CROSSOVER_RATE) {
					// crossover
					crossoverFunction.crossOver(wolf, alpha.clone());
					crossoverFunction.crossOver(wolf, beta.clone());
					crossoverFunction.crossOver(wolf, delta.clone());
				}
				if (Math.abs(A) >= 2 * Properties.CROSSOVER_RATE || C <= 2 * Properties.MUTATION_RATE) {
					// mutation
					notifyMutation(wolf);
					wolf.mutate();
				}
				if (wolf.isChanged()) {
					wolf.updateAge(currentIteration);
				}
			} catch (ConstructionFailedException e) {
				logger.info("Crossover/Mutation failed.");
			} finally {
				newGeneration.add(wolf);
			}
		}
		population = newGeneration;
		// archive
		updateFitnessFunctionsAndValues();
		//
		currentIteration++;
	}

	/** {@inheritDoc} */
	@Override
	public void initializePopulation() {
		notifySearchStarted();
		currentIteration = 0;

		// Set up initial population
		generateInitialPopulation(Properties.POPULATION);
		// Determine fitness
		calculateFitnessAndSortPopulation();

		this.notifyIteration();
	}

	/** {@inheritDoc} */
	@Override
	public void generateSolution() {
		if (Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER > 0 || Properties.ENABLE_SECONDARY_OBJECTIVE_STARVATION) {
			disableFirstSecondaryCriterion();
		}
		if (population.isEmpty())
			initializePopulation();

		logger.debug("Starting evolution");
		int starvationCounter = 0;
		double bestFitness = Double.MAX_VALUE;
		double lastBestFitness = Double.MAX_VALUE;
		if (getFitnessFunction().isMaximizationFunction()) {
			bestFitness = 0.0;
			lastBestFitness = 0.0;
		}

		while (!isFinished()) {
			logger.debug("Current population: " + getAge() + "/" + Properties.SEARCH_BUDGET);
			logger.info("Best fitness: " + getBestIndividual().getFitness());

			evolve();
			// Determine fitness
			calculateFitnessAndSortPopulation();

			////// remove Local Search?
			// applyLocalSearch();

			double newFitness = getBestIndividual().getFitness();

			if (getFitnessFunction().isMaximizationFunction())
				assert (newFitness >= bestFitness)
						: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
			else
				assert (newFitness <= bestFitness)
						: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
			bestFitness = newFitness;

			if (Double.compare(bestFitness, lastBestFitness) == 0) {
				starvationCounter++;
			} else {
				logger.info("reset starvationCounter after " + starvationCounter + " iterations");
				starvationCounter = 0;
				lastBestFitness = bestFitness;

			}

			updateSecondaryCriterion(starvationCounter);

			this.notifyIteration();
		}

		updateBestIndividualFromArchive();
		notifySearchFinished();
	}
}