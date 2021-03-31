package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.Randomness;

/**
 * WhaleOptimizationAlgorithm implementation
 *
 * @author
 */
public class WhaleOptimizationAlgorithm<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = 1178004502227853389L;
	private final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(WhaleOptimizationAlgorithm.class);

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public WhaleOptimizationAlgorithm(ChromosomeFactory<T> factory) {
		super(factory);
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>(elitism());

		double a = 2 * (1.0 - this.progress()); // [2,0] adapted to all stopping conditions

		T bestWhale = population.get(0);
		
		for (int i = newGeneration.size(); i < population.size(); i++) {
			T whale = population.get(i);

			/*
			 * A- exploration -> crossover with random whale || exploitation -> crossover
			 * with best whale 
			 * C- mutation 
			 * Spiral movement -> mutation
			 */
			double A = 2 * a * Randomness.nextDouble() - a;// Equation (2.3)
			double C = 2 * Randomness.nextDouble(); // Equation (2.4)
			double p = Randomness.nextDouble();
			try {
				if (p < Properties.CROSSOVER_RATE) {
					if (Math.abs(A) < 2 * Properties.SHRINKING_ENCIRCLING_MECHANISM_RATE) {	//shrinking encircling mechanism
						crossoverFunction.crossOver(whale, bestWhale.clone());
					} else { // search for prey
						T randomWhale = population.get(Randomness.nextInt(population.size()));
						crossoverFunction.crossOver(whale, randomWhale.clone());
					}
					if (C <= 2 * Properties.MUTATION_RATE) { // both depend on C for randomness
						notifyMutation(whale);
						whale.mutate();
					}
				} else { // spiral updating position
					notifyMutation(whale);
					whale.mutate();
				}
				if (whale.isChanged()) {
					whale.updateAge(currentIteration);
				}
			} catch (ConstructionFailedException e) {
				logger.info("Crossover/Mutation failed.");
			} finally {
				newGeneration.add(whale);
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

			// remove Local Search
			// applyLocalSearch();

			double newFitness = getBestIndividual().getFitness();

			if (Properties.ELITE > 0) {
				if (getFitnessFunction().isMaximizationFunction())
					assert (newFitness >= bestFitness)
							: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
				else
					assert (newFitness <= bestFitness)
							: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
				bestFitness = newFitness;
			}

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