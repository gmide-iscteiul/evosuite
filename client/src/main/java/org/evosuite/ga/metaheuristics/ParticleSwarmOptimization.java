package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.TimeController;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ParticleSwarmOptimization implementation
 *
 * @author
 */
public class ParticleSwarmOptimization<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = -7883265779642291283L;
	private final Logger logger = LoggerFactory.getLogger(ParticleSwarmOptimization.class);
	List<T> localMemory = new ArrayList<>();

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public ParticleSwarmOptimization(ChromosomeFactory<T> factory) {
		super(factory);
	}

	private T bestParticle(T particle1, T particle2) {
		if (getFitnessFunction().isMaximizationFunction()) {
			return (particle1.getFitness() > particle2.getFitness()) ? particle1 : particle2;
		} else {
			return (particle1.getFitness() < particle2.getFitness()) ? particle1 : particle2;
		}
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>(elitism());
		T bestGlobal = population.get(0);
		for (int i = newGeneration.size(); i < population.size(); i++) {
			T particle = population.get(i).clone();
			// update using local memory as reference
			if (population.get(i) != localMemory.get(i)) {
				T bestLocal = localMemory.get(i).clone();
				try {
					crossoverFunction.crossOver(particle, bestLocal);
				} catch (ConstructionFailedException e) {
					logger.info("Crossover failed.");
					particle = population.get(i);
				}
			}
			// update using global memory as reference
			try {
				crossoverFunction.crossOver(particle, bestGlobal.clone());
				notifyMutation(particle);
				particle.mutate();
			} catch (ConstructionFailedException e) {
				logger.info("Crossover failed.");
				particle = population.get(i);
			}
		}

		// update local memory
		for (int i = 0; i < newGeneration.size(); i++) {
			T bestParticle = bestParticle(newGeneration.get(i), localMemory.get(i));
			if (bestParticle != localMemory.get(i)) {
				localMemory.set(i, bestParticle);
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

		localMemory = population;

		this.notifyIteration();
	}

	/** {@inheritDoc} */
	@Override
	public void generateSolution() {
		if (Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER > 0 || Properties.ENABLE_SECONDARY_OBJECTIVE_STARVATION) {
			disableFirstSecondaryCriterion();
		}
		if (population.isEmpty()) {
			initializePopulation();
		}

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

			calculateFitnessAndSortPopulation();

			// Local Search
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

		TimeController.execute(this::updateBestIndividualFromArchive, "update from archive", 5_000);
		notifySearchFinished();
	}
}