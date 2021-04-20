package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.TimeController;
import org.evosuite.Properties.SelectionFunction;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.LoggingUtils;
import org.evosuite.utils.Randomness;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ArtificialAlgaeAlgorithm implementation
 *
 * @author
 */
public class ArtificialAlgaeAlgorithm<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = 4982755939858913909L;
	private final Logger logger = LoggerFactory.getLogger(ArtificialAlgaeAlgorithm.class);

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public ArtificialAlgaeAlgorithm(ChromosomeFactory<T> factory) {
		super(factory);

		if (Properties.SELECTION_FUNCTION != SelectionFunction.TOURNAMENT) {
			LoggingUtils.getEvoLogger().warn("Originally, Artificial Algae Algorithm was implemented with a '"
					+ SelectionFunction.TOURNAMENT.name() + "' selection function. You may want to consider using it.");
		}
	}

	private T bestAlgae(T algae1, T algae2) {
		if (getFitnessFunction().isMaximizationFunction()) {
			return (algae1.getFitness() > algae2.getFitness()) ? algae1 : algae2;
		} else {
			return (algae1.getFitness() < algae2.getFitness()) ? algae1 : algae2;
		}
	}

	private T getOldestAlgae(List<T> list) {
		T oldestAlgae = list.get(0);
		for (int i = 1; i < list.size(); i++) {
			if (oldestAlgae.getAge() >= list.get(i).getAge()) {
				oldestAlgae = list.get(i);
			}
		}
		return oldestAlgae;
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>(elitism());

		boolean starvation = true;
		for (int i = newGeneration.size(); i < population.size(); i++) {
			// energy = directly proportional to position;
			double energy = Properties.MAX_INITIAL_ENERGY - i;
			while (energy > 0) {
				energy -= (Properties.ENERGY_LOSS_RATE * Properties.MAX_INITIAL_ENERGY);

				T algae = population.get(i).clone();
				T selectedAlgae = selectionFunction.select(population).clone();

				try {
					crossoverFunction.crossOver(algae, selectedAlgae);
					calculateFitness(algae);
				} catch (ConstructionFailedException e) {
					logger.info("Crossover failed.");
					algae = population.get(i);
				}

				T newAlgae = bestAlgae(algae, population.get(i));
				if (newAlgae == algae) { // new solution is better
					if (newAlgae.isChanged()) {
						newAlgae.updateAge(currentIteration);
					}
					newGeneration.add(newAlgae);
					starvation = false;
					break; // makes sense to me!!!!
				} else {
					energy -= (Properties.ENERGY_LOSS_RATE * Properties.MAX_INITIAL_ENERGY);
				}
			}
			if (starvation) {
				newGeneration.add(population.get(i)); // manintain population number
			}
		}

		sortPopulation(newGeneration);
		T bestAlgae = newGeneration.get(0).clone();
		T worstAlgae = newGeneration.get(newGeneration.size() - 1);
		try {
			crossoverFunction.crossOver(worstAlgae, bestAlgae);
		} catch (ConstructionFailedException e) {
			logger.info("Crossover failed.");
			worstAlgae = newGeneration.get(newGeneration.size() - 1);
		}

		calculateFitness(worstAlgae); // all individuals with fitness calculated

		if (Randomness.nextDouble() < Properties.ADAPTATION_RATE) {
			bestAlgae = newGeneration.get(0).clone();
			T oldestAlgae = getOldestAlgae(newGeneration);
			try {
				crossoverFunction.crossOver(oldestAlgae, bestAlgae);
			} catch (ConstructionFailedException e) {
				logger.info("Crossover failed.");
				oldestAlgae = getOldestAlgae(newGeneration);
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
			// Determine fitness
			sortPopulation();

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