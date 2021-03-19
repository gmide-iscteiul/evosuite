package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.Properties.SelectionFunction;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.LoggingUtils;
import org.evosuite.utils.Randomness;

/**
 * CatSwarmOptimization implementation
 *
 * @author
 */
public class CatSwarmOptimization<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	// private static final long serialVersionUID = 5043503777821916152L;
	private static final long serialVersionUID = -8855862003166456459L;
	private final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(CatSwarmOptimization.class);

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public CatSwarmOptimization(ChromosomeFactory<T> factory) {
		super(factory);

		if (Properties.SELECTION_FUNCTION != SelectionFunction.ROULETTEWHEEL) {
			LoggingUtils.getEvoLogger()
					.warn("Originally, Cat Swarm Optimization was implemented with a '"
							+ SelectionFunction.ROULETTEWHEEL.name()
							+ "' selection function. You may want to consider using it.");
		}
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();

		T bestCat = population.get(0);
		newGeneration.add(bestCat);

		double ratio;
		for (int i = 1; i < population.size(); i++) {
			ratio = Randomness.nextDouble();
			T cat = population.get(i);
			if (ratio < Properties.CROSSOVER_RATE) { // tracing mode
				try {
					crossoverFunction.crossOver(cat, bestCat.clone());
					if (cat.isChanged()) {
						cat.updateAge(currentIteration);
					}
				} catch (ConstructionFailedException e) {
					logger.info("Crossover/Mutation failed.");
				} finally {
					newGeneration.add(cat);
				}
			} else { // seeking mode
				int number_of_copies;
				List<T> list_of_copies = new ArrayList<>();
				if (Properties.SELF_POSITION_CONSIDERATION) { // current possition counts
					number_of_copies = Properties.SEEKING_MEMORY_POOL - 1;
					list_of_copies.add(cat);
				} else {
					number_of_copies = Properties.SEEKING_MEMORY_POOL;
				}
				for (int j = 0; j < number_of_copies; j++) {
					T copy = cat.clone();
					notifyMutation(copy);
					copy.mutate();
					list_of_copies.add(copy);
				}
				calculateFitnessAndSortPopulation(list_of_copies);
				T newGenCat = selectionFunction.select(list_of_copies);
				newGeneration.add(newGenCat);
				if (!newGenCat.equals(cat)) {
					newGenCat.updateAge(currentIteration);
				}
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