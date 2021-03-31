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
 * CatSwarmOptimization implementation
 *
 * @author
 */
public class CatSwarmOptimization<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = -8855862003166456459L;
	private final Logger logger = LoggerFactory.getLogger(CatSwarmOptimization.class);

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
		List<T> newGeneration = new ArrayList<>(elitism());

		T bestCat = population.get(0);

		for (int i = newGeneration.size(); i < population.size(); i++) {
			double ratio = Randomness.nextDouble();
			T cat = population.get(i).clone();
			if (ratio < Properties.CROSSOVER_RATE) { // tracing mode
				try {
					crossoverFunction.crossOver(cat, bestCat.clone());
					if (cat.isChanged()) {
						cat.updateAge(currentIteration);
					}
				} catch (ConstructionFailedException e) {
					logger.info("Crossover/Mutation failed.");
				} finally {
					newGeneration.add(population.get(i));
				}
			} else { // seeking mode
				int numberOfCopies;
				List<T> listOfCopies = new ArrayList<>();
				if (Properties.SELF_POSITION_CONSIDERATION) { // current possition counts
					numberOfCopies = Properties.SEEKING_MEMORY_POOL - 1;
					listOfCopies.add(cat);
				} else {
					numberOfCopies = Properties.SEEKING_MEMORY_POOL;
				}
				for (int j = 0; j < numberOfCopies; j++) {
					T copy = cat.clone();
					notifyMutation(copy);
					copy.mutate();
					listOfCopies.add(copy);
				}
				calculateFitnessAndSortPopulation(listOfCopies);
				T newGenCat = selectionFunction.select(listOfCopies);
				newGeneration.add(newGenCat);
				if (newGenCat != cat) {
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