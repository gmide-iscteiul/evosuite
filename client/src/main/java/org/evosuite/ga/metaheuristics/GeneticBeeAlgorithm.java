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
 * GeneticBeeAlgorithm implementation
 *
 * @author
 */
public class GeneticBeeAlgorithm<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = 8549163302884771117L;
	private final Logger logger = LoggerFactory.getLogger(GeneticBeeAlgorithm.class);

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public GeneticBeeAlgorithm(ChromosomeFactory<T> factory) {
		super(factory);

		if (Properties.SELECTION_FUNCTION != SelectionFunction.ROULETTEWHEEL) {
			LoggingUtils.getEvoLogger()
					.warn("Originally, Genetic Bee Algorithm was implemented with a '"
							+ SelectionFunction.ROULETTEWHEEL.name()
							+ "' selection function. You may want to consider using it.");
		}
	}

	private T discoverNewFood(T bee) throws ConstructionFailedException {
		// selection of 2 random neighbors
		T neighbor1 = population.get(Randomness.nextInt(getPopulationSize())).clone();
		T neighbor2 = population.get(Randomness.nextInt(getPopulationSize())).clone();

		// Generation of children
		T child1 = bee.clone();
		T child2 = bee.clone();
		crossoverFunction.crossOver(child1, neighbor1);
		crossoverFunction.crossOver(child2, neighbor2);

		// Generation of grandchildren
		T grandchild1 = child1.clone();
		notifyMutation(grandchild1);
		grandchild1.mutate();

		T grandchild2 = child2.clone();
		notifyMutation(grandchild2);
		grandchild2.mutate();

		// Find best food source in the neighborhood
		List<T> foodSources = new ArrayList<>();
		foodSources.add(bee);
		foodSources.add(child1);
		foodSources.add(child2);
		foodSources.add(grandchild1);
		foodSources.add(grandchild2);
		calculateFitnessAndSortPopulation(foodSources); // added method to GA class

		T currentFood = foodSources.get(0);

		if (currentFood != bee) { // if a better food source was found
			currentFood.updateAge(currentIteration);
		}
		return currentFood;
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();

		// employee bee phase
		for (int i = 0; i < population.size(); i++) {
			T employeeBee = population.get(i);
			try {
				newGeneration.add(discoverNewFood(employeeBee));
			} catch (ConstructionFailedException e) {
				logger.info("Crossover/Mutation failed.");
			} finally {
				newGeneration.add(employeeBee);
			}
		}

		// onlooker bee phase
		for (int i = 0; i < (int) (population.size() * Properties.ONLOOKER_BEE_RATE); i++) {
			T onlookerBee = selectionFunction.select(newGeneration);
			try {
				T newBee = discoverNewFood(onlookerBee);
				if (newBee != onlookerBee) {
					newGeneration.remove(onlookerBee);
					newGeneration.add(newBee);
				}
			} catch (ConstructionFailedException e) {
				logger.info("Crossover/Mutation failed.");
			} finally {
				newGeneration.add(onlookerBee);
			}
		}

		// scout bee phase
		sortPopulation(newGeneration);

		boolean addNewBee = false;
		int currentScouts = 0;
		for (int i = (population.size() - 1); i >= 0; i--) {
			T scoutBee = newGeneration.get(i);
			if (currentIteration - scoutBee.getAge() > Properties.MAX_NUM_ITERATIONS_WITHOUT_IMPROVEMENT) { // replace scout bee with new individual
				newGeneration.remove(scoutBee);
				T newFoodSource = chromosomeFactory.getChromosome();
				fitnessFunctions.forEach(newFoodSource::addFitness);
				newFoodSource.updateAge(currentIteration);
				calculateFitness(newFoodSource);
				newGeneration.add(newFoodSource);
				currentScouts++;
				addNewBee = true;
				if (currentScouts >= Properties.NUMBER_OF_SCOUTS) {
					break;
				}
			}
		}

		population = newGeneration;
		// archive
		updateFitnessFunctionsAndValues();
		//
		currentIteration++;

		if (addNewBee) {
			sortPopulation();
			addNewBee = false;
		}
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

			// Local Search
			// applyLocalSearch();

//			double newFitness = getBestIndividual().getFitness();
//
//			if (getFitnessFunction().isMaximizationFunction())
//				assert (newFitness >= bestFitness)
//						: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
//			else
//				assert (newFitness <= bestFitness)
//						: "best fitness was: " + bestFitness + ", now best fitness is " + newFitness;
//			bestFitness = newFitness;
//
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