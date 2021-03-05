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
 * GeneticBeeAlgorithm implementation
 *
 * @author 
 */
public class GeneticBeeAlgorithm<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	//private static final long serialVersionUID = 5043503777821916152L;
	private static final long serialVersionUID = -8557609199714500045L;

	private final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(GeneticBeeAlgorithm.class);
		
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
		T neighbor_1 = population.get(Randomness.nextInt(getPopulationSize())).clone();
		T neighbor_2 = population.get(Randomness.nextInt(getPopulationSize())).clone();

		// Generation of children
		T child_1 = bee.clone();
		T child_2 = bee.clone();
		crossoverFunction.crossOver(child_1, neighbor_1);
		crossoverFunction.crossOver(child_2, neighbor_2);

		// Generation of grandchildren
		T grandchild_1 = child_1.clone();
		notifyMutation(grandchild_1);
		grandchild_1.mutate();

		T grandchild_2 = child_2.clone();
		notifyMutation(grandchild_2);
		grandchild_2.mutate();

		// Find best food source in the neighborhood
		List<T> foodSources = new ArrayList<>();
		foodSources.add(bee);
		foodSources.add(child_1);
		foodSources.add(child_2);
		foodSources.add(grandchild_1);
		foodSources.add(grandchild_2);
		calculateFitnessAndSortPopulation(foodSources); //added method to GA class

		T currentFood = foodSources.get(0);

		if (!currentFood.equals(bee)) { // if a better food source was found
			currentFood.updateAge(currentIteration);
		} else {
			currentFood.setDistance(currentFood.getDistance() + 1);
		}
		return currentFood;
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();

		// employee bee phase
		for (int i = 0; i < population.size(); i++) {
			T employee_bee = population.get(i);
			try {
				newGeneration.add(discoverNewFood(employee_bee));
			} catch (ConstructionFailedException e) {
				logger.info("Crossover/Mutation failed.");
				newGeneration.add(employee_bee);
				continue;
			}
		}

		// onlooker bee phase
		for (double i = 0; i < population.size()*Properties.ONLOOKER_BEE_RATE; i++) {
			T onlooker_bee = selectionFunction.select(newGeneration);
			try {
				newGeneration.remove(onlooker_bee);
				newGeneration.add(discoverNewFood(onlooker_bee));
			} catch (ConstructionFailedException e) {
				logger.info("Crossover/Mutation failed.");
				newGeneration.add(onlooker_bee);
				continue;
			}
		}

		// scout bee phase
		sortPopulation(newGeneration);

		for (int i = (population.size()-1), j=0; i > 0 && j< Properties.NUMBER_OF_SCOUTS; i--) {
			T scout_bee = newGeneration.get(i);
			if (scout_bee.getDistance() > Properties.LIMIT) {
				newGeneration.remove(scout_bee);
				T newFoodSource = chromosomeFactory.getChromosome();
				fitnessFunctions.forEach(newFoodSource::addFitness);
				newGeneration.add(newFoodSource);
				j++;
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
		if (Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER > 0
				|| Properties.ENABLE_SECONDARY_OBJECTIVE_STARVATION) {
			disableFirstSecondaryCriterion();
		}
		if (population.isEmpty())
			initializePopulation();

		logger.debug("Starting evolution");
		int starvationCounter = 0;
		double bestFitness = Double.MAX_VALUE;
		double lastBestFitness = Double.MAX_VALUE;
		if (getFitnessFunction().isMaximizationFunction()){
			bestFitness = 0.0;
			lastBestFitness = 0.0;
		} 
		//setSelectionFunction(new FitnessProportionateSelection<>()); //roullete wheel selection
		
		while (!isFinished()) {
			logger.debug("Current population: " + getAge() + "/" + Properties.SEARCH_BUDGET);
			logger.info("Best fitness: " + getBestIndividual().getFitness());
			evolve();
			// Determine fitness
			calculateFitnessAndSortPopulation();
			
		//////remove Local Search?
			//applyLocalSearch();	
			
			double newFitness = getBestIndividual().getFitness();

			if (getFitnessFunction().isMaximizationFunction())
				assert (newFitness >= bestFitness) : "best fitness was: " + bestFitness
						+ ", now best fitness is " + newFitness;
			else
				assert (newFitness <= bestFitness) : "best fitness was: " + bestFitness
						+ ", now best fitness is " + newFitness;
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