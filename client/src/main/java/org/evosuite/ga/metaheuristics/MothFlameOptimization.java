package org.evosuite.ga.metaheuristics;

import java.util.ArrayList;
import java.util.List;

import org.evosuite.Properties;
import org.evosuite.TimeController;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.utils.Randomness;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * MothFlameOptimization implementation
 *
 * @author
 */
public class MothFlameOptimization<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = 1918123998957131536L;
	private final Logger logger = LoggerFactory.getLogger(MothFlameOptimization.class);
	private List <T> bestSolutions = new ArrayList<>();

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public MothFlameOptimization(ChromosomeFactory<T> factory) {
		super(factory);
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>();
		// [Population Nº,1] adapted to all stopping conditions
		int numberOfFlames = (int) (-Properties.POPULATION * this.progress()) + Properties.POPULATION + 1;
		
		updateFlames(numberOfFlames);

		for (int i = 0; i < population.size(); i++) {
			T moth = population.get(i).clone();
			double r = -1 - this.progress();	// [-2,-1] adapted to all stopping conditions
			double t = Randomness.nextDouble(r, 1.0);	// [-2,1] adapted to all stopping conditions
			/*
			 * exploration -> mutation || exploitation -> crossover 
			 * 
			 */
			try {
				if (t < 0) {
					// crossover		
					T flame = bestSolutions.get((i * numberOfFlames) / population.size());
					crossoverFunction.crossOver(moth, flame.clone());
				} else {
					//mutation
					notifyMutation(moth);
					moth.mutate();
				}
				
				if (moth.isChanged()) {
					moth.updateAge(currentIteration);
				}
				newGeneration.add(moth);
			} catch (ConstructionFailedException e) {
				logger.info("Crossover/Mutation failed.");
				newGeneration.add(population.get(i));
			}
		}

		population = newGeneration;
		// archive
		updateFitnessFunctionsAndValues();
		//
		currentIteration++;
	}

	private void updateFlames(int numberOfFlames) {
		if (currentIteration == 0) {
			bestSolutions = population;
		} else {
			// add better solutions if found
			int size = bestSolutions.size();
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < population.size(); j++) {
					if (getFitnessFunction().isMaximizationFunction()) { // maximize ff
						if (bestSolutions.get(i).getFitness() < population.get(j).getFitness()) { 
							// better solution found
							bestSolutions.add(i, population.get(j));
						} else {
							break;
						}
					} else { // minimize ff
						if (bestSolutions.get(i).getFitness() > population.get(j).getFitness()) {
							// better solution found
							bestSolutions.add(i, population.get(j));
						} else {
							break;
						}
					}

				}
			}
			// remove excess solutions
			size = bestSolutions.size();
			for (int i = numberOfFlames; i < size; i++) {
				bestSolutions.remove(numberOfFlames);
			}
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
			// Determine fitness
			calculateFitnessAndSortPopulation();

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