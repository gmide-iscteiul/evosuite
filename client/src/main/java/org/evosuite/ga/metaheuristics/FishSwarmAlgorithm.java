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
 * FishSwarmAlgorithm implementation
 *
 * @author
 */
public class FishSwarmAlgorithm<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

	private static final long serialVersionUID = -4220392757546401915L;
	private final Logger logger = LoggerFactory.getLogger(FishSwarmAlgorithm.class);
	int indexMiddle = Properties.POPULATION / 2;

	/**
	 * Constructor
	 *
	 * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
	 */
	public FishSwarmAlgorithm(ChromosomeFactory<T> factory) {
		super(factory);
	}

	private T SwarmPhase(T fish, int indexNumber) {
		T middleFish;

		if (indexNumber != indexMiddle) {
			middleFish = population.get(indexMiddle).clone();
		} else {
			middleFish = population.get(indexMiddle + 1).clone();
		}
		try {
			crossoverFunction.crossOver(fish, middleFish);
			calculateFitness(fish);

			T newFish = bestFish(fish, population.get(indexNumber));
			if (newFish != fish) {
				return preyPhase(newFish, indexNumber);
			}
		} catch (ConstructionFailedException e) {
			logger.info("Crossover failed.");
			fish = population.get(indexNumber);
		}
		return fish;
	}

	private T FollowPhase(T fish, int indexNumber) {
		T bestFish;
		if (indexNumber != 0) {
			bestFish = population.get(0).clone();
		} else {
			bestFish = population.get(1).clone();
		}
		try {
			crossoverFunction.crossOver(fish, bestFish);
			calculateFitness(fish);

			T newFish = bestFish(fish, population.get(indexNumber));
			if (newFish != fish) {
				return preyPhase(newFish, indexNumber);
			}
		} catch (ConstructionFailedException e) {
			logger.info("Crossover failed.");
			fish = population.get(indexNumber);
		}
		return fish;
	}

	private T preyPhase(T fish, int indexNumber) {

		for (int i = 0; i < Properties.NUMBER_OF_ATTEMPTS; i++) {
			int randomNumber = Randomness.nextInt(population.size());

			if (randomNumber < indexNumber) {
				T randomFish = population.get(randomNumber).clone();
				try {
					crossoverFunction.crossOver(fish, randomFish);
				} catch (ConstructionFailedException e) {
					logger.info("Crossover failed.");
					fish = population.get(indexNumber);
				}
				return fish;
			}
		}
		return randomPhase(fish);
	}

	private T randomPhase(T fish) {
		notifyMutation(fish);
		fish.mutate();
		return fish;
	}

	private T bestFish(T fish1, T fish2) {
		if (getFitnessFunction().isMaximizationFunction()) {
			if (fish1.getFitness() > fish2.getFitness()) {
				return fish1;
			} else {
				return fish2;
			}
		} else {
			if (fish1.getFitness() < fish2.getFitness()) {
				return fish1;
			} else {
				return fish2;
			}
		}
	}

	/** {@inheritDoc} */
	@Override
	protected void evolve() {
		List<T> newGeneration = new ArrayList<>(elitism());

		for (int i = newGeneration.size(); i < population.size(); i++) {
			T oldFish = population.get(i);
			T fish1 = SwarmPhase(oldFish, i);
			T fish2 = FollowPhase(oldFish, i);

			// bestBehaviourPhase
			T newFish = bestFish(fish1, fish2);
			if (newFish.isChanged()) {
				newFish.updateAge(currentIteration);
			}
			newGeneration.add(newFish);
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